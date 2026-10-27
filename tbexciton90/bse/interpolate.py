"""Wannier interpolation: build a WannierSetup on any dense k-grid from H(R).

Given a coarse-grid WannierSetup (populated from Wannier90 output files),
construct a new WannierSetup on an arbitrary dense N1 x N2 x N3 grid by:
    H(k_fine) = Σ_R  H(R)/degen(R) · exp(i k_fine · R)
followed by k-by-k diagonalisation.  The resulting eigenvalues are the
Wannier-interpolated DFT bands; the eigenvectors are the Bloch->Wannier
coefficients needed by the BSE builder.

The interpolation is exact at the coarse-grid k-points that generated H(R),
and smoothly interpolates elsewhere with a spread controlled by the coarse
grid density.
"""
from __future__ import annotations
from typing import Tuple
from dataclasses import replace
import numpy as np

from .wannier_setup import WannierSetup


def _kgrid_fine(mp_grid: Tuple[int, int, int]) -> np.ndarray:
    """Uniform Monkhorst-Pack-like fractional k-grid, k in [0, 1)."""
    n1, n2, n3 = mp_grid
    kx = np.arange(n1) / n1
    ky = np.arange(n2) / n2
    kz = np.arange(max(n3, 1)) / max(n3, 1)
    K1, K2, K3 = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.stack([K1.ravel(), K2.ravel(), K3.ravel()], axis=1)


def _supercell_R_lattice(mp_grid: Tuple[int, int, int],
                         lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real-space lattice R = n1*a1 + n2*a2 + n3*a3 with
       n_i in [-N_i/2, ceil(N_i/2)), matching an N1 x N2 x N3 k-grid.

    This is the R-lattice on which V_scr(R) must be tabulated for the
    Fourier sum sum_R V(R) exp(i (k-k') · R) to be alias-free at the
    (Q=0) BSE matrix elements on that k-grid.

    Returns
    -------
    R_frac_int : (N_R, 3) integer fractional lattice indices
    R_cart     : (N_R, 3) cartesian lattice vectors (Å)
    """
    ranges = []
    for n in mp_grid:
        lo = -(n // 2)
        hi = n + lo         # ranges of n // 2 or (n//2)+1 depending on parity
        ranges.append(np.arange(lo, hi))
    N1, N2, N3 = np.meshgrid(*ranges, indexing="ij")
    R_frac = np.stack([N1.ravel(), N2.ravel(), N3.ravel()], axis=1)
    R_cart = R_frac @ lattice
    return R_frac.astype(int), R_cart


def interpolate_at_kpoints(coarse: WannierSetup,
                           kfrac_target: np.ndarray,
                           vb_bloch_range: Tuple[int, int],
                           cb_bloch_range: Tuple[int, int]
                           ) -> dict:
    """Wannier-interpolated bands and coefficients at arbitrary fractional k-points.

    Same interpolation as `interpolate_to_fine_grid` but for an arbitrary set
    of k-points (not necessarily a regular grid).  Used by `build_bse_hamiltonian`
    to evaluate quantities at k+Q for finite-Q exciton calculations.

    Returns a dict with keys `E_vk`, `E_ck`, `C_vb`, `C_cb` — the sliced VB/CB
    energies and Bloch->Wannier coefficients on the requested k-points.
    """
    R_cart = coarse.R_cart
    degen = coarse.R_degen.astype(float)
    H_r = coarse.H_r
    n_wann = coarse.n_wann

    B = 2.0 * np.pi * np.linalg.inv(coarse.lattice.T)
    kcart = kfrac_target @ B
    Nk = kfrac_target.shape[0]

    phase = np.exp(1j * np.einsum("Ra,ka->Rk", R_cart, kcart)) / degen[:, None]
    Hk = np.einsum("Rk,Rmn->mnk", phase, H_r)

    E_all = np.zeros((n_wann, Nk), dtype=float)
    C_full = np.zeros((n_wann, n_wann, Nk), dtype=complex)
    for ik in range(Nk):
        Hkk = 0.5 * (Hk[:, :, ik] + Hk[:, :, ik].conj().T)
        w, v = np.linalg.eigh(Hkk)
        E_all[:, ik] = w
        C_full[:, :, ik] = v

    v0, v1 = vb_bloch_range
    c0, c1 = cb_bloch_range
    return dict(
        E_vk=E_all[v0:v1, :],
        E_ck=E_all[c0:c1, :],
        C_vb=C_full[:, v0:v1, :],
        C_cb=C_full[:, c0:c1, :],
    )


def interpolate_to_fine_grid(coarse: WannierSetup,
                             mp_grid_fine: Tuple[int, int, int],
                             vb_bloch_range: Tuple[int, int],
                             cb_bloch_range: Tuple[int, int]
                             ) -> WannierSetup:
    """Return a new WannierSetup on the requested dense grid.

    Parameters
    ----------
    coarse
        The WannierSetup built from Wannier90 files (contains H(R)).
    mp_grid_fine
        (N1, N2, N3) dense k-grid.
    vb_bloch_range, cb_bloch_range
        (start, stop) slices of the Wannier-interpolated bands that
        should be treated as VB and CB in the new setup.  After
        interpolation there are exactly `n_wann` bands per k; the
        indices are into that sorted-ascending band list.
    """
    n_wann = coarse.n_wann
    R_cart = coarse.R_cart                       # (N_R, 3), A
    R_frac_int = coarse.R_frac                   # (N_R, 3), integer
    degen = coarse.R_degen.astype(float)         # (N_R,)
    H_r = coarse.H_r                             # (N_R, n_wann, n_wann), eV

    # ---- Fine k-grid (fractional and cartesian) ----
    kfrac = _kgrid_fine(mp_grid_fine)            # (Nk_fine, 3)
    B = 2.0 * np.pi * np.linalg.inv(coarse.lattice.T)
    kcart = kfrac @ B                            # (Nk_fine, 3), 1/A

    Nk = kfrac.shape[0]

    # ---- H(k) via lattice-vector FT of H(R) ----
    # phase[R, k] = exp(i k · R_cart) / degen[R]
    phase = np.exp(1j * np.einsum("Ra,ka->Rk", R_cart, kcart)) / degen[:, None]
    # H_k[n_wann, n_wann, Nk] = sum_R phase[R,k] * H_R[R, n_wann, n_wann]
    Hk = np.einsum("Rk,Rmn->mnk", phase, H_r)    # (n_wann, n_wann, Nk)

    # ---- Diagonalise per k ----
    E_all = np.zeros((n_wann, Nk), dtype=float)
    C_full = np.zeros((n_wann, n_wann, Nk), dtype=complex)     # (orbital, band, k)
    for ik in range(Nk):
        Hkk = 0.5 * (Hk[:, :, ik] + Hk[:, :, ik].conj().T)
        w, v = np.linalg.eigh(Hkk)
        E_all[:, ik] = w
        # v[:, n] is the eigenvector for band n; treat rows as orbitals.
        # Convention used elsewhere: C[orb, band, k]
        C_full[:, :, ik] = v

    # ---- Slice VB and CB ----
    v0, v1 = vb_bloch_range
    c0, c1 = cb_bloch_range
    E_vk = E_all[v0:v1, :]
    E_ck = E_all[c0:c1, :]
    C_vb = C_full[:, v0:v1, :]
    C_cb = C_full[:, c0:c1, :]

    # ---- Regenerate the BSE R-lattice to match the fine k-grid ----
    # Wannier90's WS-cell R-vectors (in coarse.R_cart) are only correct
    # for the k-grid on which they were generated.  A denser BSE k-grid
    # requires a denser R-lattice or the Fourier sum aliases (severe
    # over-binding).  We regenerate R as a plain supercell of size mp_grid.
    R_frac_new, R_cart_new = _supercell_R_lattice(mp_grid_fine, coarse.lattice)
    degen_new = np.ones(R_frac_new.shape[0], dtype=int)

    # ---- Assemble new WannierSetup on the fine grid ----
    fine = replace(
        coarse,
        mp_grid=mp_grid_fine,
        kpoints_frac=kfrac,
        kpoints_cart=kcart,
        E_ck=E_ck,
        E_vk=E_vk,
        C_vb=C_vb,
        C_cb=C_cb,
        R_frac=R_frac_new,
        R_cart=R_cart_new,
        R_degen=degen_new,
        # H_r, U, atom_groups, lattice, n_wann, wannier_centres unchanged
    )
    return fine
