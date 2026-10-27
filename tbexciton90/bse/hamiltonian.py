"""Bethe-Salpeter Hamiltonian in a Wannier basis.

Follows the atomic-orbital construction used in PyMEX
(Maity, Mostofi, Lischner, npj 2D Mater. Appl. 9, 20 (2025), Eq. 5):

    H_{cvk, c'v'k'} = (E_ck - E_vk) delta_{cc'} delta_{vv'} delta_{k,k'}
        + K_dir(cvk, c'v'k')
        + K_exch(cvk, c'v'k')

with

    K_dir  =  (1/N_R) sum_{i,j orbitals} sum_R  V^screened(R + t_i - t_j)
             * exp[i (k - k').R]  *  M^e_ij(cc', k'k)  M^h_ij(vv', k'k)

    K_exch = -(1/N_R) sum_{i,j orbitals} sum_R  V^bare(R + t_i - t_j)
             *  M^{exch}_ij(cv, k)  M^{exch,*}_ij(c'v', k')

where the electron/hole/exchange orbital-density matrices are, per the
PyMEX cyfunc.pyx convention:

    M^e_ij(cc', k'k)  =  conj(C_{i, c, k})  *  C_{i, c', k'}
    M^h_ij(vv', k'k)  =  C_{j, v, k}   *  conj(C_{j, v', k'})
    M^{exch}_ij(cv, k) = conj(C_{i, c, k}) * C_{j, v, k}

V^screened and V^bare are 2D screened/bare interactions evaluated in real
space at inter-orbital separation r = |R + t_i - t_j|.

Backend:
    - numpy on CPU by default.
    - cupy on GPU (L40S) via prefer_gpu=True.  The heavy einsum runs on GPU.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import time

from .wannier_setup import WannierSetup
from ..potentials.base import Potential
from ..constants import EV_A_COULOMB
from ..backends import get_backend, to_numpy


# ----------------------------------------------------------------------
# Real-space potential tabulation
# ----------------------------------------------------------------------
def tabulate_Vr(setup: WannierSetup,
                potential: Potential) -> np.ndarray:
    """V_r[R, i, j] = potential.V_r(|R + t_i - t_j|) with i,j = atomic sites.

    In-plane distance is used (2D materials); the z-component of the
    Wannier centre position is ignored, matching PyMEX's convention.
    """
    R = setup.R_cart                          # (N_R, 3)
    n_at = len(setup.atom_groups)
    positions = np.array([g[3] for g in setup.atom_groups])   # (n_at, 3)

    # Pairwise vectors R + t_i - t_j, shape (N_R, n_at, n_at, 3)
    dvec = (R[:, None, None, :] +
            positions[None, :, None, :] -
            positions[None, None, :, :])
    # In-plane distance
    r_inplane = np.sqrt(dvec[..., 0]**2 + dvec[..., 1]**2)
    return potential.V_r(r_inplane)           # (N_R, n_at, n_at)


def bare_Coulomb_Vr(setup: WannierSetup,
                    eps_env: float = 1.0,
                    a0_reg: float = 3.32) -> np.ndarray:
    """Unscreened 2D Coulomb tabulated on the same (R, i, j) grid.

    Regularised at r < a0_reg by clamping to r = a0_reg.
    """
    R = setup.R_cart
    positions = np.array([g[3] for g in setup.atom_groups])
    dvec = (R[:, None, None, :] +
            positions[None, :, None, :] -
            positions[None, None, :, :])
    r_inplane = np.sqrt(dvec[..., 0]**2 + dvec[..., 1]**2)
    r_reg = np.where(r_inplane < a0_reg, a0_reg, r_inplane)
    return -EV_A_COULOMB / (eps_env * r_reg)


# ----------------------------------------------------------------------
# BSE Hamiltonian assembly (vectorised, works CPU or GPU)
# ----------------------------------------------------------------------
def build_bse_hamiltonian(setup: WannierSetup,
                          screened_potential: Potential,
                          eps_env_exchange: float = 1.0,
                          include_exchange: bool = True,
                          prefer_gpu: bool = False,
                          verbose: bool = False,
                          Q_frac: "np.ndarray | None" = None,
                          coarse_setup: "WannierSetup | None" = None) -> np.ndarray:
    """Assemble the full BSE Hamiltonian and return as a dense numpy array.

    Shape returned: (Nc * Nv * Nk, Nc * Nv * Nk), complex128.

    Finite-Q excitons
    -----------------
    By default (Q_frac=None or (0,0,0)) this builds the Q=0 BSE.  For an
    exciton with centre-of-mass momentum Q ≠ 0 the electron sits at k+Q
    while the hole sits at k; pass

        Q_frac       :  length-3 fractional (crystal) coordinates of Q
        coarse_setup :  the WannierSetup returned by build_setup() (contains
                        H(R) needed to Wannier-interpolate bands at k+Q).

    The routine then evaluates E_c and the Bloch->Wannier coefficients C_cb
    on the shifted grid k+Q via `interpolate_at_kpoints`, and keeps E_v /
    C_vb on the original k-grid.
    """
    xp, on_gpu = get_backend(prefer_gpu=prefer_gpu)
    if verbose:
        print(f"  backend: {'cupy (GPU)' if on_gpu else 'numpy (CPU)'}")

    # Tabulate real-space potentials  (small arrays, keep on CPU then move)
    V_scr_r = tabulate_Vr(setup, screened_potential)      # (N_R, n_at, n_at)
    V_bare_r = bare_Coulomb_Vr(setup, eps_env=eps_env_exchange)

    N_R = setup.R_cart.shape[0]
    Nk = setup.kpoints_cart.shape[0]
    Nv = setup.C_vb.shape[1]
    Nc = setup.C_cb.shape[1]

    # ----- Finite-Q handling: replace electron E_ck, C_cb with values at k+Q
    if Q_frac is None:
        Q_frac = np.zeros(3, dtype=float)
    Q_frac = np.asarray(Q_frac, dtype=float).reshape(3)
    Q_is_zero = bool(np.allclose(Q_frac, 0.0))

    if Q_is_zero:
        C_cb_e = setup.C_cb            # (n_wann, Nc, Nk)
        E_ck_e = setup.E_ck            # (Nc, Nk)
    else:
        if coarse_setup is None:
            raise ValueError(
                "Q_frac != 0 requires coarse_setup (the pre-interpolation "
                "WannierSetup returned by build_setup()) so that bands and "
                "coefficients can be Wannier-interpolated at k+Q.")
        from .interpolate import interpolate_at_kpoints
        # Recover the VB/CB bloch slices used by the fine setup, by matching
        # the fine E arrays against a full-band diagonalisation on the fine k.
        full = interpolate_at_kpoints(
            coarse_setup, setup.kpoints_frac,
            vb_bloch_range=(0, coarse_setup.n_wann),
            cb_bloch_range=(0, coarse_setup.n_wann))
        def _find_slice(target, all_bands):
            n_full = all_bands.shape[0]
            n = target.shape[0]
            for o in range(n_full - n + 1):
                if np.allclose(all_bands[o:o + n], target, atol=1e-4):
                    return o, o + n
            raise RuntimeError(
                "Could not recover VB/CB bloch band range from fine setup — "
                "check that coarse_setup was used to build the fine setup.")
        v0, v1 = _find_slice(setup.E_vk, full["E_vk"])
        c0, c1 = _find_slice(setup.E_ck, full["E_vk"])
        # Interpolate at k+Q (folded into the first BZ by mod 1)
        kfrac_shifted = (setup.kpoints_frac + Q_frac[None, :]) % 1.0
        shifted = interpolate_at_kpoints(
            coarse_setup, kfrac_shifted,
            vb_bloch_range=(v0, v1),
            cb_bloch_range=(c0, c1))
        C_cb_e = shifted["C_cb"]       # (n_wann, Nc, Nk)  — evaluated at k+Q
        E_ck_e = shifted["E_ck"]       # (Nc, Nk)          — evaluated at k+Q
        if verbose:
            print(f"  Q_frac = {Q_frac.tolist()} : Wannier-interpolated "
                  f"electron bands/coefs at k+Q ({Nc} bands x {Nk} k-points).")

    # ----- Build the "orbital-atom expander": maps a Wannier orbital 'd' to
    # an atomic-site index 'i' via a boolean mask arr[i, d].
    n_at = len(setup.atom_groups)
    orb_of_atom = np.zeros((n_at, setup.n_wann), dtype=float)
    for i, g in enumerate(setup.atom_groups):
        orb_of_atom[i, g[1]:g[2]] = 1.0                   # 1 where orbital d is on atom i

    # ----- Move everything to the backend
    # For finite Q, electron uses shifted C_cb_e; hole uses original C_vb.
    Cv = xp.asarray(setup.C_vb)                   # (n_wann, Nv, Nk)  — hole
    Cc = xp.asarray(C_cb_e)                       # (n_wann, Nc, Nk)  — electron at k+Q
    O = xp.asarray(orb_of_atom)                   # (n_at, n_wann)
    R_cart = xp.asarray(setup.R_cart)             # (N_R, 3)
    kcart = xp.asarray(setup.kpoints_cart)        # (Nk, 3)
    # Weight the R-space potentials by the Wannier90 Wigner-Seitz degeneracy so
    # that boundary R-vectors (which appear multiple times in setup.R_cart)
    # collectively contribute V(R) with weight 1 per physical R.  For the
    # plain-supercell R-lattice used by interpolate_to_fine_grid, R_degen is
    # all ones and the weighting is a no-op — so the same normalisation works
    # for both the coarse (WS) and interpolated (supercell) paths.
    R_degen = np.asarray(setup.R_degen, dtype=float)     # (N_R,)
    V_scr = xp.asarray(V_scr_r / R_degen[:, None, None])  # (N_R, n_at, n_at)
    V_bare = xp.asarray(V_bare_r / R_degen[:, None, None])

    # Fourier phases  phi[k, k', R] = exp(i (k-k').R)
    #   dk[k,k',:] = k[k] - k[k']
    dk = kcart[:, None, :] - kcart[None, :, :]              # (Nk, Nk, 3)

    # ----------------------------------------------------------------
    # Precompute atom-summed overlap tensors
    #   Ac[i, c, c', k, k'] = sum_{d on i} conj(C_{d,c,k}) * C_{d,c',k'}
    #   Av[j, v, v', k, k'] = sum_{d on j} C_{d,v,k} * conj(C_{d,v',k'})
    # ----------------------------------------------------------------
    # per-orbital electron overlap Y_c[d, c, c', k, k']
    Yc = xp.einsum('dck,dpl->dcpkl', xp.conj(Cc), Cc)      # (n_wann, Nc, Nc, Nk, Nk)
    Ac = xp.einsum('id,dcpkl->icpkl', O, Yc)               # (n_at, Nc, Nc, Nk, Nk)
    del Yc
    Yv = xp.einsum('dvk,dql->dvqkl', Cv, xp.conj(Cv))
    Av = xp.einsum('jd,dvqkl->jvqkl', O, Yv)
    del Yv

    # Chunked R-loop: accumulate W_dir[i,j,k,k'] = (1/N_k) sum_R V(R,i,j) exp(i q · R)
    # where the sum uses the degen-weighted V(R)/degen(R) prepared above.  The 1/N_k
    # normalisation is the standard BSE convention: the BSE eigenvalue equation
    # ∑_{c'v'k'} K(cvk, c'v'k') A(c'v'k') = Ω A(cvk) requires K to scale as 1/N_k so
    # that the sum has a well-defined thermodynamic limit as N_k → ∞.  This matches
    # PyMEX's convention and makes the coarse-WS-lattice and interpolated-supercell
    # paths give consistent answers.
    W_dir = xp.zeros((n_at, n_at, Nk, Nk), dtype=complex)
    if include_exchange:
        W_ex = xp.zeros((n_at, n_at, Nk, Nk), dtype=complex)
    chunk_R = 64
    for r0 in range(0, N_R, chunk_R):
        r1 = min(r0 + chunk_R, N_R)
        phase_R = xp.exp(1j * xp.einsum('klr,Rr->klR', dk, R_cart[r0:r1]))
        W_dir += xp.einsum('Rij,klR->ijkl', V_scr[r0:r1], phase_R)
        if include_exchange:
            W_ex += xp.einsum('Rij,klR->ijkl', V_bare[r0:r1], phase_R)
        del phase_R
    W_dir /= Nk
    if include_exchange:
        W_ex /= Nk

    K_dir = xp.einsum('icpkl,jvqkl,ijkl->cvkpql',
                       Ac, Av, W_dir)                         # (Nc, Nv, Nk, Nc, Nv, Nk)
    del W_dir, Ac, Av

    if include_exchange:
        rho = xp.einsum('id,dck,dvk->icvk', O, xp.conj(Cc), Cv)   # (n_at, Nc, Nv, Nk)
        K_exch = -xp.einsum('icvk,jpql,ijkl->cvkpql',
                             rho, xp.conj(rho), W_ex)
        del rho, W_ex
    else:
        K_exch = 0.0

    # ---- Diagonal single-particle transition energies ----
    H = K_dir + (K_exch if include_exchange else 0)
    del K_dir
    if include_exchange:
        del K_exch

    # Add diagonal (c, v, k) -> (c, v, k).  For finite Q, E_c is at k+Q.
    E_v = xp.asarray(setup.E_vk)                  # (Nv, Nk)
    E_c = xp.asarray(E_ck_e)                      # (Nc, Nk)  — at k+Q if Q!=0
    # diag[c, v, k] = E_c[c, k+Q] - E_v[v, k]
    diag = E_c[:, None, :] - E_v[None, :, :]       # (Nc, Nv, Nk)
    # Add to H[c,v,k, c,v,k]
    dim = Nc * Nv * Nk
    Hflat = H.reshape(dim, dim)
    diag_flat = diag.reshape(dim)
    Hflat = Hflat + xp.diag(diag_flat)

    # ---- Ensure Hermiticity numerically  ----
    Hflat = 0.5 * (Hflat + xp.conj(Hflat.T))
    return to_numpy(Hflat)
