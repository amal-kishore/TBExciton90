"""BGW-style real-space exciton wavefunction with the hole pinned at a Wannier atom.

For a Q=0 exciton with BSE amplitudes A^S_{cvk}, the two-particle wavefunction is

    Ψ_S(r_e, r_h) = Σ_{cvk} A^S_{cvk}  ψ_c(r_e; k)  ψ_v^*(r_h; k)

In the Wannier basis, ψ_n(r; k) = Σ_i C_{i,n}(k) w_i(r - t_i - R) e^{i k·R},
so if we treat each Wannier orbital as a delta function at its centre
(coarse envelope approximation) the electron probability at atom-site
(R, i_e) with the hole pinned at (0, i_h) is

    Ψ_S(R, i_e | i_h) = Σ_{cvk} A^S_{cvk}
                       [Σ_{d on atom i_e} C_{d,c}(k)] e^{+i k·(R + t_{i_e} - t_{i_h})}
                       [Σ_{d' on atom i_h} C_{d',v}^*(k)].

This matches PyMEX's opt_exciton_r kernel and reproduces the "hole-fixed"
image conventionally shown in BerkeleyGW plotxct output.
"""
from __future__ import annotations
from typing import Tuple, List
import numpy as np


def build_atom_C(setup, cb_or_vb: str) -> np.ndarray:
    """Sum Wannier-orbital coefficients over the orbitals on each atom.

    Returns C_atom[n_atoms, n_bands, Nk] where band index is either VB or CB.
    """
    C = setup.C_cb if cb_or_vb == "cb" else setup.C_vb    # (n_orb, n_band, Nk)
    n_at = len(setup.atom_groups)
    n_band = C.shape[1]
    Nk = C.shape[2]
    C_at = np.zeros((n_at, n_band, Nk), dtype=complex)
    for a, g in enumerate(setup.atom_groups):
        d_lo, d_hi = g[1], g[2]
        C_at[a, :, :] = C[d_lo:d_hi, :, :].sum(axis=0)
    return C_at


def hole_fixed_exciton_density(setup,
                               A_S: np.ndarray,
                               hole_atom_index: int,
                               n_supercell: int = 5
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute |Ψ_S(R + t_ie, t_ih)|² for all atom sites in an n×n supercell.

    Parameters
    ----------
    setup
        WannierSetup (coarse or Wannier-interpolated).
    A_S
        (Nc, Nv, Nk) BSE amplitudes for one exciton (or a summed multiplet).
    hole_atom_index
        Index into setup.atom_groups pinning the hole (in the origin cell).
    n_supercell
        Extent (in unit cells) of the electron-image supercell,
        [-n_supercell, +n_supercell] along each in-plane lattice vector.

    Returns
    -------
    positions : (N_pts, 2)    x,y in Å of every atom site plotted
    labels    : (N_pts,)       atom label (str) — e.g. 'W', 'Se'
    density   : (N_pts,)       |Ψ|² at that site (unnormalised)
    """
    Nc, Nv, Nk = A_S.shape
    C_c_at = build_atom_C(setup, "cb")       # (n_at, Nc, Nk)
    C_v_at = build_atom_C(setup, "vb")       # (n_at, Nv, Nk)
    n_at = C_c_at.shape[0]

    # Hole coefficient sum over VB at fixed atom i_h: shape (Nv, Nk)
    Ch = np.conj(C_v_at[hole_atom_index])     # (Nv, Nk)

    # Ψ(R, i_e) = Σ_cvk A_cvk C_c^{i_e}(k) e^{i k·(R + t_ie - t_ih)} conj(C_v^{i_h}(k))
    # Contract v first with the (fixed) hole coefficients:
    Mck = np.einsum("cvk,vk->ck", A_S, Ch)          # (Nc, Nk)

    # Then Ψ(R, i_e) = Σ_k [Σ_c Mck(c,k) * C_c_at(i_e, c, k)] e^{i k·(R + t_ie - t_ih)}
    # Define phi(i_e, k) = Σ_c Mck(c,k) * C_c_at(i_e, c, k)
    phi = np.einsum("ck,eck->ek", Mck, C_c_at)      # (n_at, Nk)

    # Build the electron real-space supercell
    a1 = setup.lattice[0, :2]
    a2 = setup.lattice[1, :2]
    n_range = np.arange(-n_supercell, n_supercell + 1)
    positions = []
    labels    = []
    density   = []
    t_h = np.array(setup.atom_groups[hole_atom_index][3][:2])
    k_cart = setup.kpoints_cart[:, :2]

    for n1 in n_range:
        for n2 in n_range:
            R = n1 * a1 + n2 * a2                    # cell offset (in-plane, Å)
            for ie in range(n_at):
                t_e = np.array(setup.atom_groups[ie][3][:2])
                r_full = R + t_e - t_h
                phase = np.exp(1j * (k_cart @ r_full))          # (Nk,)
                Psi = np.sum(phi[ie] * phase)
                positions.append(R + t_e)
                labels.append(setup.atom_groups[ie][0])
                density.append(np.abs(Psi)**2)

    positions = np.asarray(positions)
    labels    = np.asarray(labels)
    density   = np.asarray(density)
    return positions, labels, density
