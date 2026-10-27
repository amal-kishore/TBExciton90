"""Assemble everything the BSE needs from Wannier90 + QE DFT bands.

Central object: WannierSetup — a plain data container populated from the
Wannier90/QE files that the BSE Hamiltonian builder consumes.

Follows the PyMEX conventions so numerical results can be cross-checked
against the reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from ..io.wan90 import (read_win, read_wannier_centres, read_hr,
                        read_full_wannier_rotation)
from ..io.bands import read_qe_bands, read_w90_eig


@dataclass
class WannierSetup:
    lattice: np.ndarray                      # (3, 3), A
    mp_grid: Tuple[int, int, int]
    kpoints_frac: np.ndarray                 # (Nk, 3)
    kpoints_cart: np.ndarray                 # (Nk, 3) 1/A
    n_wann: int
    U: np.ndarray                            # (n_bloch, n_wann, Nk)
    wannier_centres: np.ndarray              # (n_wann, 3) A
    atom_groups: List[Tuple[str, int, int, np.ndarray]]
    # ^^ each entry: (label, orbital_start, orbital_stop, atom_pos_A)
    R_frac: np.ndarray                       # (N_R, 3) integer
    R_cart: np.ndarray                       # (N_R, 3) A
    R_degen: np.ndarray                      # (N_R,)
    H_r: np.ndarray                          # (N_R, n_wann, n_wann) eV
    E_ck: np.ndarray                         # (n_cb, Nk) eV, DFT conduction bands
    E_vk: np.ndarray                         # (n_vb, Nk) eV, DFT valence bands
    C_vb: np.ndarray                         # (n_wann, n_vb, Nk) complex
    C_cb: np.ndarray                         # (n_wann, n_cb, Nk) complex

    # Diagnostic derived quantities
    @property
    def unit_cell_area_A2(self) -> float:
        """In-plane unit-cell area (for 2D materials)."""
        a1, a2 = self.lattice[0, :2], self.lattice[1, :2]
        return float(abs(a1[0] * a2[1] - a1[1] * a2[0]))

    @property
    def transition_gap_direct(self) -> float:
        """min over k of (E_cb_min - E_vb_max)."""
        return float((self.E_ck.min(axis=0) - self.E_vk.max(axis=0)).min())


def _minimum_image_distance(c: np.ndarray, p: np.ndarray,
                             lattice: np.ndarray | None) -> float:
    """Distance between c and p under periodic boundary conditions of `lattice`.

    If lattice is None, plain Euclidean distance is returned.
    Otherwise we express (c - p) in fractional coordinates, wrap into
    [-0.5, 0.5), and convert back to cartesian.
    """
    if lattice is None:
        return float(np.linalg.norm(c - p))
    d = c - p
    frac = np.linalg.solve(lattice.T, d)
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ lattice))


def _group_wannier_into_atoms(centres: np.ndarray,
                              atom_labels: List[str],
                              atom_positions: np.ndarray,
                              tol_A: float = 0.5,
                              lattice: np.ndarray | None = None
                              ) -> List[Tuple[str, int, int, np.ndarray]]:
    """Cluster Wannier centres onto atomic sites.

    Two strategies, tried in order:
      1. Strict tolerance: WF within `tol_A` of an atom is bound to it.
         Fast, works when MLWFs stay close to their initial projections.
      2. Nearest-atom fallback: if strategy 1 leaves any WF unassigned,
         assign each WF to its nearest atom.  Standard for MLWFs that
         drift after minimisation.

    Returns a list of (label, w_start, w_stop, position) tuples with
    contiguous [w_start, w_stop) ranges — required for the vectorised
    orbital-atom expander in the BSE builder.  If a physical grouping
    is non-contiguous (rare, if projections were listed atom-by-atom),
    an explicit relabelling step is needed.
    """
    n_wann = len(centres)
    # Strategy 1: tolerance-based (periodic-aware)
    tol_assign = -np.ones(n_wann, dtype=int)
    for a_idx, pos in enumerate(atom_positions):
        for i, c in enumerate(centres):
            if tol_assign[i] < 0 and _minimum_image_distance(c, pos, lattice) < tol_A:
                tol_assign[i] = a_idx

    if (tol_assign >= 0).all():
        assign = tol_assign
    else:
        # Strategy 2: nearest-atom (periodic-aware) for every WF
        assign = np.zeros(n_wann, dtype=int)
        for i, c in enumerate(centres):
            d = [_minimum_image_distance(c, p, lattice) for p in atom_positions]
            assign[i] = int(np.argmin(d))

    # Contiguity check: WFs assigned to the same atom must be adjacent in index.
    # If they are not, we permute the WF ordering so that they become adjacent —
    # but that permutation must also be applied to the U-matrix / hr.  For now
    # we assert contiguity and raise a helpful error if it fails.
    groups = []
    prev_atom = None
    start = 0
    for i in range(n_wann + 1):
        cur_atom = int(assign[i]) if i < n_wann else -1
        if cur_atom != prev_atom:
            if prev_atom is not None:
                groups.append(
                    (atom_labels[prev_atom], start, i,
                     np.asarray(atom_positions[prev_atom], dtype=float))
                )
            start = i
            prev_atom = cur_atom
    # Verify no *physical* atom (identified by position) appears in two
    # disconnected ranges. Duplicate labels (e.g. two "In" atoms) are fine.
    seen_positions = []
    for lbl, s, e, pos in groups:
        for existing in seen_positions:
            if np.linalg.norm(existing - pos) < 1e-6:
                raise ValueError(
                    f"Wannier centres assign atom {lbl}@{pos} to non-contiguous "
                    f"index ranges — atom_groups requires contiguity."
                )
        seen_positions.append(pos)
    return groups


def build_setup(seedname: str,
                bands_file: str,
                vb_wann_range: Tuple[int, int],
                cb_wann_range: Tuple[int, int],
                dft_skip_bands: int = 0,
                wan_dir: str = ".",
                atom_group_tol_A: float = 0.5,
                bands_format: str = "auto") -> WannierSetup:
    """Build a WannierSetup from a Wannier90 seedname + DFT bands file.

    Parameters
    ----------
    seedname
        e.g. 'WSe2'. Files expected: <wan_dir>/<seedname>.{win,wout,_hr.dat,_u.mat}
    bands_file
        Path to DFT eigenvalues on the same k-mesh as the Wannierisation.
        Two formats supported:
          * QE bands.x plot file (`Bands_QE`)  — set bands_format='qe'
          * Wannier90 `<seedname>.eig`          — set bands_format='w90'
        With bands_format='auto' (default) the format is inferred from
        the first line.
    vb_wann_range, cb_wann_range
        (start, stop) inclusive-of-start, exclusive-of-stop Wannier-band
        indices used to define the valence and conduction bands for the BSE.
    dft_skip_bands
        Number of DFT bands to skip before the Wannier subspace begins,
        i.e. DFT_index = wannier_index + dft_skip_bands.  For a `.eig`
        file the indexing already matches the Wannier subspace (via
        disentanglement), so use dft_skip_bands=0.
    """
    import os
    wjoin = lambda ext: os.path.join(wan_dir, f"{seedname}{ext}")

    win = read_win(wjoin(".win"))
    centres = read_wannier_centres(wjoin(".wout"))
    R_frac, degen, H_r = read_hr(wjoin("_hr.dat"))
    kpts_frac_u, U = read_full_wannier_rotation(seedname, wan_dir=wan_dir)

    # Detect format if requested
    if bands_format == "auto":
        with open(bands_file) as f:
            first = f.readline()
        bands_format = "qe" if "plot" in first.lower() else "w90"
    if bands_format == "qe":
        _, E_dft = read_qe_bands(bands_file)
    elif bands_format == "w90":
        E_dft = read_w90_eig(bands_file)
    else:
        raise ValueError(f"Unknown bands_format={bands_format!r}")

    # Reciprocal lattice (2 pi * (A^T)^-1)
    B = 2.0 * np.pi * np.linalg.inv(win.lattice.T)

    # Convert lattice-vector Rs and k-points to cartesian
    kpoints_cart = win.kpoints_frac @ B
    R_cart = R_frac @ win.lattice

    # Group Wannier centres onto atoms — periodic-image-aware
    atom_groups = _group_wannier_into_atoms(centres,
                                            win.atom_labels,
                                            win.atom_positions_A,
                                            tol_A=atom_group_tol_A,
                                            lattice=win.lattice)

    # Slice VB / CB bands
    v0, v1 = vb_wann_range
    c0, c1 = cb_wann_range
    dft_v_lo, dft_v_hi = v0 + dft_skip_bands, v1 + dft_skip_bands
    dft_c_lo, dft_c_hi = c0 + dft_skip_bands, c1 + dft_skip_bands
    E_vk = E_dft[dft_v_lo:dft_v_hi]           # (n_vb, Nk)
    E_ck = E_dft[dft_c_lo:dft_c_hi]           # (n_cb, Nk)

    # U shape (n_bloch=n_wann, n_wann_out, Nk). PyMEX slices n_wann axis:
    #     C = U^H   ->  C[wannier_idx, wann_out_idx, k] = conj(U[wann_out_idx, wannier_idx, k])
    # For BSE we want C_bands[orbital, band, k].
    # The wannier index that maps to VB band 'v' is v0..v1.
    C_full = np.conj(np.transpose(U, (1, 0, 2)))    # (n_wann_out, n_bloch, Nk) — this is U^dagger
    # Convention alignment with PyMEX:
    #   PyMEX's get_C() returns np.moveaxis(C_k_nm, 0, -1) with C_k_nm = U^H
    #   so PyMEX's C_nm_k has shape (n_wann_out, n_wann_in, Nk)
    # Then PyMEX's c_nv_k slices the 'n' (first) axis to select bands.
    # We follow the exact same slicing.
    C_vb = C_full[:, v0:v1, :]      # (n_wann_orb, n_vb, Nk)
    C_cb = C_full[:, c0:c1, :]      # (n_wann_orb, n_cb, Nk)

    return WannierSetup(
        lattice=win.lattice,
        mp_grid=win.mp_grid,
        kpoints_frac=win.kpoints_frac,
        kpoints_cart=kpoints_cart,
        n_wann=U.shape[1],       # number of Wannier orbitals, not Bloch bands
        U=U,
        wannier_centres=centres,
        atom_groups=atom_groups,
        R_frac=R_frac,
        R_cart=R_cart,
        R_degen=degen,
        H_r=H_r,
        E_ck=E_ck,
        E_vk=E_vk,
        C_vb=C_vb,
        C_cb=C_cb,
    )
