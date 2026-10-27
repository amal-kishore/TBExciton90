"""Readers for Wannier90 output files.

Handles the file formats produced by Wannier90 v3.x:
    * <seedname>.win           input file (lattice, mp_grid, kpoints, atom pos.)
    * <seedname>.wout          run log (Wannier centre positions)
    * <seedname>_hr.dat        real-space Hamiltonian H(R)_{mn}
    * <seedname>_wsvec.dat     Wigner-Seitz translation vectors
    * <seedname>_u.mat         unitary rotation U^k_{mn} (Bloch -> Wannier)

Conventions:
    * lengths in angstrom, energies in eV.
    * H(R)_{mn} in electronvolts, R in fractional lattice coordinates.
    * U^k_{mn} indexed as U[m, n]  (m: original Bloch band, n: Wannier index).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from ..constants import BOHR_TO_A


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------
@dataclass
class Win:
    """Parsed contents of <seedname>.win."""
    lattice: np.ndarray                # (3, 3), angstrom
    mp_grid: Tuple[int, int, int]      # (n1, n2, n3)
    kpoints_frac: np.ndarray           # (Nk, 3), fractional
    atom_labels: list                  # length N_at
    atom_positions_A: np.ndarray       # (N_at, 3), angstrom (cartesian)

    @property
    def n_kpoints(self) -> int:
        return int(np.prod(self.mp_grid))


# ----------------------------------------------------------------------
# .win
# ----------------------------------------------------------------------
def read_win(path: str) -> Win:
    """Parse a Wannier90 .win file."""
    with open(path) as f:
        text = f.read()
    lines = text.splitlines()

    lat = np.zeros((3, 3), dtype=float)
    mp_grid = None
    kpoints = None
    atoms_lbl: list = []
    atoms_pos: list = []

    i = 0
    while i < len(lines):
        low = lines[i].strip().lower()

        if low.startswith("begin unit_cell_cart"):
            unit = lines[i + 1].strip().lower()
            base = 2 if unit in ("bohr", "ang", "angstrom") else 1
            for j in range(3):
                row = lines[i + base + j].split()
                lat[j] = [float(x) for x in row[:3]]
            if unit == "bohr":
                lat = lat * BOHR_TO_A
            i = i + base + 3
            continue

        if low.startswith("mp_grid"):
            parts = lines[i].split(":")[-1].split() if ":" in lines[i] \
                    else lines[i].split()[1:]
            # 'mp_grid : 9 9 1'   or   'mp_grid = 9 9 1'   or   'mp_grid 9 9 1'
            parts = [p for p in parts if p not in ("=", ":")]
            mp_grid = tuple(int(p) for p in parts[:3])

        if low.startswith("begin kpoints"):
            j = i + 1
            kp = []
            while not lines[j].strip().lower().startswith("end kpoints"):
                if lines[j].strip():
                    row = lines[j].split()
                    kp.append([float(x) for x in row[:3]])
                j += 1
            kpoints = np.asarray(kp, dtype=float)
            i = j

        if low.startswith("begin atoms_cart") or low.startswith("begin atoms_frac"):
            frac = low.startswith("begin atoms_frac")
            unit = lines[i + 1].strip().lower()
            base = 2 if unit in ("bohr", "ang", "angstrom") else 1
            j = i + base
            while not lines[j].strip().lower().startswith("end atoms"):
                if lines[j].strip():
                    row = lines[j].split()
                    atoms_lbl.append(row[0])
                    atoms_pos.append([float(x) for x in row[1:4]])
                j += 1
            atoms_pos_arr = np.asarray(atoms_pos, dtype=float)
            if frac:
                atoms_pos_arr = atoms_pos_arr @ lat
            elif unit == "bohr":
                atoms_pos_arr *= BOHR_TO_A
            atoms_pos = atoms_pos_arr.tolist()
            i = j

        i += 1

    if mp_grid is None:
        raise ValueError(f"mp_grid not found in {path}")
    if kpoints is None:
        raise ValueError(f"kpoints block not found in {path}")

    return Win(lattice=lat,
               mp_grid=mp_grid,
               kpoints_frac=kpoints,
               atom_labels=atoms_lbl,
               atom_positions_A=np.asarray(atoms_pos, dtype=float))


# ----------------------------------------------------------------------
# .wout — Wannier centres
# ----------------------------------------------------------------------
def read_wannier_centres(wout_path: str) -> np.ndarray:
    """Read final Wannier centre positions in angstroms from a .wout file.

    Returns (n_wann, 3) array.  Uses the "Final State" block that
    Wannier90 prints at convergence.
    """
    with open(wout_path) as f:
        lines = f.readlines()

    # Grab n_wann from 'Number of Wannier Functions'
    n_wann = None
    for L in lines:
        if "Number of Wannier Functions" in L:
            n_wann = int(L.split(":")[-1].strip().split()[0])
            break
    if n_wann is None:
        raise ValueError(f"'Number of Wannier Functions' not found in {wout_path}")

    # Find the final "Final State" block (or the last "WF centre" block)
    centres = np.zeros((n_wann, 3), dtype=float)
    for i, L in enumerate(lines):
        if "Final State" in L:
            for k in range(n_wann):
                row = lines[i + 1 + k]
                # 'WF centre and spread    1  (  1.876220,  1.083240,  0.000000 )   1.234'
                inside = row.split("(")[1].split(")")[0]
                x, y, z = [float(t) for t in inside.split(",")]
                centres[k] = [x, y, z]
    return centres


# ----------------------------------------------------------------------
# _hr.dat
# ----------------------------------------------------------------------
def read_hr(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the Wannier90 <seedname>_hr.dat file.

    Returns
    -------
    R_frac : (N_R, 3)  integer fractional lattice vectors
    degen  : (N_R,)    Wigner-Seitz degeneracies (as printed by W90)
    Hmn    : (N_R, N_wann, N_wann)  Hamiltonian matrix elements in eV
    """
    with open(path) as f:
        lines = f.readlines()

    n_wann = int(lines[1].split()[0])
    n_R = int(lines[2].split()[0])

    # Degeneracies: 15 per line
    n_deg_lines = (n_R + 14) // 15
    degen = []
    for j in range(n_deg_lines):
        degen.extend(int(x) for x in lines[3 + j].split())
    degen = np.asarray(degen[:n_R], dtype=int)

    offset = 3 + n_deg_lines
    Hmn = np.zeros((n_R, n_wann, n_wann), dtype=complex)
    R_frac = np.zeros((n_R, 3), dtype=int)

    per_R = n_wann * n_wann
    for ir in range(n_R):
        for row in range(per_R):
            parts = lines[offset + ir * per_R + row].split()
            R_frac[ir] = [int(parts[0]), int(parts[1]), int(parts[2])]
            m = int(parts[3]) - 1
            n = int(parts[4]) - 1
            re = float(parts[5])
            im = float(parts[6])
            # PyMEX's convention: h_r[R, n, m] = re + i*im (indices reversed);
            # to match, store the same way.
            Hmn[ir, n, m] = complex(re, im)

    return R_frac, degen, Hmn


# ----------------------------------------------------------------------
# _u.mat
# ----------------------------------------------------------------------
def read_umat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Wannier90 U-matrix file (<seedname>_u.mat OR _u_dis.mat).

    Both files share the same textual format::

        <comment line>
        Nk  N_out  N_in
        (blank)
        kx ky kz         (Nk times)
        Re Im            (N_out * N_in Fortran-order rows)

    Returns
    -------
    kpoints_frac : (Nk, 3)
    U            : (N_in, N_out, Nk) complex

    For `_u.mat` (post-disentanglement rotation): N_in = N_out = n_wann.
    For `_u_dis.mat` (disentanglement rotation):  N_in = n_bnd, N_out = n_wann.

    NB: PyMEX/Wannier90 write the block column-major (Fortran); we reshape
    the same way and store as (N_in, N_out, Nk).
    """
    with open(path) as f:
        lines = f.readlines()

    header = lines[1].split()
    nk    = int(header[0])
    n_out = int(header[1])
    n_in  = int(header[2])

    kpts = np.zeros((nk, 3), dtype=float)
    U = np.zeros((n_in, n_out, nk), dtype=complex)

    block = 2 + n_out * n_in       # blank line + data lines
    for ik in range(nk):
        base = 3 + ik * block
        kpts[ik] = [float(x) for x in lines[base].split()[:3]]
        tmp = []
        for j in range(1, n_out * n_in + 1):
            parts = lines[base + j].split()
            tmp.append(complex(float(parts[0]), float(parts[1])))
        U[:, :, ik] = np.reshape(tmp, (n_in, n_out), order="F")

    return kpts, U


def read_full_wannier_rotation(seedname: str, wan_dir: str = ".",
                                eig_file: str | None = None,
                                dis_win_min: float | None = None,
                                dis_win_max: float | None = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Bloch->Wannier rotation as (n_bloch, n_wann, Nk).

    Wannier90 stores `_u_dis.mat` in a *compressed* band ordering — only
    bands within the outer disentanglement window [dis_win_min,
    dis_win_max] at each k are present, listed in ascending-energy order
    and padded with zeros up to num_bands.  To recover the true
    (n_bloch, n_wann) rotation we need to remap these compressed rows
    back to their DFT band indices, using the `.eig` file for the energies
    and the outer window bounds from the `.win`.

    If `_u_dis.mat` does not exist (no disentanglement), the composition
    is trivial and we return `_u.mat` directly.
    """
    import os
    kpts, U_post = read_umat(os.path.join(wan_dir, f"{seedname}_u.mat"))
    dis_path = os.path.join(wan_dir, f"{seedname}_u_dis.mat")
    if not os.path.exists(dis_path):
        return kpts, U_post

    _, U_dis_compressed = read_umat(dis_path)   # (n_bnd, n_wann, Nk), compressed order
    n_bnd, n_wann, Nk = U_dis_compressed.shape

    # For the compressed-order remap, we need per-k list of in-window band
    # DFT indices.  Auto-locate .eig file and .win if not provided.
    if eig_file is None:
        eig_file = os.path.join(wan_dir, f"{seedname}.eig")
    if dis_win_min is None or dis_win_max is None:
        win_path = os.path.join(wan_dir, f"{seedname}.win")
        with open(win_path) as f:
            for L in f:
                low = L.strip().lower()
                if low.startswith("dis_win_min"):
                    dis_win_min = float(L.split("=")[-1].split("!")[0].split()[0])
                elif low.startswith("dis_win_max"):
                    dis_win_max = float(L.split("=")[-1].split("!")[0].split()[0])
        if dis_win_min is None or dis_win_max is None:
            # No outer window declared -> treat as trivial (no disentanglement)
            return kpts, np.einsum("bmk,mnk->bnk", U_dis_compressed, U_post)

    # Load DFT energies (n_bnd, Nk) and build per-k index maps
    from .bands import read_w90_eig
    E = read_w90_eig(eig_file)                  # (n_bnd_dft, Nk)
    n_bnd_dft = E.shape[0]
    if n_bnd_dft != n_bnd:
        # Should never happen with a proper Wannier90 run, but bail loudly.
        raise ValueError(
            f".eig has {n_bnd_dft} bands, _u_dis.mat claims {n_bnd}; mismatch"
        )

    U_full = np.zeros((n_bnd_dft, n_wann, Nk), dtype=complex)
    for ik in range(Nk):
        # DFT bands in the outer window at this k, in ascending-energy order
        in_win = np.where((E[:, ik] >= dis_win_min) & (E[:, ik] <= dis_win_max))[0]
        order = in_win[np.argsort(E[in_win, ik])]
        ndimwin = len(order)
        # Compose U_dis (compressed) @ U_post -> U_full (compressed_bnd, n_wann)
        rot_compressed = U_dis_compressed[:ndimwin, :, ik] @ U_post[:, :, ik]
        # Scatter back to DFT-band indexing
        U_full[order, :, ik] = rot_compressed
    return kpts, U_full


# ----------------------------------------------------------------------
# _wsvec.dat  (auxiliary; not strictly needed if we use the raw hr.dat)
# ----------------------------------------------------------------------
def read_wsvec(path: str) -> None:
    """Placeholder — the WS-vector translation vectors are only needed for
    minimum-image band-interpolation.  The BSE builder does not use them."""
    return None
