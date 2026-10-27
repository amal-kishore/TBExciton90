"""Readers for DFT single-particle eigenvalues.

Two formats:

- Quantum ESPRESSO 'plot_bands' output (`Bands_QE`):
    &plot nbnd=  Nb, nks=  Nk /
    k_x  k_y  k_z
    <Nb energies in eV, wrapped ~10 per line>
    ...

- Wannier90 `<seedname>.eig`:
    band_index  kpoint_index  energy_eV
    (one line per (band, k) pair, band index fastest)
"""
from __future__ import annotations
import numpy as np


def read_qe_bands(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (kpoints_frac (Nk, 3), energies (Nb, Nk)) from a QE bands.x file."""
    with open(path) as f:
        lines = f.readlines()

    header = lines[0]
    # Parse "&plot nbnd=  Nb, nks=  Nk /"  — tolerate "nbnd= 42" or "nbnd=42"
    import re
    m_nb = re.search(r"nbnd\s*=\s*(\d+)", header)
    m_nk = re.search(r"nks\s*=\s*(\d+)", header)
    if not m_nb or not m_nk:
        raise ValueError(f"Could not parse nbnd/nks from header:\n  {header}")
    nb = int(m_nb.group(1))
    nk = int(m_nk.group(1))

    E = np.zeros((nb, nk), dtype=float)
    kpts = np.zeros((nk, 3), dtype=float)

    idx = 1
    for ik in range(nk):
        kpts[ik] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1
        vals = []
        while len(vals) < nb:
            vals.extend(float(x) for x in lines[idx].split())
            idx += 1
        E[:, ik] = vals[:nb]

    return kpts, E


def read_w90_eig(path: str) -> np.ndarray:
    """Read a Wannier90 <seedname>.eig file.

    The file has one line per (band, kpoint) pair:
        band_index  kpoint_index  energy_eV
    with band_index changing fastest and both indices 1-based.

    Returns
    -------
    E : (n_bands, n_kpoints) float array
    """
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Unexpected .eig format in {path}: shape {data.shape}")
    b_idx = data[:, 0].astype(int)
    k_idx = data[:, 1].astype(int)
    nb = b_idx.max()
    nk = k_idx.max()
    if len(data) != nb * nk:
        raise ValueError(
            f"{path}: expected {nb*nk} lines, got {len(data)}"
        )
    E = np.zeros((nb, nk), dtype=float)
    E[b_idx - 1, k_idx - 1] = data[:, 2]
    return E
