"""HDF5 I/O compatible with PyMEX file naming conventions.

We deliberately use the same dataset names as PyMEX ('eigval', 'eigvec',
'sigma_xx', 'photon', 'bse_ham') so that the same post-processing scripts
can consume outputs from either package.
"""
from __future__ import annotations
import os
import h5py
import numpy as np


def save_results(outdir: str, eigval: np.ndarray, eigvec: np.ndarray,
                 photon: np.ndarray, sigma_xx: np.ndarray,
                 H: np.ndarray | None = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(os.path.join(outdir, "Eigval.hdf5"), "w") as f:
        f.create_dataset("eigval", data=eigval)
    with h5py.File(os.path.join(outdir, "Eigvec.hdf5"), "w") as f:
        f.create_dataset("eigvec", data=eigvec)
    with h5py.File(os.path.join(outdir, "E_ph.hdf5"), "w") as f:
        f.create_dataset("photon", data=photon)
    with h5py.File(os.path.join(outdir, "SIGMA_full.hdf5"), "w") as f:
        f.create_dataset("sigma_xx", data=sigma_xx)
    if H is not None:
        with h5py.File(os.path.join(outdir, "H_eh.hdf5"), "w") as f:
            f.create_dataset("bse_ham", data=H)


def load_results(indir: str) -> dict:
    out = {}
    for fname, key in [("Eigval.hdf5", "eigval"),
                       ("Eigvec.hdf5", "eigvec"),
                       ("E_ph.hdf5", "photon"),
                       ("SIGMA_full.hdf5", "sigma_xx")]:
        path = os.path.join(indir, fname)
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                out[key] = f[key][...]
    return out
