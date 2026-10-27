"""Optical absorption from BSE eigenvalues and eigenvectors.

Simplest form: sum over bright excitons weighted by their oscillator strength,
each broadened by a Gaussian.  For the toy Dirac model the interband dipole
matrix element is essentially constant over the small K-valley patch used,
so we approximate the oscillator strength by |sum_{cvk} A^S_{cvk}|^2.
For real materials one should use the velocity matrix element:
   O_S = | sum_{cvk} A^S_{cvk} * <c,k| v_alpha |v,k> |^2
which is a drop-in replacement in this routine.

Reference: Rohlfing & Louie, Phys. Rev. B 62, 4927 (2000).
"""
from __future__ import annotations
import numpy as np


def optical_absorption(eigval: np.ndarray,
                       eigvec: np.ndarray,
                       photon_energies: np.ndarray,
                       Nc: int, Nv: int, Nk: int,
                       sigma: float = 0.010,
                       dipole: np.ndarray | None = None) -> np.ndarray:
    """
    Parameters
    ----------
    eigval
        (Nex,) exciton energies (eV).
    eigvec
        (Nex, Nex) BSE eigenvectors (columns are excitons).
    photon_energies
        Frequency grid (eV) on which to evaluate absorption.
    Nc, Nv, Nk
        Dimensions of the (c,v,k) basis.
    sigma
        Gaussian broadening (eV).
    dipole
        Optional (Nc, Nv, Nk) complex interband dipole matrix.  If None, use
        unit dipole (appropriate for the toy Dirac model near K).

    Returns
    -------
    (n_photon,) real spectrum, in arbitrary but consistent units.
    """
    if dipole is None:
        dipole = np.ones((Nc, Nv, Nk), dtype=complex)
    d_flat = dipole.ravel()

    # Oscillator strength for each exciton
    #   O_S = | sum_{cvk} A^S_{cvk} d_{cvk} |^2
    O = np.abs(eigvec.T @ d_flat)**2      # (Nex,)

    # Broaden with Gaussian
    E = np.asarray(photon_energies)[:, None]
    Om = eigval[None, :]
    G = np.exp(-((E - Om) / sigma)**2 / 2.0) / (sigma * np.sqrt(2 * np.pi))
    return (G * O[None, :]).sum(axis=1)
