"""Finite-Q exciton dispersion Ω_S(Q).

For a list of centre-of-mass momenta Q (in fractional coordinates), build
the BSE Hamiltonian at each Q, diagonalise, and return the lowest n_states
exciton energies as a function of Q.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

from .wannier_setup import WannierSetup
from .hamiltonian import build_bse_hamiltonian
from .solver import diagonalize
from ..potentials.base import Potential


def compute_exciton_dispersion(fine: WannierSetup,
                               coarse: WannierSetup,
                               potential: Potential,
                               Q_frac_list: Iterable,
                               n_states: int = 20,
                               eps_env_exchange: float = 1.0,
                               include_exchange: bool = True,
                               prefer_gpu: bool = False,
                               verbose: bool = False
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Ω_S(Q) for a list of Q vectors.

    Parameters
    ----------
    fine
        The WannierSetup on the fine k-grid (from interpolate_to_fine_grid).
        This is the mesh over which the BSE is expanded.
    coarse
        The pre-interpolation WannierSetup (from build_setup); needed so
        that bands and coefficients can be Wannier-interpolated at k+Q.
    potential
        Screened electron-hole interaction (RytovaKeldysh / ERK / ...).
    Q_frac_list
        Iterable of length-3 fractional-coordinate Q vectors.
    n_states
        Number of lowest exciton states to store per Q.
    eps_env_exchange, include_exchange, prefer_gpu, verbose
        Forwarded to build_bse_hamiltonian.

    Returns
    -------
    Q_arr : (N_Q, 3) ndarray  — the input Q_frac list, stacked.
    Omega : (N_Q, n_states) ndarray — the lowest n_states exciton eigenvalues (eV)
            at each Q.
    """
    Q_list = [np.asarray(q, dtype=float).reshape(3) for q in Q_frac_list]
    Q_arr = np.stack(Q_list, axis=0)
    N_Q = Q_arr.shape[0]

    Omega = np.zeros((N_Q, n_states), dtype=float)
    for iQ, Qf in enumerate(Q_list):
        if verbose:
            print(f"  [{iQ + 1}/{N_Q}]  Q_frac = {Qf.tolist()}")
        H = build_bse_hamiltonian(
            fine, potential,
            eps_env_exchange=eps_env_exchange,
            include_exchange=include_exchange,
            prefer_gpu=prefer_gpu,
            verbose=False,
            Q_frac=Qf,
            coarse_setup=coarse,
        )
        eigval, _ = diagonalize(H)
        Omega[iQ, :] = eigval[:n_states]
        if verbose:
            print(f"           Ω_1s(Q) = {eigval[0]:.4f} eV")
    return Q_arr, Omega
