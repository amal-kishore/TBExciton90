"""BSE diagonalisation and eigenvector unfolding."""
from __future__ import annotations
import numpy as np


def diagonalize(H: np.ndarray, hermitian_tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise the BSE Hamiltonian.

    Returns (eigenvalues, eigenvectors) with the same convention as PyMEX:
    eigval[i] is the i-th exciton energy in eV (sorted ascending), eigvec[:, i]
    the corresponding coefficient vector A^i in the (c,v,k) product basis.
    """
    non_herm = np.max(np.abs(H - H.conj().T))
    if non_herm > hermitian_tol:
        raise ValueError(
            f"BSE Hamiltonian is not Hermitian to tol {hermitian_tol}: "
            f"max |H - H^dagger| = {non_herm:.3e}"
        )
    eigval, eigvec = np.linalg.eigh((H + H.conj().T) / 2.0)
    return eigval, eigvec


def unfold_eigenvector(A_flat: np.ndarray, Nc: int, Nv: int, Nk: int) -> np.ndarray:
    """Reshape a flat A vector into A[c, v, k] tensor."""
    return A_flat.reshape(Nc, Nv, Nk)
