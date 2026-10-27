"""Direct 2D Wannier-equation solver for exciton binding energies.

Solves the radial Schroedinger equation
    [ -hbar^2/(2 mu*) (d^2/dr^2 + (1/r) d/dr - l^2/r^2) + V(r) ] R(r)
        = -E_b R(r)
in effective-mass approximation.  Using u(r) = sqrt(r) R(r) reduces this to
    [ -hbar^2/(2 mu*) d^2/dr^2 + (l^2 - 1/4) hbar^2/(2 mu* r^2) + V(r) ] u
        = -E_b u,
which is a standard 1D eigenvalue problem.  For l = 0 the centrifugal term
becomes -hbar^2/(8 mu* r^2), which is *attractive* (a purely mathematical
consequence of the substitution) but the resulting operator still has a
well-defined ground state on any finite [r_min, r_max] with Dirichlet
boundary conditions.

We use a dense eigensolver (numpy.linalg.eigh) on a uniform r-grid for
robustness — n_r ~ 2000 is more than enough for the RK/ERK ground state.

Constant: hbar^2 / (2 m_e) = 3.80998 eV * A^2.
"""
from __future__ import annotations
import numpy as np


def wannier_binding_2d(potential,
                       mu_over_me: float = 0.20,
                       r_min: float = 0.10,     # A
                       r_max: float = 400.0,    # A
                       n_r: int = 2500,
                       l: int = 0,
                       n_states: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (binding_energies_eV, radial_wavefunctions).

    binding_energies[k] > 0 for bound states, sorted deepest first.
    """
    hbar2_over_2mu = 3.80998 / mu_over_me    # eV * A^2

    r = np.linspace(r_min, r_max, n_r)
    dr = r[1] - r[0]

    # -hbar^2/(2 mu*) d^2/dr^2  (finite-difference, second order)
    diag_T = np.full(n_r, 2.0) * hbar2_over_2mu / (dr * dr)
    off_T = np.full(n_r - 1, -1.0) * hbar2_over_2mu / (dr * dr)

    # Centrifugal + substitution: + (l^2 - 1/4) hbar^2/(2 mu*) * 1/r^2
    cent = (l * l - 0.25) * hbar2_over_2mu / (r * r)

    # Potential
    Vr = potential.V_r(r)

    diag_H = diag_T + cent + Vr

    H = np.diag(diag_H) + np.diag(off_T, 1) + np.diag(off_T, -1)

    # Dirichlet boundary conditions: enforce u = 0 at both ends
    # (already implicit in the finite-difference stencil if we drop endpoints
    # from the eigenvalue problem, but we keep them for simplicity)

    eigval, eigvec = np.linalg.eigh(H)

    # Keep lowest n_states
    eigval = eigval[:n_states]
    eigvec = eigvec[:, :n_states]

    # Binding energies
    E_b = -eigval

    # Undo substitution: R(r) = u(r) / sqrt(r)
    R = eigvec / np.sqrt(r)[:, None]
    # Normalise on the physical radial measure  int 2 pi r |R|^2 dr = 1
    for k in range(R.shape[1]):
        norm = np.trapezoid(2.0 * np.pi * r * np.abs(R[:, k])**2, r)
        if norm > 0:
            R[:, k] /= np.sqrt(norm)

    return E_b, R
