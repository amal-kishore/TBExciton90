"""Abstract base class for a 2D screened electron-hole interaction potential.

A potential provides:
    V_q(q)   :  momentum-space form,  scalar or ndarray input
    V_r(r)   :  real-space form,      scalar or ndarray input
    name     :  short label used in output files
    params   :  dict of parameters, saved with results for reproducibility

All energies are in eV; all lengths are in angstrom.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class Potential(ABC):
    """Abstract 2D screened Coulomb-like potential (attractive, V < 0)."""

    name: str = "base"

    @abstractmethod
    def V_q(self, q: np.ndarray) -> np.ndarray:
        """Momentum-space potential in eV*A^2 (Fourier convention V(q) = int d^2r e^{-iq.r} V(r))."""

    @abstractmethod
    def V_r(self, r: np.ndarray) -> np.ndarray:
        """Real-space potential in eV, evaluated at in-plane separation r (A)."""

    @property
    @abstractmethod
    def params(self) -> Dict[str, float]:
        """Dictionary of parameters, for logging and reproducibility."""

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({p})"
