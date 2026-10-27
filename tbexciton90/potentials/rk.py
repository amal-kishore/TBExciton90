"""Rytova-Keldysh potential for excitons in a thin 2D dielectric slab.

Convention (dielectrically symmetric environment eps_env above and below the sheet):

    V_RK(q) = -2 pi e^2 / [ eps_env * q * (1 + r_0 * q) ]     (eV * A^2)

    V_RK(r) = -(pi e^2) / (2 eps_env r_0) *
              [ H_0(r/r_0) - Y_0(r/r_0) ]                     (eV)

where H_0 is the Struve function of order zero and Y_0 is the Bessel function
of the second kind, order zero.  r_0 is the 2D screening length in angstrom.

The on-site (r = 0) value is regularised by the substitution r -> a_0 (the
lattice constant), following the standard convention in the moire-BSE
literature (Phys. Rev. B 91, 075310 (2015); Maity et al. npj 2D 9, 20).
"""
from __future__ import annotations

from typing import Dict
import numpy as np
from scipy import special

from .base import Potential
from ..constants import EV_A_COULOMB


class RytovaKeldysh(Potential):
    """Isotropic Rytova-Keldysh potential (Rytova 1965; Keldysh 1979)."""

    name = "rk"

    def __init__(self, r0: float, eps_env: float = 1.0, a0: float | None = None):
        """
        Parameters
        ----------
        r0
            2D screening length (angstrom).
        eps_env
            Average environmental dielectric constant (eps_env = (eps_1+eps_2)/2).
        a0
            On-site regulariser length (angstrom).  If None, use r0/10 as a default.
        """
        self.r0 = float(r0)
        self.eps_env = float(eps_env)
        self.a0 = float(a0) if a0 is not None else 0.1 * self.r0

    @property
    def params(self) -> Dict[str, float]:
        return {"r0": self.r0, "eps_env": self.eps_env, "a0": self.a0}

    def V_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        pref = -2.0 * np.pi * EV_A_COULOMB / self.eps_env
        out = np.zeros_like(q)
        mask = q > 0
        out[mask] = pref / (q[mask] * (1.0 + self.r0 * q[mask]))
        return out

    def V_r(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        pref = -np.pi * EV_A_COULOMB / (2.0 * self.eps_env * self.r0)
        # Regularise on-site
        r_eff = np.where(r < self.a0, self.a0, r)
        x = r_eff / self.r0
        return pref * (special.struve(0, x) - special.y0(x))
