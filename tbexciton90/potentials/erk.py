"""Extended Rytova-Keldysh (ERK) potential.

Derived by simultaneously including bound (lattice-polarisation) and free
(carrier) surface charges on a 2D sheet.  Momentum space:

    V_ERK(q) = -2 pi e^2 / [ eps_env * (r_0 q^2 + q + q_s) ]        (eV * A^2)

where q_s = e^2 |Pi_static| / (2 eps_0 eps_env) is the free-carrier
Thomas-Fermi wavevector (angstrom^-1).

For 4 r_0 q_s < 1 the denominator factorises with real, positive roots
    A, B = (1 +/- sqrt(1 - 4 r_0 q_s)) / (2 r_0),
and the real-space form has the closed expression

    V_ERK(r) = -(pi e^2) / (2 eps_env r_0) * (1 / (A - B))
             * [ A F(A r) - B F(B r) ],    F(x) = H_0(x) - Y_0(x).

Limits:
    q_s -> 0 :  recovers Rytova-Keldysh.
    r_0 -> 0 :  recovers 2D Thomas-Fermi V(q) = -2 pi e^2 / [ eps_env (q + q_s) ].

For 4 r_0 q_s >= 1 the roots become complex conjugates; V_ERK is still real
(via the same closed form using complex Struve/Bessel from scipy.special).
We fall back to a numerical 2D Fourier transform in that regime.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
from scipy import special, integrate

from .base import Potential
from ..constants import EV_A_COULOMB


class ExtendedRytovaKeldysh(Potential):
    """Extended Rytova-Keldysh with free-carrier screening q_s."""

    name = "erk"

    def __init__(self, r0: float, q_s: float,
                 eps_env: float = 1.0, a0: float | None = None):
        """
        Parameters
        ----------
        r0
            2D screening length (angstrom).
        q_s
            Free-carrier Thomas-Fermi wavevector (1/angstrom).
        eps_env
            Average environmental dielectric constant.
        a0
            On-site regulariser length (angstrom). If None, use r0/10.
        """
        self.r0 = float(r0)
        self.q_s = float(q_s)
        self.eps_env = float(eps_env)
        self.a0 = float(a0) if a0 is not None else 0.1 * self.r0

    @property
    def params(self) -> Dict[str, float]:
        return {"r0": self.r0, "q_s": self.q_s,
                "eps_env": self.eps_env, "a0": self.a0}

    # ------------------------------------------------------------------
    def V_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        pref = -2.0 * np.pi * EV_A_COULOMB / self.eps_env
        denom = self.r0 * q * q + q + self.q_s
        # denom > 0 everywhere for physical parameters
        return pref / denom

    # ------------------------------------------------------------------
    def V_r(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        r_eff = np.where(r < self.a0, self.a0, r)
        disc = 1.0 - 4.0 * self.r0 * self.q_s
        if disc > 0:
            # Real roots — use closed form
            sqrt_disc = np.sqrt(disc)
            A = 0.5 * (1.0 + sqrt_disc) / self.r0
            B = 0.5 * (1.0 - sqrt_disc) / self.r0
            F_A = special.struve(0, A * r_eff) - special.y0(A * r_eff)
            F_B = special.struve(0, B * r_eff) - special.y0(B * r_eff)
            pref = -np.pi * EV_A_COULOMB / (2.0 * self.eps_env * self.r0)
            return pref / (A - B) * (A * F_A - B * F_B)
        else:
            # Complex roots — fall back to numerical Hankel transform
            return self._V_r_numerical(r_eff)

    def _V_r_numerical(self, r_eff: np.ndarray) -> np.ndarray:
        """Numerical inverse 2D Fourier transform for cases with complex roots."""
        out = np.zeros_like(r_eff, dtype=float)
        it = np.nditer(r_eff, flags=["multi_index"])
        for r_scalar in it:
            r_scalar = float(r_scalar)
            def integrand(q, r=r_scalar):
                return self.V_q(np.array([q]))[0] * special.j0(q * r) * q
            val, _ = integrate.quad(integrand, 1e-8, 200.0 / max(r_scalar, 1e-3),
                                    limit=400, epsabs=1e-9, epsrel=1e-7)
            out[it.multi_index] = val / (2.0 * np.pi)
        return out
