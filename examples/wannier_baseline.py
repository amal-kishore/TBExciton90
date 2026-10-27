"""Ground-truth exciton binding energies from the 2D Wannier equation.

Isotropic effective-mass solve for the Rytova-Keldysh (and extended RK)
potentials. The 2D BSE must reproduce these numbers in the
effective-mass / small-k limit.

Run::

    python3 examples/wannier_baseline.py
"""
from tbexciton90.potentials import RytovaKeldysh, ExtendedRytovaKeldysh
from tbexciton90.analysis.wannier_2d import wannier_binding_2d


def main():
    print("2D Wannier equation (effective-mass, isotropic V)")
    print("=" * 66)
    print("  reduced mass mu*/m_e = 0.20   (WSe2-like)")
    print("  Keldysh r_0           = 45.09 A  (vacuum encapsulation)")
    print()

    V_RK = RytovaKeldysh(r0=45.09, eps_env=1.0, a0=1.0)
    E_b_RK, _ = wannier_binding_2d(V_RK, mu_over_me=0.20, l=0, n_states=4)
    print("RK (undoped):")
    for k, e in enumerate(E_b_RK):
        print(f"  n = {k}   binding E_b = {e*1000:8.2f} meV")

    print()
    print("ERK: sweep in q_s (free-carrier screening wavevector, 1/A)")
    print("-" * 66)
    print(f"  {'q_s (1/A)':>12s}  {'E_1s (meV)':>10s}  "
          f"{'E_2s (meV)':>10s}  {'shift 1s':>10s}")
    for q_s in [0.0, 0.001, 0.003, 0.010, 0.030, 0.100]:
        if q_s == 0.0:
            E_b = E_b_RK
        else:
            V_ERK = ExtendedRytovaKeldysh(r0=45.09, q_s=q_s,
                                          eps_env=1.0, a0=1.0)
            E_b, _ = wannier_binding_2d(V_ERK, mu_over_me=0.20,
                                        l=0, n_states=4)
        shift = (E_b[0] - E_b_RK[0]) * 1000
        print(f"  {q_s:12.4f}  {E_b[0]*1000:10.2f}  "
              f"{E_b[1]*1000:10.2f}  {shift:+10.2f} meV")


if __name__ == "__main__":
    main()
