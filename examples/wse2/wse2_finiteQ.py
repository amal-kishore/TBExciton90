"""Finite-Q exciton dispersion of monolayer WSe2.

Builds the BSE Hamiltonian at a sequence of centre-of-mass momenta Q along
Gamma -> M in the 2D BZ, using Wannier interpolation to evaluate the
electron bands and coefficients at k+Q.

Run::

    python3 examples/wse2/wse2_finiteQ.py
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tbexciton90.bse import (build_setup, build_bse_hamiltonian, diagonalize,
                             interpolate_to_fine_grid, compute_exciton_dispersion)
from tbexciton90.bse.interpolate import interpolate_to_fine_grid as _interp
from tbexciton90.potentials import RytovaKeldysh


HERE = os.path.dirname(os.path.abspath(__file__))
WAN_DIR = os.path.join(HERE, "wannier_data")
BANDS   = os.path.join(WAN_DIR, "Bands_QE")
OUT_DIR = "results_wse2"

VB_WANN = (12, 14)
CB_WANN = (14, 16)
DFT_SKIP = 12
R0 = 45.09
EPS_ENV = 1.0
A_LATT = 3.32

MP_FINE  = (12, 12, 1)   # fine k-grid for the finite-Q sweep
N_Q      = 7             # points along Gamma -> M
N_STATES = 6             # lowest exciton states to track


def build_fine_from_coarse(coarse):
    """Auto-detect VBM/CBM in the interpolated bands and keep 2+2 bands."""
    tmp = _interp(coarse, MP_FINE, vb_bloch_range=(0, coarse.n_wann),
                  cb_bloch_range=(0, coarse.n_wann))
    E_all = tmp.E_vk
    gaps = np.min(E_all[1:], axis=1) - np.max(E_all[:-1], axis=1)
    ivb_top = int(np.argmax(gaps))
    icb_bot = ivb_top + 1
    vb_fine = (ivb_top - 1, ivb_top + 1)
    cb_fine = (icb_bot, icb_bot + 2)
    return interpolate_to_fine_grid(coarse, MP_FINE,
                                    vb_bloch_range=vb_fine,
                                    cb_bloch_range=cb_fine)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Wannier90 + QE data ...")
    coarse = build_setup(seedname="WSe2", bands_file=BANDS,
                         vb_wann_range=VB_WANN, cb_wann_range=CB_WANN,
                         dft_skip_bands=DFT_SKIP, wan_dir=WAN_DIR)
    fine = build_fine_from_coarse(coarse)
    Nk = fine.kpoints_cart.shape[0]
    print(f"  coarse n_wann={coarse.n_wann} | fine Nk={Nk} | MP={MP_FINE}")

    V = RytovaKeldysh(r0=R0, eps_env=EPS_ENV, a0=A_LATT)

    print("\n[a] Q = 0  (direct fast path)")
    t0 = time.time()
    H0 = build_bse_hamiltonian(fine, V, prefer_gpu=True, verbose=False, Q_frac=None)
    eig0, _ = diagonalize(H0)
    print(f"    Omega_1s(Q=0) = {eig0[0]:.4f} eV   ({time.time()-t0:.1f}s)")

    print("\n[b] Q = 0  via explicit interpolation (consistency check)")
    t0 = time.time()
    H0b = build_bse_hamiltonian(fine, V, prefer_gpu=True, verbose=False,
                                Q_frac=np.zeros(3), coarse_setup=coarse)
    eig0b, _ = diagonalize(H0b)
    print(f"    Omega_1s(Q=0, via interp) = {eig0b[0]:.4f} eV   "
          f"({time.time()-t0:.1f}s)")
    print(f"    |delta Omega_1s (Q=0)| = {abs(eig0b[0]-eig0[0])*1000:.3f} meV")

    # Dispersion along Gamma -> M; stay in the small-Q (parabolic) regime
    Q_max_frac = 0.15
    Q_list = [np.array([q, 0.0, 0.0]) for q in np.linspace(0.0, Q_max_frac, N_Q)]
    print(f"\n[c] Finite-Q sweep along Gamma -> M, {N_Q} points, "
          f"Q_max_frac={Q_max_frac}")
    t0 = time.time()
    Q_arr, Omega = compute_exciton_dispersion(
        fine, coarse, V, Q_list,
        n_states=N_STATES, prefer_gpu=True, verbose=True,
    )
    print(f"  dispersion done in {time.time()-t0:.1f}s")

    # Parabolic fit:  Omega_1s(Q) ~ Omega_1s(0) + (hbar^2 Q^2) / (2 M_X)
    HBAR2_OVER_2M0_eV_A2 = 3.80998   # (hbar^2 / 2 m_e) in eV * A^2
    B = 2.0 * np.pi * np.linalg.inv(coarse.lattice.T)
    Q_cart = Q_arr @ B
    Qmag = np.linalg.norm(Q_cart, axis=1)
    dE = Omega[:, 0] - Omega[0, 0]
    small = Qmag < 0.17   # 1/A — small enough for the quadratic approximation
    if small.sum() >= 3:
        coef = np.polyfit(Qmag[small] ** 2, dE[small], 1)
        M_X_over_m0 = HBAR2_OVER_2M0_eV_A2 / coef[0] if coef[0] > 0 else float("nan")
        print(f"\n  Parabolic fit near Q=0:  M_X ~ {M_X_over_m0:.2f} m_e")
    else:
        M_X_over_m0 = float("nan")
        print("  Too few small-Q points for mass fit.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for s in range(min(3, N_STATES)):
        ax.plot(Qmag, Omega[:, s] - Omega[0, 0], "o-",
                label=f"S={s}")
    ax.set_xlabel(r"$|Q|$ (1/$\mathrm{\AA}$) along $\Gamma \to M$")
    ax.set_ylabel(r"$\Omega_S(Q) - \Omega_S(0)$ (eV)")
    title = (f"WSe$_2$ exciton dispersion "
             f"(BSE, MP={MP_FINE[0]}x{MP_FINE[1]})")
    if not np.isnan(M_X_over_m0):
        title += f",  $M_X \\approx {M_X_over_m0:.2f}\\,m_e$"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig_path = os.path.join(OUT_DIR, "wse2_finiteQ_dispersion.png")
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    npz_path = os.path.join(OUT_DIR, "wse2_finiteQ_dispersion.npz")
    np.savez(npz_path, Q_frac=Q_arr, Q_cart=Q_cart, Omega=Omega,
             Qmag_invA=Qmag, M_X_over_m0=M_X_over_m0)
    print(f"Data saved:   {npz_path}")


if __name__ == "__main__":
    main()
