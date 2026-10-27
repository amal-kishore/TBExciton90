"""Exciton band structure of monolayer WSe2 along Gamma -> M -> K -> Gamma.

Uses the finite-Q BSE (Wannier interpolation of electron bands at k+Q) to
compute Omega_S(Q) along the standard hexagonal 2D BZ path, and plots the
lowest exciton dispersions.

Run::

    python3 examples/wse2/wse2_exciton_bandstructure.py
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

MP_FINE  = (30, 30, 1)
N_STATES = 8
N_SEG    = 24


HS_POINTS = {
    "$\\Gamma$": np.array([0.0, 0.0, 0.0]),
    "M":         np.array([0.5, 0.0, 0.0]),
    "K":         np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
}
PATH = ["$\\Gamma$", "M", "K", "$\\Gamma$"]


def build_fine_from_coarse(coarse):
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


def build_path(hs_points_frac, labels, n_seg, recip_basis):
    """Return Q_frac, cumulative-cartesian x, tick positions, tick labels."""
    Q_frac_list = []
    tick_pos = [0.0]
    cum = 0.0
    for iseg, (lab_a, lab_b) in enumerate(zip(labels[:-1], labels[1:])):
        Qa, Qb = hs_points_frac[lab_a], hs_points_frac[lab_b]
        offset_start = 0 if iseg == 0 else 1
        for i in range(offset_start, n_seg + 1):
            t = i / n_seg
            Q_frac_list.append((1 - t) * Qa + t * Qb)
        cum += np.linalg.norm((Qb - Qa) @ recip_basis)
        tick_pos.append(cum)

    Q_frac = np.stack(Q_frac_list, axis=0)
    Q_cart = Q_frac @ recip_basis
    dQ = np.diff(Q_cart, axis=0)
    x_along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(dQ, axis=1))])
    return Q_frac, x_along, np.array(tick_pos), labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading Wannier90 + QE data ...")
    coarse = build_setup(seedname="WSe2", bands_file=BANDS,
                         vb_wann_range=VB_WANN, cb_wann_range=CB_WANN,
                         dft_skip_bands=DFT_SKIP, wan_dir=WAN_DIR)
    fine = build_fine_from_coarse(coarse)
    Nk = fine.kpoints_cart.shape[0]
    print(f"  coarse n_wann={coarse.n_wann} | fine Nk={Nk} | MP={MP_FINE}")

    B = 2.0 * np.pi * np.linalg.inv(coarse.lattice.T)
    Q_frac, x_along, tick_pos, tick_labels = build_path(HS_POINTS, PATH,
                                                        n_seg=N_SEG,
                                                        recip_basis=B)
    print(f"\nQ-path: {' -> '.join(tick_labels)}  ({Q_frac.shape[0]} points)")

    V = RytovaKeldysh(r0=R0, eps_env=EPS_ENV, a0=A_LATT)
    t0 = time.time()
    Q_arr, Omega = compute_exciton_dispersion(
        fine, coarse, V, list(Q_frac),
        n_states=N_STATES, prefer_gpu=True, verbose=False,
    )
    print(f"  {Q_arr.shape[0]} BSE diagonalisations in {time.time()-t0:.1f}s")
    print(f"  Omega_1s(Gamma) = {Omega[0, 0]:.4f} eV")

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for s in range(N_STATES):
        color = cmap(s / max(N_STATES - 1, 1))
        ax.plot(x_along, Omega[:, s], "-", color=color, lw=1.8,
                label=f"S={s}" if s < 4 else None)
    for xt in tick_pos:
        ax.axvline(xt, color="k", lw=0.5, alpha=0.4)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Exciton centre-of-mass momentum Q")
    ax.set_ylabel(r"$\Omega_S(Q)$ (eV)")
    ax.set_title(f"WSe$_2$ monolayer: exciton band structure "
                 f"(MP={MP_FINE[0]}x{MP_FINE[1]})")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig_path = os.path.join(OUT_DIR, "wse2_exciton_bandstructure.png")
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")

    npz_path = os.path.join(OUT_DIR, "wse2_exciton_bandstructure.npz")
    np.savez(npz_path, Q_frac=Q_arr, x_along=x_along, Omega=Omega,
             tick_pos=tick_pos, tick_labels=tick_labels)
    print(f"Data saved:   {npz_path}")


if __name__ == "__main__":
    main()
