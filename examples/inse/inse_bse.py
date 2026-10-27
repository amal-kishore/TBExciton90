"""Monolayer InSe BSE: RK and extended-RK potentials, exciton spectrum,
and absorption.

Uses the Wannier90 outputs shipped in ``examples/inse/wannier_data/`` and
reads DFT eigenvalues from the Wannier90 ``.eig`` file.

Run::

    python3 examples/inse/inse_bse.py
    python3 examples/inse/inse_bse.py --cpu
"""
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tbexciton90.bse import (build_setup, build_bse_hamiltonian,
                             diagonalize, optical_absorption)
from tbexciton90.potentials import RytovaKeldysh, ExtendedRytovaKeldysh
from tbexciton90.io import save_results


HERE = os.path.dirname(os.path.abspath(__file__))
WAN_DIR = os.path.join(HERE, "wannier_data")
BANDS   = os.path.join(WAN_DIR, "inse.eig")
OUT_ROOT = "results_inse"

# BSE selection
#   HOMO = DFT band 19, LUMO = DFT band 20 (0-indexed 18, 19).
#   Use top-2 VB + bottom-2 CB.
VB_WANN = (17, 19)
CB_WANN = (19, 21)
DFT_SKIP = 0           # .eig covers the Wannier subspace already

# Potential parameters for monolayer InSe (vacuum)
#   r_0 ~ 10.5 A         (Sohier, Marzari, Gibertini, PRB 96, 075448, 2017)
#   a   =  4.074 A       (in-plane lattice constant from inse.win)
R0 = 10.5
EPS_ENV = 1.0
A_LATT = 4.074


def run_case(setup, potential, tag, photon_E, sigma_G, prefer_gpu):
    print(f"\n----- {tag}: {potential!r} -----")
    t0 = time.time()
    H = build_bse_hamiltonian(setup, potential,
                              eps_env_exchange=EPS_ENV,
                              include_exchange=True,
                              prefer_gpu=prefer_gpu, verbose=True)
    print(f"  H_eh built: shape {H.shape}, {time.time()-t0:.2f}s")

    t0 = time.time()
    eigval, eigvec = diagonalize(H)
    print(f"  Diagonalised in {time.time()-t0:.2f}s ({len(eigval)} eigenvalues)")

    E_gap  = setup.transition_gap_direct
    E_bind = E_gap - eigval[0]
    print(f"  Direct DFT gap      = {E_gap*1000:.1f} meV")
    print(f"  Omega_1s            = {eigval[0]*1000:.1f} meV")
    print(f"  Exciton binding E_b = {E_bind*1000:.1f} meV")

    Nk = setup.kpoints_cart.shape[0]
    Nv = setup.C_vb.shape[1]
    Nc = setup.C_cb.shape[1]
    sigma_xx = optical_absorption(eigval, eigvec, photon_E,
                                  Nc, Nv, Nk, sigma=sigma_G)

    outdir = os.path.join(OUT_ROOT, tag)
    save_results(outdir, eigval, eigvec, photon_E, sigma_xx, H=H)
    print(f"  Results -> {outdir}")
    return dict(eigval=eigval, eigvec=eigvec, sigma_xx=sigma_xx,
                E_gap=E_gap, E_bind=E_bind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    prefer_gpu = not args.cpu
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("Loading Wannier90 + .eig data ...")
    setup = build_setup(seedname="inse",
                        bands_file=BANDS,
                        vb_wann_range=VB_WANN,
                        cb_wann_range=CB_WANN,
                        dft_skip_bands=DFT_SKIP,
                        wan_dir=WAN_DIR)
    Nk = setup.kpoints_cart.shape[0]
    Nv = setup.C_vb.shape[1]
    Nc = setup.C_cb.shape[1]
    print(f"  n_wann = {setup.n_wann}, atoms = {[g[0] for g in setup.atom_groups]}")
    print(f"  k-grid = {setup.mp_grid} ({Nk} kpts)")
    print(f"  BSE dim = {Nc}*{Nv}*{Nk} = {Nc*Nv*Nk}")

    photon_E = np.linspace(0.5, 3.0, 500)
    sigma_G = 0.010

    V_RK = RytovaKeldysh(r0=R0, eps_env=EPS_ENV, a0=A_LATT)
    res_RK = run_case(setup, V_RK, "rk", photon_E, sigma_G, prefer_gpu)

    V_ERK = ExtendedRytovaKeldysh(r0=R0, q_s=0.030, eps_env=EPS_ENV, a0=A_LATT)
    res_ERK = run_case(setup, V_ERK, "erk", photon_E, sigma_G, prefer_gpu)

    print("\n=========== SUMMARY ===========")
    print(f"  {'':24s}   {'RK':>10s}   {'ERK':>10s}")
    print(f"  {'Omega_1s (meV)':24s}   {res_RK['eigval'][0]*1000:>10.1f}   "
          f"{res_ERK['eigval'][0]*1000:>10.1f}")
    print(f"  {'E_bind (meV)':24s}   {res_RK['E_bind']*1000:>10.1f}   "
          f"{res_ERK['E_bind']*1000:>10.1f}")
    print(f"  Screening-induced binding shift = "
          f"{(res_ERK['E_bind'] - res_RK['E_bind'])*1000:+.1f} meV")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

    ax = axes[0]
    n_show = 15
    xs = np.arange(n_show)
    ax.plot(xs, res_RK["eigval"][:n_show], "x", color="tab:blue", ms=8,
            mew=2, label="RK")
    ax.plot(xs, res_ERK["eigval"][:n_show], "s", color="tab:red", ms=5,
            label=r"ERK ($q_s = 0.03\ \mathrm{\AA}^{-1}$)")
    ax.axhline(res_RK["E_gap"], ls=":", color="0.4",
               label=f"DFT gap = {res_RK['E_gap']:.3f} eV")
    ax.set_xlabel("Exciton index $S$")
    ax.set_ylabel(r"$\Omega_S$ (eV)")
    ax.set_title(f"InSe: exciton spectrum (Nk={Nk})")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    idx = np.arange(2)
    w = 0.35
    ax.bar(idx - w/2, [res_RK["E_bind"]*1000, res_ERK["E_bind"]*1000], w,
           color=["tab:blue", "tab:red"], edgecolor="k", alpha=0.85)
    for i, v in enumerate([res_RK["E_bind"]*1000, res_ERK["E_bind"]*1000]):
        ax.text(idx[i] - w/2, v + 5, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(idx, ["RK", r"ERK ($q_s = 0.03$)"])
    ax.set_ylabel(r"1s binding energy $E_b$ (meV)")
    ax.set_title("Exciton binding energy")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.plot(photon_E, res_RK["sigma_xx"],  "-",  color="tab:blue", lw=1.6, label="RK")
    ax.plot(photon_E, res_ERK["sigma_xx"], "--", color="tab:red",  lw=1.6, label="ERK")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"Absorption $\sigma_{xx}$ (arb.)")
    ax.set_title(rf"Optical absorption ($\sigma_G = {sigma_G*1000:.0f}$ meV)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(photon_E[0], photon_E[-1])

    fig_path = os.path.join(OUT_ROOT, "inse_rk_vs_erk.png")
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()
