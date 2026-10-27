"""Monolayer WSe2 BSE: RK and extended-RK potentials, exciton spectrum,
and optical absorption.

Loads the Wannier90 + QE DFT data shipped in ``examples/wse2/wannier_data/``.
Uses the GPU (cupy) if available, else falls back to numpy transparently.

Run::

    python3 examples/wse2/wse2_bse.py
    python3 examples/wse2/wse2_bse.py --cpu
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
BANDS   = os.path.join(WAN_DIR, "Bands_QE")
OUT_ROOT = "results_wse2"

# Wannier subspace selection, matching the input used for the published
# WSe2 reference calculation:
#   Coefficients:  WAN90   WSe2_u.mat   12 14 0 VB
#   Coefficients:  WAN90   WSe2_u.mat   14 16 0 CB
#   DFT:           QE      Bands_QE     12 14 12 VB   (dft_skip_bands = 12)
#   Interaction:   Keldysh monolayer    eps_env=1.0  r0=45.09  a=3.32
VB_WANN = (12, 14)
CB_WANN = (14, 16)
DFT_SKIP = 12
R0 = 45.09            # Å
EPS_ENV = 1.0
A_LATT = 3.32         # Å (in-plane lattice constant of WSe2)


def run_case(setup, potential, tag, photon_E, sigma_G, prefer_gpu):
    print(f"\n----- {tag}: {potential!r} -----")
    t0 = time.time()
    H = build_bse_hamiltonian(setup, potential,
                              eps_env_exchange=EPS_ENV,
                              include_exchange=True,
                              prefer_gpu=prefer_gpu, verbose=True)
    print(f"  H_eh built:  shape {H.shape}, dtype {H.dtype}, {time.time()-t0:.2f}s")

    t0 = time.time()
    eigval, eigvec = diagonalize(H)
    print(f"  Diagonalised in {time.time()-t0:.2f}s ({len(eigval)} eigenvalues)")
    print("  Lowest 8 excitons (eV):")
    for i in range(8):
        print(f"    Omega_{i+1:<2d} = {eigval[i]:.4f}")

    Nk = setup.kpoints_cart.shape[0]
    Nv = setup.C_vb.shape[1]
    Nc = setup.C_cb.shape[1]
    sigma_xx = optical_absorption(eigval, eigvec, photon_E,
                                  Nc, Nv, Nk, sigma=sigma_G)

    outdir = os.path.join(OUT_ROOT, tag)
    save_results(outdir, eigval, eigvec, photon_E, sigma_xx, H=H)
    print(f"  Results saved to {outdir}")
    return dict(H=H, eigval=eigval, eigvec=eigvec, sigma_xx=sigma_xx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU (numpy) backend even if a GPU is available.")
    args = ap.parse_args()
    prefer_gpu = not args.cpu
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("Loading Wannier90 + QE data ...")
    setup = build_setup(seedname="WSe2",
                        bands_file=BANDS,
                        vb_wann_range=VB_WANN,
                        cb_wann_range=CB_WANN,
                        dft_skip_bands=DFT_SKIP,
                        wan_dir=WAN_DIR)
    print(f"  n_wann = {setup.n_wann}, atoms = {[g[0] for g in setup.atom_groups]}")
    print(f"  k-grid  = {setup.mp_grid} ({setup.kpoints_cart.shape[0]} kpts)")
    print(f"  min direct gap = {setup.transition_gap_direct:.4f} eV")

    photon_E = np.linspace(0.30, 2.20, 500)
    sigma_G = 0.010

    V_RK  = RytovaKeldysh(r0=R0, eps_env=EPS_ENV, a0=A_LATT)
    res_RK = run_case(setup, V_RK,  "rk",  photon_E, sigma_G, prefer_gpu)

    V_ERK = ExtendedRytovaKeldysh(r0=R0, q_s=0.030, eps_env=EPS_ENV, a0=A_LATT)
    res_ERK = run_case(setup, V_ERK, "erk", photon_E, sigma_G, prefer_gpu)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax = axes[0]
    n_show = 15
    xs = np.arange(n_show)
    ax.plot(xs, res_RK["eigval"][:n_show], "x", color="tab:blue", ms=8,
            mew=2, label="RK")
    ax.plot(xs, res_ERK["eigval"][:n_show], "s", color="tab:red", ms=5,
            label=r"ERK ($q_s = 0.03\ \mathrm{\AA}^{-1}$)")
    ax.set_xlabel("Exciton index $S$")
    ax.set_ylabel(r"$\Omega_S$ (eV)")
    ax.set_title(f"WSe$_2$: exciton energies  (Nk={setup.kpoints_cart.shape[0]})")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(photon_E, res_RK["sigma_xx"],  "-",  color="tab:blue", lw=1.6, label="RK")
    ax.plot(photon_E, res_ERK["sigma_xx"], "--", color="tab:red",  lw=1.6, label="ERK")
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(r"Absorption $\sigma_{xx}$ (arb.)")
    ax.set_title(rf"Optical absorption ($\sigma_G = {sigma_G*1000:.0f}$ meV)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    fig_path = os.path.join(OUT_ROOT, "wse2_rk_vs_erk.png")
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()
