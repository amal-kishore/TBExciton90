"""Dense-grid BSE on WSe2 with real-space exciton densities.

Interpolates H(R) onto a 21x21x1 grid, solves the BSE with RK, and plots
the oscillator strengths together with the lowest four exciton envelope
functions |Psi_S(r_e - r_h)|^2 in real space.

Run::

    python3 examples/wse2/wse2_density.py
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from tbexciton90.bse import (build_setup, build_bse_hamiltonian, diagonalize,
                             interpolate_to_fine_grid)
from tbexciton90.potentials import RytovaKeldysh
from tbexciton90.io import save_results


HERE = os.path.dirname(os.path.abspath(__file__))
WAN_DIR = os.path.join(HERE, "wannier_data")
BANDS   = os.path.join(WAN_DIR, "Bands_QE")
OUT_ROOT = "results_wse2_dense"

VB_WANN = (12, 14)
CB_WANN = (14, 16)
DFT_SKIP = 12
R0 = 45.09
EPS_ENV = 1.0
A_LATT = 3.32

MP_FINE = (21, 21, 1)      # 441 kpts, BSE dim = 2*2*441 = 1764

R_MAX = 20.0               # Angstrom
N_PIX = 121                # density image resolution


def build_fine_setup():
    print("Loading coarse Wannier setup ...")
    coarse = build_setup(seedname="WSe2",
                         bands_file=BANDS,
                         vb_wann_range=VB_WANN,
                         cb_wann_range=CB_WANN,
                         dft_skip_bands=DFT_SKIP,
                         wan_dir=WAN_DIR)
    print(f"  coarse:  n_wann={coarse.n_wann}  Nk={coarse.kpoints_cart.shape[0]}")

    print(f"\nInterpolating to fine grid {MP_FINE} ...")
    from tbexciton90.bse.interpolate import interpolate_to_fine_grid as _interp
    tmp = _interp(coarse, MP_FINE, vb_bloch_range=(0, coarse.n_wann),
                  cb_bloch_range=(0, coarse.n_wann))
    E_all = tmp.E_vk
    gaps = np.min(E_all[1:], axis=1) - np.max(E_all[:-1], axis=1)
    ivb_top = int(np.argmax(gaps))
    icb_bot = ivb_top + 1
    print(f"  VBM band index = {ivb_top}  (E_max = {E_all[ivb_top].max():.3f} eV)")
    print(f"  CBM band index = {icb_bot}  (E_min = {E_all[icb_bot].min():.3f} eV)")

    vb_fine = (ivb_top - 1, ivb_top + 1)
    cb_fine = (icb_bot, icb_bot + 2)
    fine = interpolate_to_fine_grid(coarse, MP_FINE,
                                    vb_bloch_range=vb_fine,
                                    cb_bloch_range=cb_fine)
    print(f"  fine:    n_wann={fine.n_wann}  Nk={fine.kpoints_cart.shape[0]}")
    print(f"  min direct gap = {fine.transition_gap_direct:.4f} eV")
    return fine


def exciton_density_smooth(A_S, k_cart, r_max=R_MAX, n_pix=N_PIX):
    """|Psi_S(r)|^2 on a fine 2D real-space grid.

        Psi_S(r) = sum_k phi_S(k) exp(i k . r),
        phi_S(k) = sum_{c,v} A^S_{c,v,k}
    """
    phi_k = A_S.sum(axis=(0, 1))
    xs = np.linspace(-r_max, r_max, n_pix)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    kx, ky = k_cart[:, 0], k_cart[:, 1]
    phase = np.exp(1j * (kx[:, None, None] * XX[None, :, :]
                         + ky[:, None, None] * YY[None, :, :]))
    Psi = np.einsum("k,kmn->mn", phi_k, phase)
    return XX, YY, np.abs(Psi) ** 2


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    fine = build_fine_setup()

    print("\nBuilding BSE Hamiltonian on fine grid ...")
    t0 = time.time()
    V = RytovaKeldysh(r0=R0, eps_env=EPS_ENV, a0=A_LATT)
    H = build_bse_hamiltonian(fine, V, prefer_gpu=True, verbose=True)
    print(f"  H shape {H.shape} built in {time.time()-t0:.1f}s")

    t0 = time.time()
    eigval, eigvec = diagonalize(H)
    print(f"  Diagonalised in {time.time()-t0:.1f}s")

    Nc = fine.C_cb.shape[1]
    Nv = fine.C_vb.shape[1]
    Nk = fine.kpoints_cart.shape[0]
    print(f"  Nex = {Nc*Nv*Nk}")
    print(f"  Omega_1 = {eigval[0]:.4f} eV   "
          f"gap = {fine.transition_gap_direct:.4f} eV   "
          f"E_b = {(fine.transition_gap_direct - eigval[0])*1000:.1f} meV")

    save_results(OUT_ROOT, eigval, eigvec,
                 np.linspace(0.3, 2.5, 500), np.zeros(500))

    # Constant-dipole approximation to the oscillator strength
    O = np.abs(eigvec.sum(axis=0)) ** 2

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.4])

    ax = fig.add_subplot(gs[0, :])
    Onorm = O / O.max()
    ax.vlines(eigval, 0, Onorm, colors="tab:blue", lw=0.7)
    for S in range(4):
        ax.annotate(f"S={S}", xy=(eigval[S], Onorm[S]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color="tab:red")
    ax.set_xlabel(r"Exciton energy $\Omega_S$ (eV)")
    ax.set_ylabel("Oscillator strength (normalised)")
    ax.set_title(f"WSe$_2$ (RK, MP={MP_FINE[0]}x{MP_FINE[1]}): "
                 f"all {len(eigval)} excitons")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    k_cart_2d = fine.kpoints_cart[:, :2]
    for i, S in enumerate([0, 1, 2, 3]):
        A_S = eigvec[:, S].reshape(Nc, Nv, Nk)
        XX, YY, rho = exciton_density_smooth(A_S, k_cart_2d)
        ax = fig.add_subplot(gs[1, i])
        im = ax.pcolormesh(XX, YY, rho / rho.max(), cmap="viridis",
                           shading="auto", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-R_MAX, R_MAX)
        ax.set_ylim(-R_MAX, R_MAX)
        ax.set_xlabel(r"$r_x$ ($\mathrm{\AA}$)")
        if i == 0:
            ax.set_ylabel(r"$r_y$ ($\mathrm{\AA}$)")
        ax.set_title(f"S = {S}   Omega = {eigval[S]*1000:.0f} meV")
        fig.colorbar(im, ax=ax, shrink=0.8, label="density (norm.)")

    out = os.path.join(OUT_ROOT, "wse2_oscillator_density.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
