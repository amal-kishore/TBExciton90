"""Real-space exciton envelopes for monolayer InSe.

InSe has a direct gap at Gamma, so no K/K' valley interference: the
envelope Psi(r) = sum_k phi_S(k) exp(+i k . r) is naturally centred around
the origin without a valley filter.

Produces both individual-state densities (S = 0..5) and multiplet-summed
(degeneracy-averaged) densities (M = 0..5).

Run::

    python3 examples/inse/inse_density.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tbexciton90.bse import (build_setup, build_bse_hamiltonian, diagonalize,
                             interpolate_to_fine_grid)
from tbexciton90.potentials import RytovaKeldysh


HERE = os.path.dirname(os.path.abspath(__file__))
WAN_DIR = os.path.join(HERE, "wannier_data")
BANDS   = os.path.join(WAN_DIR, "inse.eig")
OUT_ROOT = "results_inse"
OUT_IND = os.path.join(OUT_ROOT, "inse_density_individual.png")
OUT_SUM = os.path.join(OUT_ROOT, "inse_density_summed.png")

MP_FINE = (17, 17, 1)
ZOOM    = 30.0
N_PIX   = 351

VB_TARGET_BAND = 8        # Wannier band index of the VBM
CB_TARGET_BAND = 9        # Wannier band index of the CBM
N_VB = 2
N_CB = 2


def eval_envelope(A_S, kcart_2d, zoom, n_pix):
    """|Psi(r)|^2 on a dense pixel grid."""
    phi = A_S.sum(axis=(0, 1))
    xs = np.linspace(-zoom, zoom, n_pix)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    kx, ky = kcart_2d[:, 0], kcart_2d[:, 1]
    Psi = np.zeros((n_pix, n_pix), dtype=complex)
    chunk = 32
    for i0 in range(0, len(kx), chunk):
        i1 = min(i0 + chunk, len(kx))
        ph = np.exp(1j * (kx[i0:i1, None, None] * XX[None, :, :]
                          + ky[i0:i1, None, None] * YY[None, :, :]))
        Psi += (phi[i0:i1, None, None] * ph).sum(axis=0)
    return np.abs(Psi) ** 2, XX, YY


def multiplet_indices(eigval, tol_meV=2.0):
    out, i = [], 0
    while i < len(eigval):
        j = i
        while j + 1 < len(eigval) and abs(eigval[j+1] - eigval[i]) * 1000 < tol_meV:
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    coarse = build_setup(seedname="inse", bands_file=BANDS,
                         vb_wann_range=(17, 19), cb_wann_range=(19, 21),
                         dft_skip_bands=0, wan_dir=WAN_DIR)
    fine = interpolate_to_fine_grid(
        coarse, MP_FINE,
        vb_bloch_range=(VB_TARGET_BAND - N_VB + 1, VB_TARGET_BAND + 1),
        cb_bloch_range=(CB_TARGET_BAND, CB_TARGET_BAND + N_CB),
    )
    Nk = fine.kpoints_cart.shape[0]
    print(f"InSe fine grid {MP_FINE}: Nk={Nk}, "
          f"gap = {fine.transition_gap_direct:.3f} eV")

    V = RytovaKeldysh(r0=10.5, eps_env=1.0, a0=4.074)
    H = build_bse_hamiltonian(fine, V, prefer_gpu=True, verbose=True)
    eigval, eigvec = diagonalize(H)
    Nc, Nv = fine.C_cb.shape[1], fine.C_vb.shape[1]
    print(f"BSE dim = {Nc*Nv*Nk}. Omega_1s = {eigval[0]*1000:.1f} meV, "
          f"E_b = {(fine.transition_gap_direct - eigval[0])*1000:.1f} meV")

    kcart_2d = fine.kpoints_cart[:, :2]

    # Individual S = 0..5
    show_S = list(range(6))
    dens_i = []
    for S in show_S:
        A_S = eigvec[:, S].reshape(Nc, Nv, Nk)
        d, XX, YY = eval_envelope(A_S, kcart_2d, ZOOM, N_PIX)
        dens_i.append(d)
        print(f"  S={S}  Omega={eigval[S]*1000:.1f} meV")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, S in enumerate(show_S):
        ax = axes[i // 3, i % 3]
        d = dens_i[i]
        vmax = d.max()
        vmin = vmax * 1e-4
        pcm = ax.pcolormesh(XX, YY, np.clip(d, vmin, None), cmap="Blues",
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            shading="auto")
        ax.plot(0, 0, marker="*", color="#e34948", markersize=16,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-ZOOM, ZOOM)
        ax.set_ylim(-ZOOM, ZOOM)
        ax.set_title(f"S = {S}   Omega = {eigval[S]*1000:.0f} meV", fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")
        if i // 3 == 1:
            ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
        if i == 2:
            fig.colorbar(pcm, ax=ax, shrink=0.7,
                         label=r"$|\Psi_e(r)|^2$ (log, arb.)")

    fig.suptitle(f"InSe individual exciton envelopes "
                 f"(MP={MP_FINE[0]}x{MP_FINE[1]}, hole at *)",
                 weight="bold", fontsize=12)
    fig.savefig(OUT_IND, dpi=170, bbox_inches="tight")
    print(f"Individual saved: {OUT_IND}")

    # Degeneracy-summed M = 0..5
    mult = multiplet_indices(eigval)
    show_M = list(range(min(6, len(mult))))
    dens_m, E_m, deg_m = [], [], []
    for M in show_M:
        lo, hi = mult[M]
        d_sum = np.zeros_like(dens_i[0])
        for S in range(lo, hi):
            A_S = eigvec[:, S].reshape(Nc, Nv, Nk)
            d, _, _ = eval_envelope(A_S, kcart_2d, ZOOM, N_PIX)
            d_sum += d
        dens_m.append(d_sum)
        E_m.append(eigval[lo:hi].mean())
        deg_m.append(hi - lo)

    print("multiplets:")
    for M in show_M:
        print(f"  M={M} deg={deg_m[M]} <Omega>={E_m[M]*1000:.1f} meV")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for i, M in enumerate(show_M):
        ax = axes[i // 3, i % 3]
        d = dens_m[i]
        vmax = d.max()
        vmin = vmax * 1e-4
        pcm = ax.pcolormesh(XX, YY, np.clip(d, vmin, None), cmap="Blues",
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            shading="auto")
        ax.plot(0, 0, marker="*", color="#e34948", markersize=16,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(-ZOOM, ZOOM)
        ax.set_ylim(-ZOOM, ZOOM)
        ax.set_title(f"M = {M}, deg = {deg_m[i]}, "
                     f"<Omega> = {E_m[i]*1000:.0f} meV", fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel(r"$y$ ($\mathrm{\AA}$)")
        if i // 3 == 1:
            ax.set_xlabel(r"$x$ ($\mathrm{\AA}$)")
        if i == 2:
            fig.colorbar(pcm, ax=ax, shrink=0.7,
                         label=r"$|\Psi_e(r)|^2$ (log, arb.)")

    fig.suptitle(f"InSe degeneracy-summed exciton envelopes "
                 f"(MP={MP_FINE[0]}x{MP_FINE[1]}, hole at *)",
                 weight="bold", fontsize=12)
    fig.savefig(OUT_SUM, dpi=170, bbox_inches="tight")
    print(f"Summed saved: {OUT_SUM}")


if __name__ == "__main__":
    main()
