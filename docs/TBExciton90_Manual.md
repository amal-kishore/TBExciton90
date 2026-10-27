# TBExciton90 Manual

TBExciton90 solves the Bethe-Salpeter equation (BSE) for excitons in a
Wannier / tight-binding basis. The DFT single-particle structure enters
through a Wannier90 output set (`.win`, `.wout`, `_hr.dat`, `_u.mat`,
`_u_dis.mat`); the screened electron-hole interaction is modelled by a
2D Rytova-Keldysh (RK) kernel or its extended form (ERK) that includes
free-carrier screening. Both zero and finite centre-of-mass momentum Q
are supported.


## 1. Physics

### 1.1 BSE in a Wannier basis

For a two-band model with valence-band index v and conduction-band index
c on a Monkhorst-Pack k-grid, the BSE Hamiltonian at centre-of-mass
momentum Q is

```
   H^Q_{cvk, c'v'k'} = (E_{c, k+Q} - E_{v, k}) delta_{cc'} delta_{vv'} delta_{kk'}
                       + K^{dir, Q}_{cvk, c'v'k'}
                       + K^{xch, Q}_{cvk, c'v'k'} .
```

The direct kernel K^dir uses the statically screened interaction V(q)
projected onto the Wannier basis,

```
   K^dir_{cvk, c'v'k'} = - V(k - k')
                         * < u_{c, k+Q} | u_{c', k'+Q} >
                         * < u_{v, k}   | u_{v', k'}   > .
```

The exchange kernel K^xch uses the bare 2D Coulomb interaction (screened
only by the environment eps_env) and is normally kept at Q = 0 for
singlet excitons.

Band energies E_{c, k+Q} and the overlap matrix elements
< u_{c, k+Q} | ... > are obtained by Wannier interpolation of H(R) and
the rotation matrices from the coarse Wannier90 grid.

### 1.2 Rytova-Keldysh potential (`potentials.RytovaKeldysh`)

For a 2D sheet in a symmetric dielectric environment eps_env,

```
   V_RK(q) = -2 pi e^2 / [ eps_env * q * (1 + r_0 q) ] ,

   V_RK(r) = -(pi e^2) / (2 eps_env r_0) * [ H_0(r / r_0) - Y_0(r / r_0) ] ,
```

where H_0 and Y_0 are the Struve and Bessel-Y functions of order zero,
and r_0 is the 2D screening length (angstrom). The on-site value is
regularised by r -> max(r, a_0), with a_0 defaulting to r_0 / 10.
Keldysh, JETP Lett. 29, 658 (1979); Cudazzo, Tokatly, Rubio, PRB 84,
085406 (2011).

### 1.3 Extended Rytova-Keldysh (`potentials.ExtendedRytovaKeldysh`)

Including a free-carrier Thomas-Fermi contribution q_s on the sheet,

```
   V_ERK(q) = -2 pi e^2 / [ eps_env * (r_0 q^2 + q + q_s) ] .
```

For 4 r_0 q_s < 1 this factorises with real roots A, B and has a closed
real-space form in terms of H_0 - Y_0; otherwise the code falls back to
a numerical 2D Fourier transform. q_s -> 0 recovers RK; r_0 -> 0 reduces
to the 2D Thomas-Fermi limit.

### 1.4 Finite-Q extension

At finite Q the direct kernel still involves V(k - k'), but the
projectors
< u_{c, k+Q} | u_{c', k'+Q} > require the Bloch-to-Wannier
transformation at shifted momenta. These are obtained from the coarse
Wannier90 setup by H(R) interpolation and k-by-k diagonalisation;
see `bse.interpolate.interpolate_at_kpoints` and
`bse.dispersion.compute_exciton_dispersion`.


## 2. Package layout

```
tbexciton90/
    constants.py          BOHR_TO_A, EV_A_COULOMB
    bse/
        wannier_setup.py  WannierSetup container, build_setup()
        hamiltonian.py    build_bse_hamiltonian() -- Q = 0 and finite Q
        solver.py         diagonalize(), unfold_eigenvector()
        absorption.py     optical_absorption() (Lorentzian broadening)
        interpolate.py    interpolate_to_fine_grid(),
                          interpolate_at_kpoints()
        dispersion.py     compute_exciton_dispersion()
        exciton_wf.py     real-space exciton envelopes
    potentials/
        base.py           abstract Potential interface (V_q, V_r)
        rk.py             RytovaKeldysh
        erk.py            ExtendedRytovaKeldysh
    io/
        wan90.py          .win, .wout, _hr.dat, _u.mat, _u_dis.mat
        bands.py          QE `bands.x` plot files and Wannier90 .eig
        hdf5.py           save_results / load_results
    analysis/
        wannier_2d.py     2D Wannier equation (effective-mass reference)
    backends/
        __init__.py       get_backend(), to_numpy() -- numpy / cupy switch
```


## 3. API

### 3.1 Loading a Wannier90 run

```python
from tbexciton90.bse import build_setup

setup = build_setup(
    seedname="WSe2",
    bands_file="path/to/Bands_QE",   # QE bands.x plot file, or .eig
    vb_wann_range=(12, 14),          # [start, stop) Wannier-band indices
    cb_wann_range=(14, 16),
    dft_skip_bands=12,               # DFT = Wannier index + skip
    wan_dir="path/to/wannier",
)
```

`setup` is a `WannierSetup` with the lattice, k-mesh, Hamiltonian H(R),
the Wannier rotations U^k, the DFT valence/conduction eigenvalues
E_{vk}, E_{ck}, and the Bloch-to-Wannier coefficients C_vb, C_cb.
Diagnostic properties: `transition_gap_direct`, `unit_cell_area_A2`.

### 3.2 Building and solving the BSE at Q = 0

```python
from tbexciton90.bse import build_bse_hamiltonian, diagonalize, optical_absorption
from tbexciton90.potentials import RytovaKeldysh
import numpy as np

V = RytovaKeldysh(r0=45.09, eps_env=1.0, a0=3.32)

H = build_bse_hamiltonian(setup, V,
                          eps_env_exchange=1.0,
                          include_exchange=True,
                          prefer_gpu=True, verbose=True)
eigval, eigvec = diagonalize(H)

photon_E = np.linspace(0.3, 2.2, 500)
sigma_xx = optical_absorption(eigval, eigvec, photon_E,
                              Nc=setup.C_cb.shape[1],
                              Nv=setup.C_vb.shape[1],
                              Nk=setup.kpoints_cart.shape[0],
                              sigma=0.010)
```

`eigvec[:, S]` has shape (Nc * Nv * Nk,); reshape to (Nc, Nv, Nk) for
A^S_{cvk}. Oscillator strength in the constant-dipole approximation is
`|eigvec.sum(axis=0)| ** 2`.

### 3.3 Wannier interpolation on a dense grid

```python
from tbexciton90.bse import interpolate_to_fine_grid

fine = interpolate_to_fine_grid(coarse, mp_fine=(21, 21, 1),
                                vb_bloch_range=(ivb_top - 1, ivb_top + 1),
                                cb_bloch_range=(icb_bot, icb_bot + 2))
```

`fine` has the same `WannierSetup` interface; `fine.n_wann` is the
Wannier dimension and `fine.kpoints_cart.shape[0] == prod(mp_fine)`.
H is interpolated as

```
   H(k_fine) = sum_R  H(R) / degen(R) * exp(i k_fine . R) .
```

### 3.4 Finite-Q BSE

```python
from tbexciton90.bse import compute_exciton_dispersion
import numpy as np

Q_frac = [np.array([q, 0.0, 0.0]) for q in np.linspace(0.0, 0.15, 7)]
Q_arr, Omega = compute_exciton_dispersion(
    fine, coarse, V, Q_frac,
    n_states=6, prefer_gpu=True, verbose=True,
)
# Q_arr : (N_Q, 3)   Omega : (N_Q, n_states)
```

Internally this calls `build_bse_hamiltonian(..., Q_frac=Q,
coarse_setup=coarse)` at each Q and diagonalises. A single Q = 0 solve
is also available via `build_bse_hamiltonian(fine, V, Q_frac=None)`,
which uses the direct (no-interpolation) fast path.

For a parabolic small-Q fit of the exciton mass M_X,

```
   Omega_S(Q) ~ Omega_S(0) + (hbar^2 Q^2) / (2 M_X) ,
```

use (hbar^2 / 2 m_e) = 3.80998 eV * A^2 and
`M_X / m_e = (hbar^2 / 2 m_e) / slope`, where slope is the linear
coefficient of Omega_S(Q) vs. |Q|^2.

### 3.5 Real-space exciton envelopes

```python
from tbexciton90.bse.exciton_wf import hole_fixed_exciton_density
# or evaluate directly:
#   Psi_S(r) = sum_k phi_S(k) exp(i k . r),
#   phi_S(k) = sum_{c,v} A^S_{c,v,k} .
```

See `examples/wse2/wse2_density.py` and `examples/inse/inse_density.py`
for both constant-dipole oscillator plots and real-space envelopes
|Psi_S(r_e - r_h)|^2 on a fine Cartesian mesh.

### 3.6 Reference 2D Wannier equation

```python
from tbexciton90.analysis.wannier_2d import wannier_binding_2d

E_b, R_grid = wannier_binding_2d(V, mu_over_me=0.20, l=0, n_states=4)
```

Isotropic finite-difference solve of the 2D hydrogen-like equation for
the given potential and reduced mass. The ground state of the BSE must
reproduce `E_b[0]` in the effective-mass / small-k limit.

### 3.7 I/O

- `tbexciton90.io.save_results(outdir, eigval, eigvec, photon_E, sigma,
  H=...)` writes `Eigval.hdf5`, `Eigvec.hdf5`, `E_ph.hdf5`,
  `SIGMA_full.hdf5` in a PyMEX-compatible layout.
- `tbexciton90.io.load_results(outdir)` reloads them.


## 4. Examples

| Script                                               | What it does |
|------------------------------------------------------|--------------|
| `examples/wse2/wse2_bse.py`                          | Monolayer WSe2 BSE: RK vs. extended RK, exciton spectrum and absorption |
| `examples/wse2/wse2_finiteQ.py`                      | Finite-Q dispersion along Gamma -> M, parabolic exciton-mass fit |
| `examples/wse2/wse2_exciton_bandstructure.py`        | Exciton band structure along Gamma -> M -> K -> Gamma |
| `examples/wse2/wse2_density.py`                      | Dense-grid BSE + real-space envelopes for the lowest four excitons |
| `examples/inse/inse_bse.py`                          | Monolayer InSe BSE (direct gap at Gamma) |
| `examples/inse/inse_density.py`                      | InSe real-space envelopes, individual and multiplet-summed |
| `examples/wannier_baseline.py`                       | 2D Wannier-equation reference solve (effective mass) |

The Wannier90 + QE inputs for WSe2 and InSe are shipped under
`examples/<material>/wannier_data/`, so each script runs without
external data. Outputs land in `results_<material>/` in the current
working directory.


## 5. GPU acceleration

`backends.get_backend(prefer_gpu=True)` returns `cupy` if it is
importable and a device is available; otherwise it falls back to
`numpy`. The BSE Hamiltonian builder and `compute_exciton_dispersion`
accept a `prefer_gpu=True` keyword. For a CUDA-12 pip install,

```
pip install -e .[gpu]
```

For CPU-only runs, pass `--cpu` to the example scripts or
`prefer_gpu=False` to the API.


## 6. Conventions and gotchas

- Energies in electron-volts, lengths in angstrom.
- H(R) stored on the Wannier90 R-grid, divided by the degeneracy
  factor `R_degen` before entering the direct-kernel Fourier sum.
- Wannier/Bloch band ordering: `U` has shape (n_bloch, n_wann, Nk); the
  conjugate-transpose `C = U^dagger` is sliced by Wannier-band index
  with the same `(start, stop)` convention as the published PyMEX
  reference, so `vb_wann_range` and `cb_wann_range` are directly
  reusable from an existing input file.
- `dft_skip_bands` is the number of DFT bands below the Wannier
  subspace (DFT_index = Wannier_index + skip). Use 0 if the bands file
  is a Wannier90 `.eig` (which already covers only the subspace).
- Finite-Q solves require both the `fine` (BSE k-mesh) and `coarse`
  (original Wannier90 grid) setups, because the k+Q coefficients must
  be Wannier-interpolated from H(R) and U^k.
- `include_exchange=True` and `eps_env_exchange=1.0` are the defaults
  used for singlet 2D excitons; set `include_exchange=False` for a pure
  direct-only (triplet) solve.
