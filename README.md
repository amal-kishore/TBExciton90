# TBExciton90

Bethe-Salpeter solver for excitons in a Wannier / tight-binding basis.
Takes Wannier90 output plus DFT eigenvalues (QE `bands.x` or `*.eig`),
builds the electron-hole Hamiltonian with a Rytova-Keldysh (or extended
RK) screened interaction, and diagonalises it.

The code supports both Q = 0 (optical excitons) and finite centre-of-mass
momentum Q, so the full exciton band structure
Omega_S(Q) along a BZ path can be computed in one run.

A `cupy` GPU backend is used automatically if available; otherwise the
code falls back to `numpy`.

## Install

```
git clone https://github.com/amal-kishore/TBExciton90.git
cd TBExciton90
pip install -e .
```

Optional GPU acceleration:

```
pip install -e .[gpu]
```

## Quick start

Monolayer WSe2, standard RK vs. extended RK, exciton spectrum + absorption:

```
python examples/wse2/wse2_bse.py
```

Finite-Q exciton dispersion along Gamma -> M (and parabolic fit for the
exciton mass):

```
python examples/wse2/wse2_finiteQ.py
```

Exciton band structure along Gamma -> M -> K -> Gamma:

```
python examples/wse2/wse2_exciton_bandstructure.py
```

Dense-grid run with real-space envelopes |Psi_S(r_e - r_h)|^2:

```
python examples/wse2/wse2_density.py
```

Monolayer InSe (direct gap at Gamma):

```
python examples/inse/inse_bse.py
python examples/inse/inse_density.py
```

2D Wannier-equation reference solve (effective-mass, no DFT input needed):

```
python examples/wannier_baseline.py
```

The Wannier90 + QE inputs for WSe2 and InSe are shipped under
`examples/<material>/wannier_data/`, so each script runs out of the box.
Outputs are written to `results_<material>/` in the current directory.

## Package layout

```
tbexciton90/
    bse/            BSE Hamiltonian (Q = 0 and finite-Q),
                    diagonalisation, absorption, exciton envelopes,
                    Wannier interpolation on dense grids
    potentials/     RytovaKeldysh, ExtendedRytovaKeldysh
    io/             Wannier90 (.win, .wout, _hr.dat, _u.mat,
                    _u_dis.mat) and QE `bands.x` readers; HDF5 output
    analysis/       2D Wannier-equation reference solver
    backends/       numpy / cupy backend switch
```

## What's new in 0.2

- Finite-Q BSE: compute Omega_S(Q) at arbitrary centre-of-mass momentum
  via Wannier interpolation of the electron bands and projectors.
- `bse.compute_exciton_dispersion` sweeps a list of Q-points and returns
  the lowest-N eigenvalues at each.
- Extended Rytova-Keldysh potential with a free-carrier screening
  wavevector q_s, for doped / lightly screened monolayers.
- Package renamed `tbexciton90` (previously `wanbse` in development).

## References

- L. V. Keldysh, JETP Lett. 29, 658 (1979).
- P. Cudazzo, I. V. Tokatly, A. Rubio, Phys. Rev. B 84, 085406 (2011).
- G. Berghauser, E. Malic, Phys. Rev. B 89, 125309 (2014).
- D. Y. Qiu, F. H. da Jornada, S. G. Louie,
  Phys. Rev. Lett. 115, 176801 (2015).
- Y. Cho, T. C. Berkelbach, Phys. Rev. B 97, 041409(R) (2018).

## License

MIT. See `LICENSE`.
