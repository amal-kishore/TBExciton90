# BSE Solver for Silicon

This directory contains a standalone Bethe-Salpeter Equation (BSE) solver specifically designed for silicon using Wannier90 output files.

## Features

- **Scissor shift correction**: Adjustable band gap correction (default: +0.23 eV)
- **Advanced screening models**: 
  - Constant screening (ε∞ = 11.7)
  - Penn model with q-dependent screening
- **Full 8-band basis**: Uses all sp³ Wannier functions
- **Command-line interface**: Easy parameter control

## Usage

### Basic run with constant screening:
```bash
python bse_solver_silicon.py --scissor 0.23 --screening constant --eps 11.7 --kmesh 18
```

### Penn model screening:
```bash
python bse_solver_silicon.py --scissor 0.23 --screening penn --eps 11.7 --kmesh 18
```

### Quick test with small k-mesh:
```bash
python bse_solver_silicon.py --kmesh 4
```

### Command-line options:
- `--scissor`: Scissor shift for conduction bands in eV (default: 0.23)
- `--screening`: Screening model - 'constant' or 'penn' (default: constant)
- `--eps`: Static dielectric constant ε∞ (default: 11.7)
- `--kmesh`: Monkhorst-Pack k-points per axis (default: 18)

## Results

The solver generates:
- `exciton_energies_final.dat`: Exciton energies with metadata
- `exciton_wavefunctions_final.npy`: Exciton eigenvectors
- `absorption_spectrum_final.png`: Optical absorption spectrum
- `bse_vs_independent_final.png`: BSE vs independent particle comparison

## Physical Model

- **Band structure**: 8-band sp³ Wannier functions from silicon_hr.dat
- **Gap correction**: Scissor shift brings TB gap from ~0.9 eV to experimental 1.12 eV
- **Coulomb interaction**: W(r) = e²/(ε(r)·r) with screening models
- **BSE matrix**: H_BSE = H_diag + K_direct (exchange neglected)

## Current Limitations

- Binding energies still too large (~400 meV with Penn model vs ~15 meV experimental)
- One unphysical negative-energy exciton (artifact of simplified Coulomb kernel)
- Need momentum-dependent screening ε(q,ω) for quantitative accuracy
- Exchange contributions not included

## Band Structure Visualization

To plot the silicon band structure:
```bash
python plot_bands_correct.py
```

This generates:
- `silicon_bands_corrected.png`: Band structure along L-Γ-X-K-Γ path
- Shows the small TB gap (~0.05 eV) before scissor correction

## Physics Notes

The large discrepancy between the TB gap (0.895 eV) and minimum optical transition (2.97 eV) 
occurs because the lowest transitions are between bands 4→5 at specific k-points, not at the 
actual VBM/CBM locations. This is a feature of the silicon band structure with the 8-band 
sp³ basis.

## References

- Wannier90: https://wannier.org/
- pythTB: https://pythtb.readthedocs.io/
- Penn model: Penn, D. R., Phys. Rev. 128, 2093 (1962)