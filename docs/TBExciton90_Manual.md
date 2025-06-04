# TBExciton90 Technical Manual

## Excitonic Properties from Wannier-based Tight-binding Models

Version 0.1.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [Examples](#examples)
6. [API Reference](#api-reference)

---

## Introduction

TBExciton90 is a Python package for computing exciton properties in crystalline materials using tight-binding models from Wannier90 calculations.

### What are Excitons?

Excitons are bound states of electrons and holes that determine optical properties of semiconductors:

- **Electron**: Negative charge in conduction band
- **Hole**: Positive charge (missing electron) in valence band  
- **Exciton**: Bound electron-hole pair
- **Binding Energy**: Energy to separate electron and hole
- **Optical Gap**: Energy of lowest bright exciton

### Key Features

- Direct Wannier90 integration
- Complete exciton framework
- GPU acceleration support
- MPI parallelization
- Publication-ready plots

---

## Theoretical Background

### Tight-Binding Hamiltonian

The Hamiltonian in k-space from Wannier90:

```
H(k) = Σ_R H(R) exp(ik·R)
```

Where:
- H(R) = hopping integrals from `*_hr.dat`
- k = crystal momentum
- R = lattice vectors

### Bethe-Salpeter Equation (BSE)

The exciton equation:

```
H_BSE |Ψ> = Ω |Ψ>
```

Where:
- H_BSE = exciton Hamiltonian
- |Ψ> = exciton wavefunction
- Ω = exciton energy

The BSE Hamiltonian:

```
H_BSE = (E_c - E_v) - K
```

- First term: electron-hole pair energy
- K: electron-hole interaction

### Optical Properties

Oscillator strength (brightness):

```
f = |<0|P|Ψ>|²
```

- Bright excitons: f > 0 (optically active)
- Dark excitons: f = 0 (optically forbidden)

---

## Installation

### Basic Install

```bash
pip install tbexciton90
```

### Development Install

```bash
git clone https://github.com/amal-kishore/TBExciton90
cd TBExciton90
pip install -e .
```

### GPU Support

```bash
pip install cupy-cuda11x  # CUDA 11.x
pip install cupy-cuda12x  # CUDA 12.x
```

### MPI Support

```bash
# Install MPI first (system-dependent)
pip install mpi4py
```

---

## Usage Guide

### Command Line

**Main calculation command** (computes band structure, solves BSE for excitons, calculates optical properties, and generates plots):

```bash
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt
```

**Available commands:**
- `compute` - Run full exciton calculation
- `plot` - Generate plots from existing results
- `generate-config` - Create configuration files
- `test` - Test installation

**Generate configuration file:**
```bash
tbx90 generate-config --example gpu --output my_config.yaml
```

**Run with configuration:**
```bash
tbx90 compute --config my_config.yaml
```

**With command-line options:**
```bash
tbx90 compute \
  --input silicon_hr.dat \
  --kpoints silicon_band.kpt \
  --num-valence 4 \
  --num-conduction 4 \
  --screening 0.5 \
  --gpu
```

**Generate plots from existing results:**
```bash
# Plot all types
tbx90 plot --results-dir ./results --output-dir ./plots

# Plot specific types
tbx90 plot --plot-type bands
tbx90 plot --plot-type excitons --output-dir ./exciton_plots
tbx90 plot --plot-type absorption
tbx90 plot --plot-type wavefunctions --results-dir ./my_results
```

**Test installation:**
```bash
tbx90 test
```

### Python API

```python
import tbexciton90 as tbx

# Quick calculation
results = tbx.compute_excitons(
    hr_file='silicon_hr.dat',
    kpt_file='silicon_band.kpt',
    num_valence=4,
    num_conduction=4
)

print(f"Optical gap: {results['exciton_energies'][0]:.3f} eV")
print(f"Binding energy: {results['binding_energy']:.3f} eV")
```

### Detailed Analysis

```python
from tbexciton90 import Wannier90Parser, TightBindingModel, BSESolver
from tbexciton90.visualization import AdvancedExcitonPlotter

# Parse Wannier90 files
parser = Wannier90Parser()
parser.parse_hr_file('silicon_hr.dat')
parser.parse_kpt_file('silicon_band.kpt')

# Compute bands
tb_model = TightBindingModel(parser)
eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)

# Solve BSE
bse_solver = BSESolver(
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    kpoints=parser.kpoints
)
exciton_energies, exciton_states = bse_solver.solve()

# Generate plots
plotter = AdvancedExcitonPlotter()
plotter.plot_band_structure(parser.kpoints, eigenvalues, num_valence=4)
plotter.plot_exciton_spectrum_enhanced(exciton_energies, oscillator_strengths)
```

---

## Examples

### Silicon Example

Located in `examples/silicon/`:

```bash
cd examples/silicon
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt
```

Expected results:
- Band gap: ~1.1 eV (indirect gap)
- Exciton binding: ~15 meV
- Lowest bright exciton: ~1.085 eV

### Basic Python Example

See `examples/basic_example.py` for minimal usage.

### Advanced Example

See `examples/final_demo.py` for comprehensive analysis.

---

## API Reference

### Core Classes

#### Wannier90Parser
- `parse_hr_file(filename)`: Read Hamiltonian
- `parse_kpt_file(filename)`: Read k-points
- `parse_centres_file(filename)`: Read Wannier centers

#### TightBindingModel
- `compute_bands(kpoints)`: Calculate band structure
- `get_band_gap()`: Find electronic gap

#### BSESolver
- `solve()`: Solve BSE for excitons
- `compute_oscillator_strength()`: Calculate optical activity
- `transform_to_realspace()`: Get real-space wavefunction

#### AdvancedExcitonPlotter
- `plot_band_structure()`: Electronic bands
- `plot_exciton_spectrum_enhanced()`: Exciton energies
- `plot_absorption_comparison_enhanced()`: Optical absorption
- `plot_exciton_wavefunction_enhanced()`: Real-space visualization

### Plotting Options

The `plot` command generates high-quality plots from existing calculation results:

**Plot Types:**
- `bands` - Electronic band structure
- `excitons` - Exciton energy spectrum 
- `absorption` - Optical absorption comparison
- `wavefunctions` - Real-space exciton wavefunctions
- `summary` - Overview plot with all key results
- `all` - Generate all available plots (default)

**Examples:**
```bash
# All plots (high quality by default)
tbx90 plot

# Only band structure
tbx90 plot --plot-type bands

# Custom input/output directories
tbx90 plot --results-dir ./my_calculation --output-dir ./my_plots
```

### Configuration

Create `config.yaml`:

```yaml
computation:
  num_valence: 4
  num_conduction: 4
  screening_length: 10.0
  interaction_strength: 0.5

parallel:
  use_gpu: true
  use_mpi: false

output:
  save_plots: true
  save_data: true
  format: hdf5
```

---

## Performance Tips

1. **Small systems** (< 100 k-points): CPU is fine
2. **Medium systems** (< 1000 k-points): Use GPU
3. **Large systems** (> 1000 k-points): GPU + MPI

Memory scaling: O(N_k × N_v × N_c)

---

## Troubleshooting

### Common Issues

**Import Error**:
```bash
pip install -e .  # Reinstall in development mode
```

**GPU Not Found**:
```python
import cupy
print(cupy.cuda.is_available())  # Should be True
```

**MPI Issues**:
```bash
mpirun -np 4 python script.py  # Use mpirun
```

---

## Citation

If you use TBExciton90 in your research:

```bibtex
@software{tbexciton90,
  title = {TBExciton90: Excitonic properties from Wannier-based tight-binding models},
  author = {TBExciton90 Development Team},
  year = {2024},
  url = {https://github.com/amal-kishore/TBExciton90}
}
```

---

## Support

- Issues: https://github.com/amal-kishore/TBExciton90/issues
- Discussions: https://github.com/amal-kishore/TBExciton90/discussions