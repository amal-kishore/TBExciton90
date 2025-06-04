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

### Understanding k-points and Q-points

**k-points (Crystal Momentum)**
- k-points represent momentum states of electrons in the crystal
- They come from the periodicity of the crystal lattice  
- Each k-point corresponds to a Bloch wave: ψ(r) = e^(ik·r) u(r)
- Your calculations sample the Brillouin zone with a k-point grid

**Q-points (Exciton Momentum)**
- Q-points represent the total momentum of the exciton (electron-hole pair)
- Q = k_electron - k_hole (difference in momenta)
- Most optical calculations focus on Q=0 (optically allowed excitons)

### BSE Mathematical Framework

**Step 1: Single-Particle States**
Solve the tight-binding Hamiltonian for each k-point:
```
H(k) |ψ_nk⟩ = E_nk |ψ_nk⟩
```
This gives electron energies E_nk and wavefunctions ψ_nk.

**Step 2: Electron-Hole Pairs**
Create electron-hole pairs by exciting electron from valence (v) to conduction (c):
```
|vck⟩ = c†_ck c_vk |0⟩
```

**Step 3: Exciton Wavefunction**
The exciton is a linear combination of electron-hole pairs:
```
|Ψ_S⟩ = Σ_vck A^S_vck |vck⟩
```
The A^S_vck coefficients are what we solve for.

**Step 4: BSE Matrix Equation**
```
H^BSE · A^S = Ω^S · A^S
```
Where:
```
H^BSE = (E_ck - E_vk) + Interaction_Kernel
```
- First term: Free electron-hole pair energy
- Second term: Coulomb attraction between electron and hole

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
- `convergence` - Test k-point convergence
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

**With MPI parallelization:**
```bash
# Run on 4 processes
mpirun -np 4 tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt --mpi

# Run on 8 processes with GPU
mpirun -np 8 tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt --mpi --gpu
```

**Generate plots from existing results:**
```bash
# Plot all types (band structure, exciton spectrum, optical absorption)
tbx90 plot --results-dir ./results --output-dir ./plots

# Plot specific types
tbx90 plot --plot-type bands
tbx90 plot --plot-type excitons --output-dir ./exciton_plots
tbx90 plot --plot-type absorption
```

**Test k-point convergence:**
```bash
# For 3D material (like silicon)
tbx90 convergence --input silicon_hr.dat --material-type 3D

# For 2D material
tbx90 convergence --input material_hr.dat --material-type 2D --num-valence 4

# With custom parameters
tbx90 convergence --input silicon_hr.dat --num-valence 4 --num-conduction 4 --screening 0.1
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

### Plotting Options

The `plot` command generates high-quality plots from existing calculation results:

**Plot Types:**
- `bands` - Electronic band structure
- `excitons` - Exciton energy spectrum 
- `absorption` - Optical absorption comparison
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
3. **Large systems** (> 1000 k-points): Use MPI parallelization
4. **Very large systems** (> 5000 k-points): Use both GPU + MPI

**MPI Usage:**
```bash
# For large calculations
mpirun -np 4 tbx90 compute --input system_hr.dat --kpoints system_band.kpt --mpi

# For very large calculations with GPU
mpirun -np 8 tbx90 compute --input system_hr.dat --kpoints system_band.kpt --mpi --gpu
```

Memory scaling: O(N_k × N_v × N_c)

---

## Convergence Testing

**Why convergence testing is essential:**

BSE calculations must be converged with respect to k-point sampling to ensure reliable results. Insufficient k-points can lead to:
- Incorrect exciton binding energies
- Missing exciton states
- Wrong oscillator strengths

### Automatic Convergence Testing

```bash
# Test convergence for silicon (3D material)
tbx90 convergence --input silicon_hr.dat --material-type 3D
```

**What this does:**
1. Tests k-grids: 2×2×2, 4×4×4, 6×6×6, 8×8×8, 10×10×10
2. Computes band gap, exciton energies, binding energies for each grid
3. Checks convergence criterion (< 1 meV difference)
4. Generates convergence plots
5. Extrapolates to infinite k-points
6. Recommends optimal k-grid

**Output files:**
- `kpoint_convergence.png` - Convergence plots
- `convergence_results.json` - Detailed numerical results
- `convergence_summary.txt` - Human-readable summary

### Convergence Criteria

**Converged calculation:**
- Band gap difference < 1 meV between successive grids
- Exciton energy difference < 1 meV
- Binding energy difference < 1 meV

**Recommended k-grids:**
- **3D materials**: Start with 6×6×6, check convergence
- **2D materials**: Start with 12×12×1, check convergence  
- **1D materials**: Start with 24×1×1, check convergence

### Manual Convergence Check

```python
from tbexciton90.utils.convergence import ConvergenceTest

# Initialize convergence tester
conv_test = ConvergenceTest('silicon_hr.dat')

# Define k-grids to test
k_grids = [(4,4,4), (6,6,6), (8,8,8), (10,10,10)]

# Run convergence test
results = conv_test.test_kpoint_convergence(
    k_grids=k_grids,
    num_valence=4,
    num_conduction=4,
    output_dir='./convergence'
)

# Check if converged
if results['exciton_convergence']:
    print("Calculation is converged!")
    print(f"Extrapolated exciton energy: {results['exciton_extrapolated']:.6f} eV")
```

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
  author = {amal kishore},
  year = {2025},
  url = {https://github.com/amal-kishore/TBExciton90}
}
```

---

## Support

- Issues: https://github.com/amal-kishore/TBExciton90/issues
- Discussions: https://github.com/amal-kishore/TBExciton90/discussions