# TBExciton90

<p align="center">
  <strong>Excitonic Properties from Wannier-based Tight-binding Models</strong>
  <br>
  <em>Fast • GPU-Accelerated • Comprehensive</em>
</p>

<p align="center">
  <a href="https://github.com/amal-kishore/TBExciton90/blob/main/INSTALL.md">Installation</a> •
  <a href="https://github.com/amal-kishore/TBExciton90/blob/main/docs/TBExciton90_Manual.md">Manual</a> •
  <a href="https://github.com/amal-kishore/TBExciton90/tree/main/examples">Examples</a> •
  <a href="https://github.com/amal-kishore/TBExciton90/issues">Support</a>
</p>

---

## What is TBExciton90?

TBExciton90 computes **exciton properties** (bound electron-hole pairs) in materials using tight-binding models from **Wannier90**. It's designed for condensed matter physicists studying optical properties of semiconductors and 2D materials.

### Key Features

- **Easy to Use**: One command to go from Wannier90 files to excitonic properties  
- **Fast**: GPU acceleration and MPI parallelization support  
- **Scientific**: Proper treatment of electron-hole interactions  
- **Comprehensive**: Band structures, exciton spectra, and optical absorption  

## Quick Start

### Installation
```bash
pip install tbexciton90
```

### Basic Usage
```bash
# Main command: computes bands, solves BSE, calculates optical properties, generates plots
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt

# Generate configuration file
tbx90 generate-config --example minimal --output config.yaml

# Run with configuration
tbx90 compute --config config.yaml

# With GPU acceleration
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt --gpu

# With MPI parallelization (4 processes)
mpirun -np 4 tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt --mpi

# Generate plots from existing results
tbx90 plot --results-dir ./results --plot-type all
tbx90 plot --plot-type excitons
```

### Python API
```python
import tbexciton90 as tbx

# Quick analysis
results = tbx.compute_excitons(
    hr_file='material_hr.dat',
    kpt_file='material_band.kpt'
)

print(f"Optical gap: {results['exciton_energies'][0]:.3f} eV")
print(f"Binding energy: {results['binding_energy']:.3f} eV")
```

## What You Get

After running TBExciton90, you'll have:

- **Electronic band structure**
- **Exciton energy spectrum** distinguishing bright (optically active) vs dark states  
- **Optical absorption** comparing with/without electron-hole interactions

<p align="center">
  <img src="docs/example_plots.png" alt="Example TBExciton90 outputs" width="800">
</p>

## Input Requirements

You need **Wannier90** output files:
- `*_hr.dat` - Tight-binding Hamiltonian
- `*_band.kpt` - k-points for band structure  
- `*_centres.xyz` - Wannier function centers (optional)

These are standard outputs from any Wannier90 calculation.

## Who Should Use This?

- **Experimentalists** wanting to understand optical spectra
- **Theorists** studying excitons in 2D materials
- **Students** learning about many-body effects in solids
- **Anyone** working with Wannier90 who needs excitonic properties

## Performance

- **Small systems** (< 100 k-points): Seconds on laptop
- **Medium systems** (< 1000 k-points): Minutes on workstation  
- **Large systems** (> 1000 k-points): Use GPU acceleration
- **Huge systems** (> 10,000 k-points): Use MPI parallelization

## Documentation

- 📖 **[Technical Manual](docs/TBExciton90_Manual.md)** - Complete physics and mathematics
- 📦 **[Installation Guide](INSTALL.md)** - Setup with GPU/MPI support
- 🚀 **[Examples](examples/)** - Tutorial notebooks and scripts
- 🎯 **[API Reference](docs/api/)** - Function documentation

## Citing TBExciton90

If you use TBExciton90 in your research, please cite:

```bibtex
@software{tbexciton90,
  title = {TBExciton90: Excitonic properties from Wannier-based tight-binding models},
  author = {amal kishore},
  year = {2025},
  url = {https://github.com/amal-kishore/TBExciton90},
  version = {0.1.0}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Bug reports**: [GitHub Issues](https://github.com/amal-kishore/TBExciton90/issues)
- **Questions**: [GitHub Discussions](https://github.com/amal-kishore/TBExciton90/discussions)  
- **Email**: amalk4905@gmail.com

## License

TBExciton90 is open source under the [MIT License](LICENSE).

