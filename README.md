# TBExciton90

<p align="center">
  <strong>Tight-Binding Exciton Calculations from Wannier90</strong>
  <br>
  <em>Beautiful • Fast • GPU-Accelerated</em>
</p>

<p align="center">
  <a href="https://github.com/amal-kishore/TBExciton90/blob/main/INSTALL.md">📦 Installation</a> •
  <a href="https://github.com/amal-kishore/TBExciton90/blob/main/docs/TBExciton90_Manual.pdf">📖 Manual</a> •
  <a href="https://github.com/amal-kishore/TBExciton90/tree/main/examples">🚀 Examples</a> •
  <a href="https://github.com/amal-kishore/TBExciton90/issues">💬 Support</a>
</p>

---

## What is TBExciton90?

TBExciton90 computes **exciton properties** (bound electron-hole pairs) in materials using tight-binding models from **Wannier90**. It's designed for condensed matter physicists studying optical properties of semiconductors and 2D materials.

### Key Features

✨ **Easy to Use**: One command to go from Wannier90 files to beautiful plots  
⚡ **Fast**: GPU acceleration and MPI parallelization support  
🎨 **Beautiful**: Publication-ready plots with professional aesthetics  
🔬 **Scientific**: Proper treatment of electron-hole interactions  
📊 **Comprehensive**: Band structures, exciton spectra, absorption, and wavefunctions  

## Quick Start

### Installation
```bash
pip install tbexciton90
```

### Basic Usage
```bash
# Compute excitons from Wannier90 outputs
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt

# With GPU acceleration (if available)
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt --gpu
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

- 📈 **Electronic band structure** with highlighted band edges
- 🌟 **Exciton energy spectrum** distinguishing bright (optically active) vs dark states  
- 📊 **Optical absorption** comparing with/without electron-hole interactions
- 🌊 **Real-space wavefunctions** showing exciton size and shape
- 📋 **Summary data** in both human-readable and HDF5 formats

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
- **Anyone** working with Wannier90 who wants beautiful plots!

## Performance

- **Small systems** (< 100 k-points): Seconds on laptop
- **Medium systems** (< 1000 k-points): Minutes on workstation  
- **Large systems** (> 1000 k-points): Use GPU acceleration
- **Huge systems** (> 10,000 k-points): Use MPI parallelization

## Documentation

- 📖 **[Technical Manual](docs/TBExciton90_Manual.pdf)** - Complete physics and mathematics
- 📦 **[Installation Guide](INSTALL.md)** - Setup with GPU/MPI support
- 🚀 **[Examples](examples/)** - Tutorial notebooks and scripts
- 🎯 **[API Reference](docs/api/)** - Function documentation

## Citing TBExciton90

If you use TBExciton90 in your research, please cite:

```bibtex
@software{tbexciton90,
  title = {TBExciton90: Tight-binding exciton calculations from Wannier90},
  author = {TBExciton90 Development Team},
  year = {2024},
  url = {https://github.com/amal-kishore/TBExciton90},
  version = {0.1.0}
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/amal-kishore/TBExciton90/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/amal-kishore/TBExciton90/discussions)  
- 📧 **Email**: amal.kishore@example.com

## License

TBExciton90 is open source under the [MIT License](LICENSE).

---

<p align="center">
  <em>Made with ❤️ for the condensed matter physics community</em>
</p>