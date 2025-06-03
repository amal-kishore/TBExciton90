# TBExciton90 Documentation

This directory contains the comprehensive technical documentation for TBExciton90.

## Contents

- **`TBExciton90_Manual.tex`** - Complete LaTeX source for the technical manual
- **`TBExciton90_Manual.pdf`** - Compiled PDF manual (generated)
- **`Makefile`** - Build system for the documentation
- **`build_manual.sh`** - Automated build script

## Building the Manual

### Prerequisites

You need a LaTeX distribution installed:

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**CentOS/RHEL:**
```bash
sudo yum install texlive-scheme-full
```

**macOS:**
```bash
brew install --cask mactex
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

### Build Options

#### Option 1: Automated Build Script
```bash
./build_manual.sh
```

#### Option 2: Using Make
```bash
# Full build (recommended)
make all

# Quick build (single pass)
make quick

# View PDF
make view

# Clean auxiliary files
make clean
```

#### Option 3: Manual Compilation
```bash
pdflatex TBExciton90_Manual.tex
pdflatex TBExciton90_Manual.tex  # Second pass for references
```

## Manual Contents

The technical manual includes:

### 1. Introduction
- What are excitons?
- Why TBExciton90?
- Target audience and applications

### 2. Theoretical Background
- **Tight-binding models** from Wannier90
- **Electronic band structure** calculations
- **Bethe-Salpeter equation** formulation
- **Optical properties** and absorption spectra
- **Real-space wavefunctions** and exciton analysis

### 3. Implementation Details
- Software architecture
- Key algorithms and optimizations
- File format specifications
- Performance considerations

### 4. Usage Examples
- Command-line interface
- Python API examples
- Configuration files
- Advanced customization

### 5. Performance and Optimization
- Computational complexity
- Memory requirements
- GPU acceleration
- MPI parallelization

### 6. Validation and Benchmarks
- Test systems and validation
- Performance benchmarks
- Comparison with other codes

### 7. Troubleshooting
- Common issues and solutions
- Debugging techniques
- Performance optimization

### 8. Advanced Topics
- Custom screening models
- Post-processing tools
- Integration with other software

### 9. API Reference
- Complete function documentation
- Class hierarchies
- Usage examples

### 10. Appendices
- File format specifications
- Physical constants
- Unit conversions
- References

## Mathematics Coverage

The manual provides complete mathematical formulations for:

- **Tight-binding Hamiltonian construction**: $H_{nm}(\mathbf{k}) = \sum_{\mathbf{R}} H_{nm}(\mathbf{R}) e^{i\mathbf{k} \cdot \mathbf{R}}$

- **Bethe-Salpeter equation**: $\sum_{v'c'\mathbf{k}'} H^{\text{BSE}}_{vc\mathbf{k},v'c'\mathbf{k}'} A^S_{v'c'\mathbf{k}'} = \Omega^S A^S_{vc\mathbf{k}}$

- **Optical absorption**: $\alpha(\omega) = \sum_S f^S \frac{\Gamma/\pi}{(\omega - \Omega^S)^2 + \Gamma^2}$

- **Real-space transformations**: $\Psi^S(\mathbf{R}) = \sum_{\mathbf{k}} e^{i\mathbf{k} \cdot \mathbf{R}} \sum_{vc} A^S_{vc\mathbf{k}}$

## Target Audience

The technical manual is designed for:

- **Graduate students** learning many-body theory
- **Researchers** implementing exciton calculations
- **Developers** contributing to TBExciton90
- **Advanced users** requiring detailed understanding

## Additional Resources

- **Quick Start**: See main [README.md](../README.md) for basic usage
- **Installation**: See [INSTALL.md](../INSTALL.md) for setup instructions
- **Examples**: See [examples/](../examples/) for tutorial scripts
- **API Docs**: Auto-generated from source code (coming soon)

## Contributing to Documentation

To improve the documentation:

1. Edit the LaTeX source (`TBExciton90_Manual.tex`)
2. Test compilation: `./build_manual.sh`
3. Check the generated PDF
4. Submit a pull request

### LaTeX Style Guidelines

- Use `\code{}` for inline code
- Use `lstlisting` environment for code blocks
- Include equations with proper numbering
- Add figures and tables where helpful
- Keep sections well-organized and cross-referenced

---

**Note**: The PDF manual is the authoritative technical reference for TBExciton90. The main README.md provides a user-friendly overview.