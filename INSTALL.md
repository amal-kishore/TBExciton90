# Installation Guide for TBExciton90

## Quick Installation

### Prerequisites

- Python 3.8 or higher
- Git (for development installation)

### Basic Installation

```bash
pip install tbexciton90
```

## Development Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/TBExciton90.git
cd TBExciton90
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv tbx90_env

# Activate (Linux/Mac)
source tbx90_env/bin/activate

# Activate (Windows)
tbx90_env\Scripts\activate
```

### 3. Install Package

```bash
# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[gpu,mpi,dev]"
```

## GPU Support (Optional)

### CUDA 11.x

```bash
pip install cupy-cuda11x
```

### CUDA 12.x

```bash
pip install cupy-cuda12x
```

### Verification

```python
import cupy as cp
print(f"GPU available: {cp.cuda.is_available()}")
```

## MPI Support (Optional)

### Install MPI

**Ubuntu/Debian:**
```bash
sudo apt-get install mpich libmpich-dev
```

**CentOS/RHEL:**
```bash
sudo yum install mpich mpich-devel
```

**macOS:**
```bash
brew install mpich
```

### Install mpi4py

```bash
pip install mpi4py
```

### Verification

```bash
mpirun -np 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.Get_rank()}')"
```

## Testing Installation

### Basic Test

```bash
tbx90 test
```

### Run Example

```bash
# Use included silicon example
cd examples/silicon

# Run calculation (computes bands, excitons, optical properties, generates plots)
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt

# Or test with Python
python ../basic_example.py
```

### Python Test

```python
import tbexciton90 as tbx
print(f"TBExciton90 version: {tbx.__version__}")

# Test imports
from tbexciton90 import (
    Wannier90Parser, TightBindingModel, BSESolver,
    OpticalProperties, AdvancedExcitonPlotter
)
print("All modules imported successfully!")
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError:**
```bash
# Make sure you're in the right environment
which python
pip list | grep tbexciton90
```

**CUDA Issues:**
```bash
# Check CUDA version
nvcc --version

# Install correct CuPy version
pip uninstall cupy-cuda11x cupy-cuda12x
pip install cupy-cuda11x  # for CUDA 11.x
```

**MPI Issues:**
```bash
# Check MPI installation
which mpirun
mpirun --version

# Reinstall mpi4py
pip uninstall mpi4py
pip install mpi4py --no-cache-dir
```

### Performance Tips

1. **Use virtual environments** to avoid conflicts
2. **Install GPU support** for large systems (>1000 k-points)
3. **Use MPI** for very large calculations (>10000 exciton states)
4. **Check memory usage** with `htop` or `nvidia-smi`

## Command Line Usage

After installation, you have access to:

- `tbx90` - Full command
- `tbexciton90` - Alternative command

### Examples

```bash
# Basic calculation (does everything: bands, excitons, optics, plots)
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt

# Generate config file
tbx90 generate-config --example gpu --output my_config.yaml

# Run with config
tbx90 compute --config my_config.yaml

# With MPI parallelization
mpirun -np 4 tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt --mpi

# Generate plots from existing results
tbx90 plot --results-dir ./results --plot-type all
tbx90 plot --plot-type bands

# Test installation
tbx90 test

# Get help
tbx90 --help
tbx90 compute --help
tbx90 plot --help
```

## Next Steps

1. Read the [README](README.md) for detailed usage
2. Check [examples/](examples/) for tutorials
3. Explore the [API documentation](docs/)
4. Join our community discussions

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/TBExciton90/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/TBExciton90/discussions)
- **Documentation**: [Read the Docs](https://tbexciton90.readthedocs.io/)

---

*Installation problems? Please open an issue with your system details!*