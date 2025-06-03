#!/usr/bin/env python3
"""
Example using GPU acceleration with exciton-wannier90.
"""

import numpy as np
import time
from exciton_wannier90 import Wannier90Parser, TightBindingModel, BSESolver
from exciton_wannier90.utils import ParallelManager

# Initialize parallel manager with GPU
parallel = ParallelManager(use_gpu=True)
print(f"Using GPU: {parallel.use_gpu}")

# Get memory info
mem_info = parallel.get_memory_info()
if 'gpu' in mem_info:
    print(f"GPU memory: {mem_info['gpu']['free_gb']:.1f}/{mem_info['gpu']['total_gb']:.1f} GB")

# Parse files
parser = Wannier90Parser()
parser.parse_hr_file('silicon_hr.dat')
parser.parse_kpt_file('silicon_band.kpt')

# Create TB model with GPU
tb_model = TightBindingModel(parser, use_gpu=True)

# Time the band calculation
t0 = time.time()
eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)
print(f"Band calculation time: {time.time() - t0:.2f} seconds")

# Solve BSE with GPU
bse_solver = BSESolver(tb_model, num_valence=4, num_conduction=4, use_gpu=True)

t0 = time.time()
exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
    parser.kpoints, eigenvalues, eigenvectors, num_states=50
)
print(f"BSE calculation time: {time.time() - t0:.2f} seconds")

print(f"\nLowest 5 exciton energies:")
for i in range(5):
    print(f"  State {i}: {exciton_energies[i]:.4f} eV")

# Cleanup GPU memory
parallel.cleanup()