#!/usr/bin/env python3
"""
Basic example of using exciton-wannier90 package.
"""

import numpy as np
import matplotlib.pyplot as plt
from exciton_wannier90 import Wannier90Parser, TightBindingModel, BSESolver
from exciton_wannier90.visualization import ExcitonPlotter

# Parse Wannier90 outputs
parser = Wannier90Parser()
parser.parse_hr_file('silicon_hr.dat')
parser.parse_kpt_file('silicon_band.kpt')
parser.parse_centres_file('silicon_centres.xyz')

# Create tight-binding model
tb_model = TightBindingModel(parser)

# Compute band structure
eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)

# Analyze bands
num_valence = 4
vbm = eigenvalues[:, num_valence-1].max()
cbm = eigenvalues[:, num_valence].min()
bandgap = cbm - vbm

print(f"Electronic band gap: {bandgap:.3f} eV")

# Setup BSE solver
bse_solver = BSESolver(tb_model, num_valence=4, num_conduction=4)
bse_solver.set_screening('constant', {'W0': 0.1})

# Solve for excitons
exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
    parser.kpoints, eigenvalues, eigenvectors, num_states=10
)

# Calculate binding energy
exciton_binding = bandgap - exciton_energies[0]
print(f"Exciton binding energy: {exciton_binding:.3f} eV")
print(f"Optical gap: {exciton_energies[0]:.3f} eV")

# Plot results
plotter = ExcitonPlotter()
plotter.plot_band_structure(parser.kpoints, eigenvalues, num_valence)
plotter.plot_exciton_spectrum(exciton_energies)