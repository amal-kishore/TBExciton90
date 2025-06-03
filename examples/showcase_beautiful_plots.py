#!/usr/bin/env python3
"""
Showcase beautiful plots from TBExciton90.
"""

import numpy as np
import sys
sys.path.append('..')

from tbexciton90 import Wannier90Parser, TightBindingModel, BSESolver
from tbexciton90.solvers import OpticalProperties
from tbexciton90.visualization import BeautifulExcitonPlotter

print("="*60)
print("TBExciton90: Beautiful Exciton Analysis")
print("="*60)

# Parse Wannier90 outputs
parser = Wannier90Parser()
parser.parse_hr_file('silicon_hr.dat')
parser.parse_kpt_file('silicon_band.kpt')
parser.parse_centres_file('silicon_centres.xyz')

# Compute band structure
tb_model = TightBindingModel(parser)
eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)

# Band analysis
num_valence = 4
vbm = eigenvalues[:, num_valence-1].max()
cbm = eigenvalues[:, num_valence].min()
bandgap = cbm - vbm

print(f"\nElectronic Properties:")
print(f"  Band gap: {bandgap:.3f} eV")

# Solve BSE
bse_solver = BSESolver(tb_model, num_valence=4, num_conduction=4)
bse_solver.set_screening('constant', {'W0': 0.1})

print("\nSolving BSE...")
exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
    parser.kpoints, eigenvalues, eigenvectors, num_states=100
)

# Compute optical properties
optical = OpticalProperties(tb_model, bse_solver)
oscillator_strengths = optical.compute_oscillator_strengths(
    parser.kpoints, eigenvectors, exciton_wavefunctions
)

bright_indices = optical.identify_bright_excitons(oscillator_strengths)
print(f"\nFound {len(bright_indices)} bright excitons")

# Compute absorption
energy_range = (max(0, bandgap - 1.0), bandgap + 2.0)
energies_with, absorption_with = optical.compute_absorption_with_interaction(
    exciton_energies, oscillator_strengths, energy_range=energy_range
)
energies_without, absorption_without = optical.compute_absorption_without_interaction(
    eigenvalues, num_valence, energy_range=energy_range
)

# Transform bright exciton wavefunctions
R_grid = np.zeros((200, 3))
R_grid[:, 0] = np.linspace(-20, 20, 200)

bright_wavefunctions_R = []
for i in range(min(2, len(bright_indices))):
    idx = bright_indices[i]
    wf_R = bse_solver.transform_to_realspace(
        exciton_wavefunctions[:, idx], parser.kpoints, R_grid
    )
    bright_wavefunctions_R.append(wf_R)

# Create beautiful plots
print("\nGenerating beautiful plots...")
plotter = BeautifulExcitonPlotter(output_dir="beautiful_results")

# 1. Band structure
fig1 = plotter.plot_band_structure(
    parser.kpoints, eigenvalues, num_valence,
    save_name="beautiful_bands.png"
)

# 2. Exciton spectrum
fig2 = plotter.plot_exciton_spectrum_beautiful(
    exciton_energies[:50], oscillator_strengths[:50],
    save_name="beautiful_exciton_spectrum.png"
)

# 3. Absorption comparison
fig3 = plotter.plot_absorption_comparison_beautiful(
    energies_with, absorption_with,
    energies_without, absorption_without,
    exciton_energies, oscillator_strengths,
    save_name="beautiful_absorption.png"
)

# 4. Exciton wavefunctions
for i, wf_R in enumerate(bright_wavefunctions_R):
    energy = exciton_energies[bright_indices[i]]
    fig = plotter.plot_exciton_wavefunction_beautiful(
        R_grid, wf_R, state_index=i+1, energy=energy,
        save_name=f"beautiful_exciton_S{i+1}.png"
    )

print("\nBeautiful plots saved to 'beautiful_results/' directory!")
print("="*60)