#!/usr/bin/env python3
"""
Complete example demonstrating exciton analysis with bright/dark distinction
and absorption with/without electron-hole interaction.
"""

import numpy as np
import matplotlib.pyplot as plt
from exciton_wannier90 import Wannier90Parser, TightBindingModel, BSESolver
from exciton_wannier90.solvers import OpticalProperties
from exciton_wannier90.visualization import ExcitonPlotter

# Configuration
NUM_VALENCE = 4
NUM_CONDUCTION = 4
NUM_EXCITON_STATES = 100
SCREENING_PARAMETER = 0.1  # eV

print("="*60)
print("Exciton Analysis with Bright/Dark Distinction")
print("="*60)

# Step 1: Parse Wannier90 outputs
parser = Wannier90Parser()
parser.parse_hr_file('silicon_hr.dat')
parser.parse_kpt_file('silicon_band.kpt')
parser.parse_centres_file('silicon_centres.xyz')

print(f"\nParsed {parser.num_wann} Wannier functions")
print(f"Number of k-points: {parser.num_kpts}")

# Step 2: Compute band structure
tb_model = TightBindingModel(parser)
eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)

# Analyze band gap
vbm = eigenvalues[:, NUM_VALENCE-1].max()
cbm = eigenvalues[:, NUM_VALENCE].min()
bandgap = cbm - vbm

print(f"\nElectronic properties:")
print(f"  VBM: {vbm:.3f} eV")
print(f"  CBM: {cbm:.3f} eV")
print(f"  Band gap: {bandgap:.3f} eV")

# Step 3: Solve BSE for excitons
bse_solver = BSESolver(tb_model, num_valence=NUM_VALENCE, num_conduction=NUM_CONDUCTION)
bse_solver.set_screening('constant', {'W0': SCREENING_PARAMETER})

print(f"\nSolving BSE for {NUM_EXCITON_STATES} exciton states...")
exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
    parser.kpoints, eigenvalues, eigenvectors, num_states=NUM_EXCITON_STATES
)

# Step 4: Compute optical properties
optical = OpticalProperties(tb_model, bse_solver)

# Calculate oscillator strengths
oscillator_strengths = optical.compute_oscillator_strengths(
    parser.kpoints, eigenvectors, exciton_wavefunctions
)

# Identify bright excitons
bright_indices = optical.identify_bright_excitons(oscillator_strengths, threshold=0.01)

print(f"\nExciton analysis:")
print(f"  Total exciton states: {len(exciton_energies)}")
print(f"  Bright excitons: {len(bright_indices)}")
print(f"  Dark excitons: {len(exciton_energies) - len(bright_indices)}")
print(f"  Lowest exciton: {exciton_energies[0]:.3f} eV")
print(f"  Exciton binding: {bandgap - exciton_energies[0]:.3f} eV")

# List first few bright excitons
print(f"\nBright excitons:")
for i, idx in enumerate(bright_indices[:5]):
    print(f"  S{i+1}: E = {exciton_energies[idx]:.3f} eV, f = {oscillator_strengths[idx]:.3f}")

# Step 5: Compute absorption spectra
print("\nComputing optical absorption...")

# With electron-hole interaction
energy_range = (max(0, bandgap - 1.0), bandgap + 2.0)
energies_with, absorption_with = optical.compute_absorption_with_interaction(
    exciton_energies, oscillator_strengths, energy_range=energy_range
)

# Without electron-hole interaction
energies_without, absorption_without = optical.compute_absorption_without_interaction(
    eigenvalues, NUM_VALENCE, energy_range=energy_range
)

# Step 6: Transform bright exciton wavefunctions to real space
print("\nTransforming exciton wavefunctions to real space...")

# Create real-space grid for electron-hole separation
R_max = 20.0  # Angstroms
R_points = 100
R_grid = np.zeros((R_points, 3))
R_grid[:, 0] = np.linspace(-R_max, R_max, R_points)

# Transform first two bright excitons
bright_wavefunctions_R = []
for i in range(min(2, len(bright_indices))):
    idx = bright_indices[i]
    wf_R = bse_solver.transform_to_realspace(
        exciton_wavefunctions[:, idx], parser.kpoints, R_grid
    )
    bright_wavefunctions_R.append(wf_R)
    
    # Calculate exciton radius
    density = np.abs(wf_R)**2
    density /= np.sum(density)
    r_avg = np.sum(np.abs(R_grid[:, 0]) * density)
    print(f"  S{i+1} average radius: {r_avg:.2f} Å")

# Step 7: Create comprehensive plots
print("\nGenerating analysis plots...")

plotter = ExcitonPlotter(output_dir="exciton_analysis")

# 1. Exciton spectrum with bright/dark distinction
fig1 = plotter.plot_exciton_spectrum(
    exciton_energies[:50], 
    oscillator_strengths[:50],
    save_name="exciton_spectrum_annotated.png"
)

# 2. Absorption comparison
fig2 = plotter.plot_optical_absorption_comparison(
    energies_with, absorption_with,
    energies_without, absorption_without,
    exciton_energies, oscillator_strengths,
    save_name="absorption_comparison.png"
)

# 3. Bright exciton wavefunctions
for i, wf_R in enumerate(bright_wavefunctions_R):
    fig = plt.figure(figsize=(10, 6))
    
    # Plot real and imaginary parts
    ax1 = fig.add_subplot(211)
    ax1.plot(R_grid[:, 0], np.real(wf_R), 'b-', label='Real part')
    ax1.plot(R_grid[:, 0], np.imag(wf_R), 'r--', label='Imaginary part')
    ax1.set_ylabel('Ψ(R)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Bright Exciton S{i+1} Wavefunction (E = {exciton_energies[bright_indices[i]]:.3f} eV)')
    
    # Plot probability density
    ax2 = fig.add_subplot(212)
    density = np.abs(wf_R)**2
    ax2.plot(R_grid[:, 0], density, 'k-', linewidth=2)
    ax2.fill_between(R_grid[:, 0], density, alpha=0.3)
    ax2.set_xlabel('Electron-hole separation R (Å)')
    ax2.set_ylabel('|Ψ(R)|²')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'exciton_analysis/bright_exciton_S{i+1}_detailed.png', dpi=300)
    plt.close()

print("\nAnalysis complete! Check the 'exciton_analysis' directory for plots.")
print("="*60)