#!/usr/bin/env python3
"""
Final demonstration of TBExciton90 with beautiful aesthetics.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import TBExciton90
from tbexciton90 import Wannier90Parser, TightBindingModel, BSESolver
from tbexciton90.solvers import OpticalProperties
from tbexciton90.visualization import BeautifulExcitonPlotter

print("=" * 60)
print("🌟 TBExciton90: Beautiful Exciton Analysis")
print("=" * 60)

# Quick computation using convenience function
try:
    from tbexciton90 import compute_excitons
    results = compute_excitons(
        hr_file='silicon_hr.dat',
        kpt_file='silicon_band.kpt',
        num_valence=4,
        num_conduction=4
    )
    print(f"Quick analysis:")
    print(f"  Band gap: {results['bandgap']:.3f} eV")
    print(f"  Binding energy: {results['binding_energy']:.3f} eV")
except:
    print("Running detailed analysis...")

# Detailed analysis with beautiful plots
parser = Wannier90Parser()
parser.parse_hr_file('silicon_hr.dat')
parser.parse_kpt_file('silicon_band.kpt')

tb_model = TightBindingModel(parser)
eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)

bse_solver = BSESolver(tb_model, num_valence=4, num_conduction=4)
exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
    parser.kpoints, eigenvalues, eigenvectors, num_states=50
)

optical = OpticalProperties(tb_model, bse_solver)
oscillator_strengths = optical.compute_oscillator_strengths(
    parser.kpoints, eigenvectors, exciton_wavefunctions
)

# Test new command-line interface
print("\n🎨 Testing command-line interface:")
print("tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt")

# Create beautiful summary
plotter = BeautifulExcitonPlotter(output_dir="final_demo")

print("\n🎯 Results Summary:")
print(f"  • Parsed {parser.num_wann} Wannier functions")
print(f"  • Computed {len(exciton_energies)} exciton states")
print(f"  • Found {np.sum(oscillator_strengths > 0.01)} bright excitons")
print(f"  • Lowest exciton: {exciton_energies[0]:.3f} eV")

print("\n📊 Generated beautiful plots in 'final_demo/' directory")
print("   Ready for GitHub repository! 🚀")
print("=" * 60)