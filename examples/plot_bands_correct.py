#!/usr/bin/env python3
"""
Corrected band structure plot following the example code structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from pythtb import w90

def plot_bands_correct_path():
    """
    Plot band structure using the correct k-path from the example.
    """
    print("=== Silicon Band Structure (Corrected Path) ===\n")
    
    # Load Wannier90 model (note: they use "example_a" folder)
    # We'll use our examples/silicon folder
    silicon = w90("examples/silicon", "silicon")
    
    # Get tight-binding model without hopping terms above 0.01 eV
    my_model = silicon.model(min_hopping_norm=0.01)
    
    # CORRECTED PATH - this is different from what I used!
    # Their path: L -> Gamma -> X -> K -> Gamma
    path = [[0.5, 0.5, 0.5],      # L
            [0.0, 0.0, 0.0],      # Gamma
            [0.5, -0.5, 0.0],     # X (different from mine!)
            [0.375, -0.375, 0.0], # K (different from mine!)
            [0.0, 0.0, 0.0]]      # Gamma
    
    # Labels of the nodes
    k_label = (r'$L$', r'$\Gamma$', r'$X$', r'$K$', r'$\Gamma$')
    
    # Construct the actual path
    (k_vec, k_dist, k_node) = my_model.k_path(path, 301)  # More points for smoother plot
    
    # Solve for all eigenvalues
    evals = my_model.solve_all(k_vec)
    
    # Analyze the gap properly
    print(f"Number of bands: {evals.shape[0]}")
    print(f"Number of k-points: {evals.shape[1]}")
    
    # Find band gap - looking at ALL bands, not splitting arbitrarily
    # The gap should be between the highest occupied and lowest unoccupied
    # For 8-band silicon model, typically bands 1-4 are valence-like
    
    # Method 1: Find the gap by looking for the largest energy gap
    all_energies = np.sort(evals.flatten())
    energy_gaps = np.diff(all_energies)
    max_gap_idx = np.argmax(energy_gaps)
    gap_lower = all_energies[max_gap_idx]
    gap_upper = all_energies[max_gap_idx + 1]
    estimated_gap = gap_upper - gap_lower
    
    print(f"\n=== Gap Analysis (Method 1: Largest gap in spectrum) ===")
    print(f"Gap lower bound: {gap_lower:.3f} eV")
    print(f"Gap upper bound: {gap_upper:.3f} eV")
    print(f"Estimated gap: {estimated_gap:.3f} eV")
    
    # Method 2: Assume bands 1-4 are valence, 5-8 are conduction
    if evals.shape[0] == 8:
        VBM = np.max(evals[:4, :])  # Max of first 4 bands
        CBM = np.min(evals[4:, :])  # Min of last 4 bands
        gap_direct = CBM - VBM
        
        print(f"\n=== Gap Analysis (Method 2: Fixed 4+4 split) ===")
        print(f"VBM (max of bands 1-4): {VBM:.3f} eV")
        print(f"CBM (min of bands 5-8): {CBM:.3f} eV")
        print(f"Band gap: {gap_direct:.3f} eV")
        
        # Find where VBM and CBM occur
        vbm_loc = np.unravel_index(np.argmax(evals[:4, :]), evals[:4, :].shape)
        cbm_loc = np.unravel_index(np.argmin(evals[4:, :]), evals[4:, :].shape)
        
        print(f"VBM at band {vbm_loc[0]+1}, k-point {vbm_loc[1]}")
        print(f"CBM at band {cbm_loc[0]+5}, k-point {cbm_loc[1]}")
        
        if vbm_loc[1] == cbm_loc[1]:
            print("Gap type: Direct")
        else:
            print("Gap type: Indirect")
    
    # Create figure following the example style
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all bands in black
    for i in range(evals.shape[0]):
        ax.plot(k_dist, evals[i], "k-", linewidth=2)
    
    # Add vertical lines at high-symmetry points
    for n in range(len(k_node)):
        ax.axvline(x=k_node[n], linewidth=0.5, color='k')
    
    # If we found the gap, shade it
    if evals.shape[0] == 8:
        ax.axhline(y=VBM, color='red', linestyle='--', alpha=0.5, label=f'VBM = {VBM:.3f} eV')
        ax.axhline(y=CBM, color='blue', linestyle='--', alpha=0.5, label=f'CBM = {CBM:.3f} eV')
        ax.fill_between(k_dist, VBM, CBM, alpha=0.1, color='yellow', label=f'Gap = {gap_direct:.3f} eV')
    
    ax.set_xlabel("Path in k-space", fontsize=14)
    ax.set_ylabel("Band energy (eV)", fontsize=14)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(k_label)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig("silicon_bands_corrected.pdf")
    fig.savefig("silicon_bands_corrected.png", dpi=300)
    
    # Also create a zoomed version near the gap
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    for i in range(evals.shape[0]):
        ax2.plot(k_dist, evals[i], "k-", linewidth=2)
    
    for n in range(len(k_node)):
        ax2.axvline(x=k_node[n], linewidth=0.5, color='k')
    
    if evals.shape[0] == 8:
        ax2.axhline(y=VBM, color='red', linestyle='--', alpha=0.5, label=f'VBM = {VBM:.3f} eV')
        ax2.axhline(y=CBM, color='blue', linestyle='--', alpha=0.5, label=f'CBM = {CBM:.3f} eV')
        ax2.fill_between(k_dist, VBM, CBM, alpha=0.2, color='yellow')
        ax2.set_ylim(VBM - 1, CBM + 1)
    
    ax2.set_xlabel("Path in k-space", fontsize=14)
    ax2.set_ylabel("Band energy (eV)", fontsize=14)
    ax2.set_xlim(k_dist[0], k_dist[-1])
    ax2.set_xticks(k_node)
    ax2.set_xticklabels(k_label)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Zoomed view near band gap", fontsize=14)
    
    fig2.tight_layout()
    fig2.savefig("silicon_gap_zoom.png", dpi=300)
    
    print(f"\nPlots saved:")
    print(f"  - silicon_bands_corrected.pdf")
    print(f"  - silicon_bands_corrected.png") 
    print(f"  - silicon_gap_zoom.png")
    
    return gap_direct if evals.shape[0] == 8 else estimated_gap

if __name__ == "__main__":
    gap = plot_bands_correct_path()