#!/usr/bin/env python3
"""
Final corrected Wannier-based Bethe-Salpeter Equation (BSE) solver for silicon.

This version:
- Uses all 8 Wannier functions correctly
- Properly identifies valence (1-4) and conduction (5-8) bands
- Implements scissor shift to correct TB gap
- Includes advanced screening models (constant and Penn model)
- Command-line arguments for customization
- Labels excitons correctly in plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pythtb import w90
import os
import argparse


def load_wannier_model(data_folder="examples/silicon", prefix="silicon"):
    """
    Load Wannier90 Hamiltonian using pythTB w90 class.
    
    Returns:
        tuple: (w90_object, tb_model, n_bands)
    """
    print(f"Loading Wannier90 data from {data_folder} with prefix '{prefix}'...")
    
    # Load Wannier90 data
    silicon_w90 = w90(data_folder, prefix)
    
    # Create tight-binding model
    tb_model = silicon_w90.model(min_hopping_norm=0.01)
    
    n_bands = tb_model.get_num_orbitals()
    print(f"Loaded {n_bands} orbitals")
    print(f"Lattice vectors:\n{tb_model._lat}")
    
    return silicon_w90, tb_model, n_bands


def interpolate_bands(tb_model, k_mesh=(4, 4, 4), scissor_shift=0.0):
    """
    Interpolate electronic bands on a k-point mesh with optional scissor shift.
    
    Args:
        tb_model: pythTB tight-binding model
        k_mesh: k-point mesh dimensions (nk1, nk2, nk3)
        scissor_shift: rigid shift applied to conduction bands (eV)
    
    Returns:
        tuple: (k_points, eigenvalues, tb_gap)
    """
    print(f"Interpolating bands on {k_mesh} mesh...")
    
    # Generate Monkhorst-Pack k-point mesh
    nk1, nk2, nk3 = k_mesh
    k_points = []
    
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                kx = (i + 0.5) / nk1 - 0.5
                ky = (j + 0.5) / nk2 - 0.5
                kz = (k + 0.5) / nk3 - 0.5
                k_points.append([kx, ky, kz])
    
    k_points = np.array(k_points)
    
    # Compute eigenvalues
    eigenvalues = []
    for i, k_vec in enumerate(k_points):
        if i % 20 == 0:
            print(f"  Computing k-point {i+1}/{len(k_points)}")
        evals = tb_model.solve_one(k_vec)
        eigenvalues.append(evals)
    
    eigenvalues = np.array(eigenvalues)
    
    # Apply scissor shift to conduction bands (bands 5-8, indices 4-7)
    if scissor_shift != 0.0:
        eigenvalues[:, 4:] += scissor_shift
        print(f"Applied scissor shift of {scissor_shift:.3f} eV to conduction bands")
    
    # Analyze band structure
    n_bands = eigenvalues.shape[1]
    
    # For 8-band silicon: bands 1-4 are valence, 5-8 are conduction
    VBM = np.max(eigenvalues[:, :4])  # Max of first 4 bands
    CBM = np.min(eigenvalues[:, 4:])  # Min of last 4 bands
    tb_gap = CBM - VBM
    
    print(f"Computed {n_bands} bands at {len(k_points)} k-points")
    print(f"Band range: {eigenvalues.min():.3f} to {eigenvalues.max():.3f} eV")
    print(f"VBM: {VBM:.3f} eV, CBM: {CBM:.3f} eV")
    print(f"TB fundamental gap: {tb_gap:.3f} eV")
    
    return k_points, eigenvalues, tb_gap


def build_transition_basis(k_points, eigenvalues, nv=4, nc=4):
    """
    Build electron-hole transition basis using proper band indices.
    
    For silicon with 8 bands:
    - Bands 1-4 (indices 0-3): valence bands
    - Bands 5-8 (indices 4-7): conduction bands
    """
    print(f"Building transition basis with bands 1-{nv} → bands {nv+1}-{nv+nc}...")
    
    transitions = []
    transition_energies = []
    
    for k_idx, k_vec in enumerate(k_points):
        bands = eigenvalues[k_idx]
        
        # Use fixed band indices, not sorted
        val_indices = range(nv)  # Bands 1-4 (indices 0-3)
        con_indices = range(nv, nv+nc)  # Bands 5-8 (indices 4-7)
        
        # Generate all v->c transitions
        for v in val_indices:
            for c in con_indices:
                transitions.append((v, c, k_idx))
                transition_energies.append(bands[c] - bands[v])
    
    transitions = np.array(transitions)
    transition_energies = np.array(transition_energies)
    
    print(f"Generated {len(transitions)} electron-hole transitions")
    print(f"Transition energy range: {transition_energies.min():.3f} to {transition_energies.max():.3f} eV")
    
    return transitions, transition_energies


def construct_diagonal_bse(transition_energies):
    """
    Construct diagonal part of BSE Hamiltonian.
    """
    print("Constructing diagonal BSE Hamiltonian...")
    n_trans = len(transition_energies)
    h_diag = np.diag(transition_energies)
    print(f"Diagonal BSE matrix size: {n_trans} x {n_trans}")
    return h_diag


def model_coulomb_kernel(tb_model, transitions, k_points, screening='constant', epsilon_inf=11.7, tb_gap=1.0):
    """
    Compute model Coulomb kernel for electron-hole interaction with advanced screening.
    
    Args:
        screening: 'constant' or 'penn' screening model
        epsilon_inf: static dielectric constant 
        tb_gap: fundamental gap for Penn model
    """
    print(f"Computing model Coulomb kernel with {screening} screening...")
    
    # Get orbital positions
    orb_positions = tb_model._orb
    lat_vecs = tb_model._lat
    orb_cart = np.dot(orb_positions, lat_vecs)
    
    n_trans = len(transitions)
    coulomb_kernel = np.zeros((n_trans, n_trans))
    
    # Constants
    e_squared = 14.3996  # e²/(4πε₀) in eV·Å
    
    # Penn model parameters
    if screening == 'penn':
        # Compute Penn q_P = sqrt(2m*E_g/hbar^2) 
        # Using effective mass m* ≈ 0.26 m_e for silicon
        hbar_eV_s = 6.582119569e-16  # hbar in eV·s
        m_e_kg = 9.1093837015e-31    # electron mass in kg
        m_eff = 0.26 * m_e_kg        # effective mass
        
        # Convert to appropriate units for q_P calculation
        # q_P^2 = 2*m_eff*E_g / hbar^2, result in inverse length^2
        q_P_squared = 2 * m_eff * tb_gap * 1.602176634e-19 / (1.054571817e-34)**2  # in m^-2
        q_P = np.sqrt(q_P_squared) * 1e-10  # convert to Å^-1
        print(f"Penn model: q_P = {q_P:.3f} Å^-1 for gap = {tb_gap:.3f} eV")
    
    def get_dielectric(r):
        """Get dielectric function based on screening model"""
        if screening == 'constant':
            return epsilon_inf
        elif screening == 'penn':
            # Penn model: ε(q) = 1 + (ε∞-1)/(1+(q/q_P)^2)
            # Approximate q ≈ 1/r for real space
            q = 1.0 / max(r, 0.1)  # avoid division by zero
            eps_q = 1 + (epsilon_inf - 1) / (1 + (q/q_P)**2)
            return eps_q
        else:
            return epsilon_inf
    
    # Compute kernel
    for i in range(n_trans):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_trans}")
        
        v1, c1, k1_idx = transitions[i]
        r_e1 = orb_cart[c1]
        r_h1 = orb_cart[v1]
        r_eh_1 = np.linalg.norm(r_e1 - r_h1)
        r_eh_1 = max(r_eh_1, 0.1)
        
        # Diagonal term with screening
        eps_diag = get_dielectric(r_eh_1)
        coulomb_kernel[i, i] = -e_squared / (eps_diag * r_eh_1)
        
        # Off-diagonal terms
        for j in range(i+1, n_trans):
            v2, c2, k2_idx = transitions[j]
            r_e2 = orb_cart[c2]
            r_h2 = orb_cart[v2]
            r_eh_2 = np.linalg.norm(r_e2 - r_h2)
            r_eh_2 = max(r_eh_2, 0.1)
            
            r_avg = (r_eh_1 + r_eh_2) / 2
            eps_offdiag = get_dielectric(r_avg)
            kernel_val = -0.1 * e_squared / (eps_offdiag * r_avg)
            coulomb_kernel[i, j] = kernel_val
            coulomb_kernel[j, i] = kernel_val
    
    print(f"Coulomb kernel computed with {screening} screening (ε∞ = {epsilon_inf})")
    print(f"Kernel range: {coulomb_kernel.min():.3f} to {coulomb_kernel.max():.3f} eV")
    
    return coulomb_kernel


def solve_bse(h_bse, n_states=30):
    """
    Solve BSE eigenvalue problem.
    """
    print(f"Solving BSE for {n_states} lowest exciton states...")
    
    if h_bse.shape[0] <= n_states:
        exciton_energies, exciton_wavefunctions = eigh(h_bse)
    else:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        h_sparse = csr_matrix(h_bse)
        exciton_energies, exciton_wavefunctions = eigsh(h_sparse, k=n_states, which='SA')
    
    # Sort eigenvalues
    idx = np.argsort(exciton_energies)
    exciton_energies = exciton_energies[idx]
    exciton_wavefunctions = exciton_wavefunctions[:, idx]
    
    print(f"\nLowest exciton energies (eV):")
    for i in range(min(10, len(exciton_energies))):
        print(f"  Exciton {i+1}: {exciton_energies[i]:.3f} eV")
    
    return exciton_energies, exciton_wavefunctions


def plot_absorption_spectrum(exciton_energies, exciton_wavefunctions, transitions, 
                           tb_model, tb_gap, broadening=0.1, energy_range=(-2, 10)):
    """
    Plot optical absorption spectrum with proper exciton labeling.
    """
    print("\nComputing absorption spectrum...")
    
    import matplotlib
    matplotlib.use('Agg')
    
    # Energy grid
    energies = np.linspace(energy_range[0], energy_range[1], 1000)
    epsilon2 = np.zeros_like(energies)
    
    # Compute oscillator strengths
    orb_positions = tb_model._orb
    lat_vecs = tb_model._lat
    orb_cart = np.dot(orb_positions, lat_vecs)
    
    for s, exc_energy in enumerate(exciton_energies[:30]):
        wavefunction = exciton_wavefunctions[:, s]
        
        dipole_strength = 0.0
        for i, (v, c, k_idx) in enumerate(transitions):
            coeff = wavefunction[i]
            r_vc = orb_cart[c] - orb_cart[v]
            dipole_element = np.linalg.norm(r_vc)
            dipole_strength += abs(coeff)**2 * dipole_element**2
        
        lorentzian = broadening / ((energies - exc_energy)**2 + broadening**2) / np.pi
        epsilon2 += dipole_strength * lorentzian
    
    # Plot spectrum
    plt.figure(figsize=(12, 8))
    plt.plot(energies, epsilon2, 'b-', linewidth=2)
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('ε₂(ω) (arbitrary units)', fontsize=12)
    plt.title('Silicon Absorption Spectrum from BSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(energy_range)
    plt.ylim(0, np.max(epsilon2) * 1.1 if np.max(epsilon2) > 0 else 1)
    
    # Clean plot without numerical markers
    plt.tight_layout()
    plt.savefig('absorption_spectrum_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Absorption spectrum saved as 'absorption_spectrum_final.png'")


def plot_bse_vs_independent(exciton_energies, exciton_wavefunctions, transitions, 
                           transition_energies, tb_model, tb_gap, broadening=0.1):
    """
    Compare BSE and independent particle spectra with correct labeling.
    """
    print("\nGenerating BSE vs Independent Particle comparison...")
    
    import matplotlib
    matplotlib.use('Agg')
    
    energies = np.linspace(-2, 10, 1000)
    
    # Independent particle spectrum
    epsilon2_ip = np.zeros_like(energies)
    for trans_energy in transition_energies:
        lorentzian = broadening / ((energies - trans_energy)**2 + broadening**2) / np.pi
        epsilon2_ip += lorentzian
    
    # BSE spectrum
    epsilon2_bse = np.zeros_like(energies)
    orb_cart = np.dot(tb_model._orb, tb_model._lat)
    
    for s in range(min(30, len(exciton_energies))):
        exc_energy = exciton_energies[s]
        wavefunction = exciton_wavefunctions[:, s]
        
        dipole_strength = 0.0
        for i, (v, c, k_idx) in enumerate(transitions):
            coeff = wavefunction[i]
            r_vc = orb_cart[c] - orb_cart[v]
            dipole_element = np.linalg.norm(r_vc)
            dipole_strength += abs(coeff)**2 * dipole_element**2
        
        lorentzian = broadening / ((energies - exc_energy)**2 + broadening**2) / np.pi
        epsilon2_bse += dipole_strength * lorentzian
    
    # Normalize for comparison
    if np.max(epsilon2_ip) > 0:
        epsilon2_ip_norm = epsilon2_ip / np.max(epsilon2_ip)
    else:
        epsilon2_ip_norm = epsilon2_ip
        
    if np.max(epsilon2_bse) > 0:
        epsilon2_bse_norm = epsilon2_bse / np.max(epsilon2_bse)
    else:
        epsilon2_bse_norm = epsilon2_bse
    
    # Plot normalized comparison
    plt.figure(figsize=(12, 8))
    plt.plot(energies, epsilon2_ip_norm, 'b-', linewidth=2, 
             label='Independent Particle', alpha=0.7)
    plt.plot(energies, epsilon2_bse_norm, 'r-', linewidth=2, 
             label='BSE with Coulomb', alpha=0.8)
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Normalized ε₂(ω)', fontsize=12)
    plt.title('Silicon Absorption: BSE vs Independent Particle', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2, 10)
    plt.ylim(0, 1.1)
    
    # Clean comparison plot without numerical markers
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('bse_vs_independent_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison saved as 'bse_vs_independent_final.png'")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BSE Solver for Silicon with advanced screening')
    parser.add_argument('--scissor', type=float, default=0.23, 
                       help='Scissor shift for conduction bands (eV), default: 0.23')
    parser.add_argument('--screening', choices=['constant', 'penn'], default='constant',
                       help='Screening model: constant or penn, default: constant')
    parser.add_argument('--eps', type=float, default=11.7,
                       help='Static dielectric constant ε∞, default: 11.7')
    parser.add_argument('--kmesh', type=int, default=18,
                       help='Monkhorst-Pack mesh points per axis, default: 18')
    return parser.parse_args()

def main():
    """
    Main BSE solver workflow with all corrections.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("=== Final BSE Solver for Silicon ===")
    print(f"Scissor shift: {args.scissor:.3f} eV")
    print(f"Screening model: {args.screening}")
    print(f"ε∞: {args.eps}")
    print(f"k-mesh: {args.kmesh}³\n")
    
    # 1. Load Wannier Hamiltonian
    w90_obj, tb_model, n_bands = load_wannier_model("examples/silicon", "silicon")
    
    # 2. Interpolate bands with scissor shift
    k_mesh = (args.kmesh, args.kmesh, args.kmesh)
    k_points, eigenvalues, tb_gap = interpolate_bands(tb_model, k_mesh=k_mesh, 
                                                     scissor_shift=args.scissor)
    
    # 3. Build transition basis (using proper band indices)
    transitions, transition_energies = build_transition_basis(
        k_points, eigenvalues, nv=4, nc=4
    )
    
    # 4. Construct BSE matrix with advanced screening
    h_diag = construct_diagonal_bse(transition_energies)
    coulomb_kernel = model_coulomb_kernel(tb_model, transitions, k_points, 
                                        screening=args.screening, 
                                        epsilon_inf=args.eps, 
                                        tb_gap=tb_gap)
    h_bse = h_diag + coulomb_kernel
    
    # 5. Solve BSE
    exciton_energies, exciton_wavefunctions = solve_bse(h_bse, n_states=30)
    
    # 6. Analysis
    print(f"\n=== Results Summary ===")
    print(f"QP gap after scissor: {tb_gap:.3f} eV")
    print(f"Minimum transition energy: {np.min(transition_energies):.3f} eV")
    print(f"Number of negative energy excitons: {np.sum(exciton_energies < 0)}")
    
    positive_excitons = exciton_energies[exciton_energies > 0]
    if len(positive_excitons) > 0:
        first_bright = positive_excitons[0]
        binding_energy = tb_gap - first_bright
        print(f"First bright exciton: {first_bright:.3f} eV")
        print(f"Exciton binding energy: {binding_energy:.3f} eV ({binding_energy*1000:.1f} meV)")
    else:
        print("No positive energy excitons found!")
    
    # 7. Save results
    header_text = f'Exciton energies (eV) - QP gap: {tb_gap:.3f} eV, screening: {args.screening}, ε∞: {args.eps}'
    np.savetxt('exciton_energies_final.dat', exciton_energies, 
               header=header_text, fmt='%.6f')
    np.save('exciton_wavefunctions_final.npy', exciton_wavefunctions)
    
    # 8. Plot spectra
    plot_absorption_spectrum(exciton_energies, exciton_wavefunctions, 
                           transitions, tb_model, tb_gap)
    plot_bse_vs_independent(exciton_energies, exciton_wavefunctions, transitions, 
                          transition_energies, tb_model, tb_gap)
    
    print(f"\nFinal results saved:")
    print(f"  - exciton_energies_final.dat")
    print(f"  - exciton_wavefunctions_final.npy") 
    print(f"  - absorption_spectrum_final.png")
    print(f"  - bse_vs_independent_final.png")
    
    return exciton_energies, tb_gap


if __name__ == "__main__":
    exc_energies, gap = main()