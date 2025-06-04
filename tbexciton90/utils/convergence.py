"""Convergence testing utilities for TBExciton90."""

import numpy as np
import os
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from ..core.parser import Wannier90Parser
from ..core.tb_model import TightBindingModel
from ..solvers.bse_solver import BSESolver
from ..solvers.optical_properties import OpticalProperties

logger = logging.getLogger(__name__)


class ConvergenceTest:
    """Test convergence with respect to k-point sampling."""
    
    def __init__(self, hr_file: str, centres_file: str = None):
        """
        Initialize convergence tester.
        
        Args:
            hr_file: Path to Wannier90 HR file
            centres_file: Path to Wannier centres file (optional)
        """
        self.hr_file = hr_file
        self.centres_file = centres_file
        self.results = {}
        
    def test_kpoint_convergence(self, 
                               k_grids: List[Tuple[int, int, int]],
                               num_valence: int = 4,
                               num_conduction: int = 4,
                               num_exciton_states: int = 5,
                               screening: float = 0.1,
                               output_dir: str = "./convergence_test") -> Dict:
        """
        Test convergence with respect to k-point grid density.
        
        Args:
            k_grids: List of k-point grids to test [(nx1,ny1,nz1), (nx2,ny2,nz2), ...]
            num_valence: Number of valence bands
            num_conduction: Number of conduction bands  
            num_exciton_states: Number of exciton states to compute
            screening: Screening parameter
            output_dir: Directory to save results
            
        Returns:
            Dictionary with convergence results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting k-point convergence test with {len(k_grids)} grids")
        
        # Store results for each k-grid
        k_points_list = []
        band_gaps = []
        exciton_energies_list = []
        binding_energies = []
        oscillator_strengths_list = []
        
        for i, (nx, ny, nz) in enumerate(k_grids):
            logger.info(f"Testing k-grid {i+1}/{len(k_grids)}: {nx}×{ny}×{nz}")
            
            try:
                # Generate k-point grid
                kpt_file = self._generate_kpoint_file(nx, ny, nz, output_dir, i)
                
                # Run calculation
                results = self._run_single_calculation(
                    kpt_file, num_valence, num_conduction, 
                    num_exciton_states, screening
                )
                
                # Store results
                total_k = nx * ny * nz
                k_points_list.append(total_k)
                band_gaps.append(results['bandgap'])
                exciton_energies_list.append(results['exciton_energies'])
                binding_energies.append(results['binding_energy'])
                oscillator_strengths_list.append(results['oscillator_strengths'])
                
                logger.info(f"  Band gap: {results['bandgap']:.4f} eV")
                logger.info(f"  Lowest exciton: {results['exciton_energies'][0]:.4f} eV")
                logger.info(f"  Binding energy: {results['binding_energy']:.4f} eV")
                
            except Exception as e:
                logger.error(f"Failed calculation for k-grid {nx}×{ny}×{nz}: {e}")
                continue
        
        # Compile results
        convergence_results = {
            'k_grids': k_grids,
            'k_points': k_points_list,
            'band_gaps': band_gaps,
            'exciton_energies': exciton_energies_list,
            'binding_energies': binding_energies,
            'oscillator_strengths': oscillator_strengths_list
        }
        
        # Analyze convergence
        convergence_analysis = self._analyze_convergence(convergence_results)
        convergence_results.update(convergence_analysis)
        
        # Generate plots
        self._plot_convergence(convergence_results, output_dir)
        
        # Save results
        self._save_convergence_results(convergence_results, output_dir)
        
        return convergence_results
    
    def _generate_kpoint_file(self, nx: int, ny: int, nz: int, 
                             output_dir: str, index: int) -> str:
        """Generate k-point file for given grid."""
        # Create uniform k-point grid
        kpoints = []
        weights = []
        
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    kx = ix / nx
                    ky = iy / ny  
                    kz = iz / nz
                    kpoints.append([kx, ky, kz])
                    weights.append(1.0/(nx*ny*nz))
        
        # Write k-point file
        kpt_file = os.path.join(output_dir, f"kpoints_{nx}x{ny}x{nz}.kpt")
        with open(kpt_file, 'w') as f:
            f.write(f"{len(kpoints)}\n")
            for i, (kpt, w) in enumerate(zip(kpoints, weights)):
                f.write(f"{kpt[0]:12.8f} {kpt[1]:12.8f} {kpt[2]:12.8f} {w:12.8f}\n")
        
        return kpt_file
    
    def _run_single_calculation(self, kpt_file: str, num_valence: int, 
                               num_conduction: int, num_exciton_states: int,
                               screening: float) -> Dict:
        """Run single BSE calculation."""
        # Parse files
        parser = Wannier90Parser()
        parser.parse_hr_file(self.hr_file)
        parser.parse_kpt_file(kpt_file)
        if self.centres_file:
            parser.parse_centres_file(self.centres_file)
        
        # Compute band structure
        tb_model = TightBindingModel(parser)
        eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)
        
        # Get band gap
        vbm = eigenvalues[:, num_valence-1].max()
        cbm = eigenvalues[:, num_valence].min()
        bandgap = cbm - vbm
        
        # Solve BSE
        bse_solver = BSESolver(tb_model, num_valence=num_valence, 
                              num_conduction=num_conduction)
        bse_solver.set_screening('constant', {'W0': screening})
        
        exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
            parser.kpoints, eigenvalues, eigenvectors, num_states=num_exciton_states
        )
        
        # Compute optical properties
        optical = OpticalProperties(tb_model, bse_solver)
        oscillator_strengths = optical.compute_oscillator_strengths(
            parser.kpoints, eigenvectors, exciton_wavefunctions
        )
        
        # Calculate binding energy
        binding_energy = bandgap - exciton_energies[0]
        
        return {
            'bandgap': bandgap,
            'exciton_energies': exciton_energies,
            'binding_energy': binding_energy,
            'oscillator_strengths': oscillator_strengths
        }
    
    def _analyze_convergence(self, results: Dict) -> Dict:
        """Analyze convergence criteria."""
        k_points = np.array(results['k_points'])
        band_gaps = np.array(results['band_gaps'])
        binding_energies = np.array(results['binding_energies'])
        
        # Get lowest exciton energies
        lowest_excitons = [energies[0] for energies in results['exciton_energies']]
        lowest_excitons = np.array(lowest_excitons)
        
        analysis = {}
        
        # Check convergence (difference between last two calculations)
        if len(k_points) >= 2:
            bandgap_diff = abs(band_gaps[-1] - band_gaps[-2])
            exciton_diff = abs(lowest_excitons[-1] - lowest_excitons[-2])
            binding_diff = abs(binding_energies[-1] - binding_energies[-2])
            
            analysis['bandgap_convergence'] = bandgap_diff < 0.001  # 1 meV
            analysis['exciton_convergence'] = exciton_diff < 0.001   # 1 meV  
            analysis['binding_convergence'] = binding_diff < 0.001   # 1 meV
            
            analysis['convergence_thresholds'] = {
                'bandgap_diff': bandgap_diff,
                'exciton_diff': exciton_diff, 
                'binding_diff': binding_diff
            }
        
        # Estimate convergence based on trend
        if len(k_points) >= 3:
            # Fit to 1/N_k dependence
            try:
                inv_k = 1.0 / k_points
                
                # Fit band gap
                bg_fit = np.polyfit(inv_k, band_gaps, 1)
                analysis['bandgap_extrapolated'] = bg_fit[1]  # Intercept at 1/N_k = 0
                
                # Fit exciton energy
                ex_fit = np.polyfit(inv_k, lowest_excitons, 1)
                analysis['exciton_extrapolated'] = ex_fit[1]
                
                # Fit binding energy
                be_fit = np.polyfit(inv_k, binding_energies, 1)
                analysis['binding_extrapolated'] = be_fit[1]
                
            except:
                logger.warning("Could not perform extrapolation analysis")
        
        return analysis
    
    def _plot_convergence(self, results: Dict, output_dir: str):
        """Generate convergence plots."""
        k_points = results['k_points']
        band_gaps = results['band_gaps']
        binding_energies = results['binding_energies']
        lowest_excitons = [energies[0] for energies in results['exciton_energies']]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Band gap convergence
        ax1.plot(k_points, band_gaps, 'o-', color='blue', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of k-points')
        ax1.set_ylabel('Band Gap (eV)')
        ax1.set_title('Band Gap Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Exciton energy convergence  
        ax2.plot(k_points, lowest_excitons, 'o-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of k-points')
        ax2.set_ylabel('Lowest Exciton Energy (eV)')
        ax2.set_title('Exciton Energy Convergence')
        ax2.grid(True, alpha=0.3)
        
        # Binding energy convergence
        ax3.plot(k_points, binding_energies, 'o-', color='green', linewidth=2, markersize=6)
        ax3.set_xlabel('Number of k-points')
        ax3.set_ylabel('Binding Energy (eV)')
        ax3.set_title('Binding Energy Convergence')
        ax3.grid(True, alpha=0.3)
        
        # Convergence vs 1/N_k
        inv_k = [1.0/k for k in k_points]
        ax4.plot(inv_k, lowest_excitons, 'o-', color='purple', linewidth=2, markersize=6)
        ax4.set_xlabel('1/N_k')
        ax4.set_ylabel('Lowest Exciton Energy (eV)')
        ax4.set_title('Convergence Extrapolation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kpoint_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence plots saved to {output_dir}/kpoint_convergence.png")
    
    def _save_convergence_results(self, results: Dict, output_dir: str):
        """Save convergence results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_results[key] = [arr.tolist() for arr in value]
            else:
                json_results[key] = value
        
        # Save to JSON
        with open(os.path.join(output_dir, 'convergence_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save summary to text file
        with open(os.path.join(output_dir, 'convergence_summary.txt'), 'w') as f:
            f.write("K-Point Convergence Test Summary\n")
            f.write("=" * 40 + "\n\n")
            
            for i, (k_grid, k_total) in enumerate(zip(results['k_grids'], results['k_points'])):
                f.write(f"Grid {i+1}: {k_grid[0]}×{k_grid[1]}×{k_grid[2]} ({k_total} k-points)\n")
                f.write(f"  Band gap: {results['band_gaps'][i]:.6f} eV\n")
                f.write(f"  Lowest exciton: {results['exciton_energies'][i][0]:.6f} eV\n")
                f.write(f"  Binding energy: {results['binding_energies'][i]:.6f} eV\n\n")
            
            if 'convergence_thresholds' in results:
                f.write("Convergence Analysis:\n")
                f.write("-" * 20 + "\n")
                thresh = results['convergence_thresholds']
                f.write(f"Band gap difference: {thresh['bandgap_diff']:.6f} eV\n")
                f.write(f"Exciton difference: {thresh['exciton_diff']:.6f} eV\n")
                f.write(f"Binding difference: {thresh['binding_diff']:.6f} eV\n\n")
                
                f.write(f"Converged (< 1 meV): ")
                if results.get('bandgap_convergence', False) and results.get('exciton_convergence', False):
                    f.write("YES\n")
                else:
                    f.write("NO - Need finer k-point grid\n")
        
        logger.info(f"Convergence results saved to {output_dir}")


def suggest_kpoint_grids(material_type: str = "3D") -> List[Tuple[int, int, int]]:
    """Suggest k-point grids for convergence testing."""
    if material_type.lower() == "3d":
        return [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (10,10,10)]
    elif material_type.lower() == "2d":
        return [(4,4,1), (6,6,1), (8,8,1), (12,12,1), (16,16,1)]
    elif material_type.lower() == "1d":
        return [(8,1,1), (12,1,1), (16,1,1), (24,1,1), (32,1,1)]
    else:
        return [(2,2,2), (4,4,4), (6,6,6), (8,8,8)]