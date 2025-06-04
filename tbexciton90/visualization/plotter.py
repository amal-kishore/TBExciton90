"""Plotting utilities for exciton calculations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from typing import Optional, Tuple, List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class ExcitonPlotter:
    """Plotting utilities for electronic and excitonic properties."""
    
    def __init__(self, output_dir: str = "./results", style: str = "publication"):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plots
            style: Plotting style ("publication", "presentation", "default")
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        self._set_style(style)
        
    def _set_style(self, style: str):
        """Set matplotlib style."""
        if style == "publication":
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'lines.linewidth': 2,
                'axes.linewidth': 1.5,
                'xtick.major.width': 1.5,
                'ytick.major.width': 1.5,
            })
        elif style == "presentation":
            plt.rcParams.update({
                'font.size': 16,
                'axes.labelsize': 18,
                'axes.titlesize': 20,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'legend.fontsize': 16,
                'figure.dpi': 150,
                'savefig.dpi': 150,
                'lines.linewidth': 3,
                'axes.linewidth': 2,
                'xtick.major.width': 2,
                'ytick.major.width': 2,
            })
            
    def plot_band_structure(self, kpoints: np.ndarray, eigenvalues: np.ndarray,
                          num_valence: int, k_labels: Optional[List[str]] = None,
                          energy_range: Optional[Tuple[float, float]] = None,
                          save_name: str = "band_structure.png") -> Figure:
        """
        Plot electronic band structure.
        
        Args:
            kpoints: k-points array
            eigenvalues: Band energies
            num_valence: Number of valence bands
            k_labels: Labels for high-symmetry points
            energy_range: Energy range to plot
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Calculate k-path distance
        kdist = self._calculate_kpath_distance(kpoints)
        
        # Plot bands
        num_bands = eigenvalues.shape[1]
        for band in range(num_bands):
            if band < num_valence:
                color = 'blue'
                label = 'Valence' if band == 0 else None
            else:
                color = 'red'
                label = 'Conduction' if band == num_valence else None
                
            ax.plot(kdist, eigenvalues[:, band], color=color, alpha=0.7, label=label)
        
        # Mark band edges
        vbm = eigenvalues[:, num_valence-1].max()
        cbm = eigenvalues[:, num_valence].min()
        ax.axhline(y=vbm, color='black', linestyle='--', alpha=0.5, label=f'VBM: {vbm:.3f} eV')
        ax.axhline(y=cbm, color='black', linestyle=':', alpha=0.5, label=f'CBM: {cbm:.3f} eV')
        
        # Add band gap annotation
        gap = cbm - vbm
        gap_k = kdist[eigenvalues[:, num_valence].argmin()]
        ax.annotate(f'Gap: {gap:.3f} eV', xy=(gap_k, (vbm + cbm)/2),
                   xytext=(gap_k + 0.1, (vbm + cbm)/2),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
        
        # Formatting
        ax.set_xlabel('k-path')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Electronic Band Structure')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        if energy_range:
            ax.set_ylim(energy_range)
            
        # Add high-symmetry point labels
        if k_labels:
            self._add_kpoint_labels(ax, kdist, k_labels)
            
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved band structure plot to {filepath}")
        
        return fig
    
    def plot_exciton_spectrum(self, exciton_energies: np.ndarray,
                            oscillator_strengths: Optional[np.ndarray] = None,
                            num_states: int = 50,
                            save_name: str = "exciton_spectrum.png") -> Figure:
        """
        Plot exciton energy spectrum with distinction between bright and dark states.
        
        Args:
            exciton_energies: Exciton energies
            oscillator_strengths: Oscillator strengths (optional)
            num_states: Number of states to plot
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Limit number of states
        states_to_plot = min(num_states, len(exciton_energies))
        energies = exciton_energies[:states_to_plot]
        
        if oscillator_strengths is not None:
            # Separate bright and dark excitons
            strengths = oscillator_strengths[:states_to_plot]
            bright_threshold = 0.1 * strengths.max() if strengths.max() > 0 else 0.1
            
            bright_idx = strengths > bright_threshold
            dark_idx = ~bright_idx
            
            # Plot dark excitons
            if np.any(dark_idx):
                ax.scatter(np.where(dark_idx)[0], energies[dark_idx],
                          c='gray', s=30, alpha=0.5, label='Dark excitons')
            
            # Plot bright excitons
            if np.any(bright_idx):
                bright_energies = energies[bright_idx]
                bright_strengths = strengths[bright_idx]
                scatter = ax.scatter(np.where(bright_idx)[0], bright_energies,
                                   c=bright_strengths, s=200*bright_strengths/bright_strengths.max(),
                                   cmap='hot', alpha=0.8, edgecolors='black',
                                   label='Bright excitons', linewidth=1.5)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Oscillator Strength')
                
                # Annotate first few bright states with shorter arrows
                bright_indices = np.where(bright_idx)[0]
                for i, (idx, E) in enumerate(zip(bright_indices[:3], bright_energies[:3])):
                    # Calculate shorter offset for annotation
                    energy_range = energies.max() - energies.min()
                    offset_x = 0.5  # Small horizontal offset
                    offset_y = 0.01 * energy_range  # Small vertical offset (1% of range)
                    
                    ax.annotate(f'S{i+1}\n{E:.3f} eV', 
                               xy=(idx, E), xytext=(idx + offset_x, E + offset_y),
                               arrowprops=dict(arrowstyle='->', alpha=0.7, lw=1),
                               fontsize=9, ha='left')
        else:
            # Simple scatter plot
            ax.scatter(range(states_to_plot), energies,
                      c='purple', s=50, alpha=0.7)
        
        # Mark lowest exciton
        ax.scatter(0, energies[0], c='red', s=200, marker='*',
                  label=f'Lowest: {energies[0]:.3f} eV', zorder=5)
        
        ax.set_xlabel('Exciton State Index')
        ax.set_ylabel('Exciton Energy (eV)')
        ax.set_title('Exciton Energy Spectrum (Bright vs Dark States)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved exciton spectrum plot to {filepath}")
        
        return fig
    
    def plot_exciton_wavefunction(self, R_grid: np.ndarray,
                                wavefunction: np.ndarray,
                                state_index: int = 0,
                                plot_type: str = "1d",
                                save_name: Optional[str] = None) -> Figure:
        """
        Plot exciton wavefunction in real space.
        
        Args:
            R_grid: Real-space grid points
            wavefunction: Exciton wavefunction
            state_index: Index of exciton state
            plot_type: Type of plot ("1d", "2d", "3d")
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        if save_name is None:
            save_name = f"exciton_wavefunction_state{state_index}.png"
            
        # Calculate probability density
        density = np.abs(wavefunction)**2
        
        if plot_type == "1d":
            fig = self._plot_wavefunction_1d(R_grid, density, state_index)
        elif plot_type == "2d":
            fig = self._plot_wavefunction_2d(R_grid, density, state_index)
        else:
            fig = self._plot_wavefunction_3d(R_grid, density, state_index)
            
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved wavefunction plot to {filepath}")
        
        return fig
    
    def _plot_wavefunction_1d(self, R_grid: np.ndarray, density: np.ndarray,
                            state_index: int) -> Figure:
        """1D wavefunction plot."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract x-coordinates
        if R_grid.shape[1] >= 1:
            x_coords = R_grid[:, 0]
        else:
            x_coords = np.arange(len(density))
            
        ax.plot(x_coords, density, 'b-', linewidth=2)
        ax.fill_between(x_coords, density, alpha=0.3)
        
        ax.set_xlabel('Position x (Å)')
        ax.set_ylabel('|Ψ(R)|²')
        ax.set_title(f'Exciton Wavefunction Density (State {state_index})')
        ax.grid(True, alpha=0.3)
        
        # Add characteristic length
        max_idx = np.argmax(density)
        half_max = density.max() / 2
        indices = np.where(density > half_max)[0]
        if len(indices) > 1:
            fwhm = x_coords[indices[-1]] - x_coords[indices[0]]
            ax.annotate(f'FWHM: {fwhm:.2f} Å',
                       xy=(x_coords[max_idx], density[max_idx]),
                       xytext=(x_coords[max_idx] + 1, density[max_idx] * 0.8),
                       arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def _plot_wavefunction_2d(self, R_grid: np.ndarray, density: np.ndarray,
                            state_index: int) -> Figure:
        """2D wavefunction plot."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # Reshape density for 2D plotting
        grid_size = int(np.sqrt(len(density)))
        if grid_size**2 == len(density):
            density_2d = density.reshape(grid_size, grid_size)
            x = R_grid[:grid_size, 0]
            y = R_grid[::grid_size, 1] if R_grid.shape[1] > 1 else x
            
            im = ax.contourf(x, y, density_2d, levels=50, cmap='viridis')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('|Ψ(R)|²')
            
            ax.set_xlabel('x (Å)')
            ax.set_ylabel('y (Å)')
        else:
            # Fallback to scatter plot
            ax.scatter(R_grid[:, 0], R_grid[:, 1] if R_grid.shape[1] > 1 else np.zeros_like(R_grid[:, 0]),
                      c=density, s=50, cmap='viridis')
            ax.set_xlabel('x (Å)')
            ax.set_ylabel('y (Å)')
            
        ax.set_title(f'Exciton Wavefunction Density (State {state_index})')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
    
    def _plot_wavefunction_3d(self, R_grid: np.ndarray, density: np.ndarray,
                            state_index: int) -> Figure:
        """3D wavefunction plot."""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create isosurface at different density levels
        density_levels = [0.1, 0.5, 0.9] * density.max()
        
        # For simplicity, show scatter plot colored by density
        scatter = ax.scatter(R_grid[:, 0], 
                           R_grid[:, 1] if R_grid.shape[1] > 1 else np.zeros_like(R_grid[:, 0]),
                           R_grid[:, 2] if R_grid.shape[1] > 2 else np.zeros_like(R_grid[:, 0]),
                           c=density, s=50*density/density.max(), 
                           cmap='viridis', alpha=0.7)
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('|Ψ(R)|²')
        
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')
        ax.set_title(f'Exciton Wavefunction Density (State {state_index})')
        
        return fig
    
    def plot_optical_absorption_comparison(self, 
                                         energies_with: np.ndarray, 
                                         absorption_with: np.ndarray,
                                         energies_without: np.ndarray,
                                         absorption_without: np.ndarray,
                                         exciton_energies: Optional[np.ndarray] = None,
                                         oscillator_strengths: Optional[np.ndarray] = None,
                                         save_name: str = "optical_absorption_comparison.png") -> Figure:
        """
        Plot optical absorption spectrum with and without electron-hole interaction.
        
        Args:
            energies_with: Energy grid with interaction
            absorption_with: Absorption with e-h interaction
            energies_without: Energy grid without interaction
            absorption_without: Absorption without e-h interaction
            exciton_energies: Exciton energies (optional)
            oscillator_strengths: Oscillator strengths (optional)
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Plot absorption without interaction
        ax.plot(energies_without, absorption_without, 'k--', linewidth=2, 
                label='Without e-h interaction', alpha=0.7)
        ax.fill_between(energies_without, absorption_without, alpha=0.2, color='gray')
        
        # Plot absorption with interaction
        ax.plot(energies_with, absorption_with, 'b-', linewidth=2.5, 
                label='With e-h interaction')
        ax.fill_between(energies_with, absorption_with, alpha=0.3, color='blue')
        
        # Mark bright exciton peaks
        if exciton_energies is not None and oscillator_strengths is not None:
            bright_threshold = 0.1 * oscillator_strengths.max() if oscillator_strengths.max() > 0 else 0.1
            bright_idx = oscillator_strengths > bright_threshold
            
            bright_count = 0
            for i, (E_exc, f_osc) in enumerate(zip(exciton_energies, oscillator_strengths)):
                if f_osc > bright_threshold and energies_with.min() <= E_exc <= energies_with.max():
                    idx = np.argmin(np.abs(energies_with - E_exc))
                    ax.axvline(x=E_exc, color='red', linestyle=':', alpha=0.5)
                    
                    if bright_count < 3:  # Label first 3 bright excitons
                        ax.annotate(f'S{bright_count+1}\n{E_exc:.3f} eV', 
                                   xy=(E_exc, absorption_with[idx]),
                                   xytext=(E_exc + 0.1, absorption_with[idx] + 0.1),
                                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                                   fontsize=10, color='red')
                    bright_count += 1
        
        # Calculate and annotate binding energy
        if len(energies_without) > 0 and len(energies_with) > 0:
            # Find onset of absorption
            threshold = 0.01
            onset_without = energies_without[absorption_without > threshold][0] if np.any(absorption_without > threshold) else 0
            onset_with = energies_with[absorption_with > threshold][0] if np.any(absorption_with > threshold) else 0
            
            if onset_without > 0 and onset_with > 0:
                binding_energy = onset_without - onset_with
                ax.text(0.05, 0.95, f'Exciton binding: {binding_energy:.3f} eV',
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')
        
        ax.set_xlabel('Energy (eV)', fontsize=14)
        ax.set_ylabel('Absorption (arb. units)', fontsize=14)
        ax.set_title('Optical Absorption: Effect of Electron-Hole Interaction', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set reasonable x-limits
        if len(energies_with) > 0:
            x_min = max(0, energies_with[0])
            x_max = min(energies_with[-1], energies_with[0] + 3.0)
            ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved absorption comparison to {filepath}")
        
        return fig
    
    def plot_optical_absorption(self, energies: np.ndarray, absorption: np.ndarray,
                              exciton_energies: Optional[np.ndarray] = None,
                              save_name: str = "optical_absorption.png") -> Figure:
        """
        Plot optical absorption spectrum.
        
        Args:
            energies: Energy grid
            absorption: Absorption spectrum
            exciton_energies: Exciton peak positions (optional)
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(energies, absorption, 'b-', linewidth=2, label='Absorption')
        ax.fill_between(energies, absorption, alpha=0.3)
        
        # Mark exciton peaks
        if exciton_energies is not None:
            for i, E_exc in enumerate(exciton_energies[:10]):  # Show first 10
                if energies.min() <= E_exc <= energies.max():
                    idx = np.argmin(np.abs(energies - E_exc))
                    ax.axvline(x=E_exc, color='red', linestyle='--', alpha=0.5)
                    if i < 3:  # Label first 3
                        ax.annotate(f'S{i}', xy=(E_exc, absorption[idx]),
                                   xytext=(E_exc + 0.05, absorption[idx] + 0.1),
                                   arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Absorption (arb. units)')
        ax.set_title('Optical Absorption Spectrum')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved absorption spectrum to {filepath}")
        
        return fig
    
    def plot_summary(self, results: Dict[str, Any], save_name: str = "summary.png") -> Figure:
        """
        Create a summary plot with multiple panels.
        
        Args:
            results: Dictionary with calculation results
            save_name: Filename to save plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Band structure
        if 'eigenvalues' in results:
            ax1 = fig.add_subplot(gs[0, :])
            kdist = self._calculate_kpath_distance(results['kpoints'])
            for band in range(results['eigenvalues'].shape[1]):
                ax1.plot(kdist, results['eigenvalues'][:, band], 'b-', alpha=0.5)
            ax1.set_xlabel('k-path')
            ax1.set_ylabel('Energy (eV)')
            ax1.set_title('Electronic Band Structure')
            ax1.grid(True, alpha=0.3)
        
        # Panel 2: Exciton spectrum
        if 'exciton_energies' in results:
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(range(len(results['exciton_energies'][:50])),
                       results['exciton_energies'][:50], c='purple', s=50)
            ax2.set_xlabel('State Index')
            ax2.set_ylabel('Energy (eV)')
            ax2.set_title('Exciton Spectrum')
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Absorption spectrum
        if 'absorption' in results:
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(results['absorption_energies'], results['absorption'], 'g-')
            ax3.set_xlabel('Energy (eV)')
            ax3.set_ylabel('Absorption')
            ax3.set_title('Optical Absorption')
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Exciton wavefunction
        if 'exciton_wavefunction' in results:
            ax4 = fig.add_subplot(gs[2, :])
            density = np.abs(results['exciton_wavefunction'])**2
            ax4.plot(results['R_grid'][:, 0], density, 'r-')
            ax4.fill_between(results['R_grid'][:, 0], density, alpha=0.3)
            ax4.set_xlabel('Position (Å)')
            ax4.set_ylabel('|Ψ|²')
            ax4.set_title('Exciton Wavefunction (Ground State)')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Exciton Calculation Summary', fontsize=16)
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, bbox_inches='tight')
        logger.info(f"Saved summary plot to {filepath}")
        
        return fig
    
    def _calculate_kpath_distance(self, kpoints: np.ndarray) -> np.ndarray:
        """Calculate cumulative distance along k-path."""
        kdist = np.zeros(len(kpoints))
        for i in range(1, len(kpoints)):
            kdist[i] = kdist[i-1] + np.linalg.norm(kpoints[i] - kpoints[i-1])
        return kdist
    
    def _add_kpoint_labels(self, ax, kdist: np.ndarray, labels: List[str]):
        """Add high-symmetry point labels to band plot."""
        # Assume labels are for equally spaced points
        n_labels = len(labels)
        label_positions = np.linspace(kdist[0], kdist[-1], n_labels)
        
        for pos, label in zip(label_positions, labels):
            ax.axvline(x=pos, color='gray', linestyle='-', alpha=0.5)
            
        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels)