"""Beautiful plotting utilities for TBExciton90."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, List, Dict, Any
import os
import logging

from .plot_style import (
    set_publication_style, COLORS, PALETTES, 
    get_gradient_colormap, add_colorbar, add_text_box,
    save_figure
)

logger = logging.getLogger(__name__)

# Apply publication style
set_publication_style()


class BeautifulExcitonPlotter:
    """Beautiful plotting utilities for exciton calculations."""
    
    def __init__(self, output_dir: str = "./results"):
        """Initialize plotter with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_band_structure(self, kpoints: np.ndarray, eigenvalues: np.ndarray,
                          num_valence: int, k_labels: Optional[List[str]] = None,
                          energy_range: Optional[Tuple[float, float]] = None,
                          save_name: str = "band_structure.png") -> Figure:
        """Plot beautiful electronic band structure."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate k-path distance
        kdist = self._calculate_kpath_distance(kpoints)
        
        # Plot bands with gradient colors
        num_bands = eigenvalues.shape[1]
        
        # Valence bands - blue gradient
        val_colors = plt.cm.Blues(np.linspace(0.3, 0.9, num_valence))
        for i, band in enumerate(range(num_valence)):
            ax.plot(kdist, eigenvalues[:, band], color=val_colors[i], 
                   linewidth=2.5, alpha=0.8)
        
        # Conduction bands - red gradient
        num_cond = num_bands - num_valence
        cond_colors = plt.cm.Reds(np.linspace(0.3, 0.9, num_cond))
        for i, band in enumerate(range(num_valence, num_bands)):
            ax.plot(kdist, eigenvalues[:, band], color=cond_colors[i], 
                   linewidth=2.5, alpha=0.8)
        
        # Mark band edges
        vbm = eigenvalues[:, num_valence-1].max()
        cbm = eigenvalues[:, num_valence].min()
        
        # Add shaded regions
        ax.axhspan(vbm, cbm, alpha=0.1, color=COLORS['warning'], label='Band gap')
        
        # Add band edge lines
        ax.axhline(y=vbm, color=COLORS['primary'], linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'VBM: {vbm:.3f} eV')
        ax.axhline(y=cbm, color=COLORS['secondary'], linestyle='--', 
                  linewidth=2, alpha=0.7, label=f'CBM: {cbm:.3f} eV')
        
        # Add text box with gap info
        gap = cbm - vbm
        gap_text = f'Band gap\n{gap:.3f} eV'
        add_text_box(ax, gap_text, loc='upper right', 
                    facecolor=COLORS['light'], edgecolor=COLORS['dark'])
        
        # Styling
        ax.set_xlabel('Wave vector', fontsize=16, fontweight='bold')
        ax.set_ylabel('Energy (eV)', fontsize=16, fontweight='bold')
        ax.set_title('Electronic Band Structure', fontsize=18, fontweight='bold', pad=20)
        
        # Better grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Legend with custom styling
        legend = ax.legend(loc='lower right', frameon=True, 
                          fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        if energy_range:
            ax.set_ylim(energy_range)
            
        # Add high-symmetry point labels
        if k_labels:
            self._add_kpoint_labels(ax, kdist, k_labels)
            
        plt.tight_layout()
        
        # Save with high quality
        filepath = os.path.join(self.output_dir, save_name)
        save_figure(fig, filepath, dpi=300)
        logger.info(f"Saved band structure plot to {filepath}")
        
        return fig
    
    def plot_exciton_spectrum_beautiful(self, exciton_energies: np.ndarray,
                                      oscillator_strengths: Optional[np.ndarray] = None,
                                      num_states: int = 50,
                                      save_name: str = "exciton_spectrum.png") -> Figure:
        """Plot beautiful exciton energy spectrum."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        states_to_plot = min(num_states, len(exciton_energies))
        energies = exciton_energies[:states_to_plot]
        
        if oscillator_strengths is not None:
            strengths = oscillator_strengths[:states_to_plot]
            bright_threshold = 0.01 * strengths.max() if strengths.max() > 0 else 0.01
            
            # Separate bright and dark
            bright_mask = strengths > bright_threshold
            dark_mask = ~bright_mask
            
            # Plot dark excitons as small circles
            if np.any(dark_mask):
                dark_idx = np.where(dark_mask)[0]
                ax.scatter(dark_idx, energies[dark_mask], 
                          s=50, c=COLORS['dark'], alpha=0.3,
                          edgecolors='none', label='Dark excitons', zorder=1)
            
            # Plot bright excitons with size proportional to oscillator strength
            if np.any(bright_mask):
                bright_idx = np.where(bright_mask)[0]
                bright_energies = energies[bright_mask]
                bright_strengths = strengths[bright_mask]
                
                # Normalize sizes
                sizes = 100 + 400 * bright_strengths / bright_strengths.max()
                
                # Create colormap
                cmap = get_gradient_colormap(COLORS['purple'], COLORS['pink'])
                
                scatter = ax.scatter(bright_idx, bright_energies,
                                   c=bright_strengths, s=sizes,
                                   cmap=cmap, alpha=0.8,
                                   edgecolors=COLORS['dark'], linewidth=2,
                                   label='Bright excitons', zorder=3)
                
                # Add colorbar
                cbar = add_colorbar(fig, ax, scatter, label='Oscillator Strength')
                
                # Annotate brightest states
                sorted_idx = np.argsort(bright_strengths)[::-1]
                for i in range(min(3, len(sorted_idx))):
                    idx = bright_idx[sorted_idx[i]]
                    energy = bright_energies[sorted_idx[i]]
                    strength = bright_strengths[sorted_idx[i]]
                    
                    # Add arrow and label
                    ax.annotate(f'S{i+1}', xy=(idx, energy),
                               xytext=(idx + 2, energy + 0.02),
                               fontsize=14, fontweight='bold',
                               color=COLORS['secondary'],
                               arrowprops=dict(arrowstyle='->', 
                                             color=COLORS['secondary'],
                                             linewidth=2))
        else:
            # Simple plot
            ax.scatter(range(states_to_plot), energies,
                      s=100, c=COLORS['primary'], alpha=0.7,
                      edgecolors=COLORS['dark'], linewidth=1.5)
        
        # Highlight ground state
        ax.scatter(0, energies[0], s=300, marker='*',
                  color=COLORS['accent'], edgecolors=COLORS['dark'],
                  linewidth=2, zorder=5)
        
        # Add ground state info
        gs_text = f'Ground state\n{energies[0]:.3f} eV'
        add_text_box(ax, gs_text, loc='lower right',
                    facecolor=COLORS['light'], edgecolor=COLORS['dark'])
        
        # Styling
        ax.set_xlabel('Exciton State Index', fontsize=16, fontweight='bold')
        ax.set_ylabel('Energy (eV)', fontsize=16, fontweight='bold')
        ax.set_title('Exciton Energy Spectrum', fontsize=18, fontweight='bold', pad=20)
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Legend
        legend = ax.legend(loc='upper left', frameon=True,
                          fancybox=True, shadow=True, fontsize=14)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        save_figure(fig, filepath, dpi=300)
        logger.info(f"Saved exciton spectrum plot to {filepath}")
        
        return fig
    
    def plot_absorption_comparison_beautiful(self, 
                                           energies_with: np.ndarray, 
                                           absorption_with: np.ndarray,
                                           energies_without: np.ndarray,
                                           absorption_without: np.ndarray,
                                           exciton_energies: Optional[np.ndarray] = None,
                                           oscillator_strengths: Optional[np.ndarray] = None,
                                           save_name: str = "absorption_comparison.png") -> Figure:
        """Plot beautiful absorption spectrum comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot absorption without interaction
        ax.plot(energies_without, absorption_without, 
               color=COLORS['dark'], linewidth=3, linestyle='--',
               label='Independent particles', alpha=0.7, zorder=1)
        ax.fill_between(energies_without, absorption_without, 
                       alpha=0.2, color=COLORS['dark'])
        
        # Plot absorption with interaction
        ax.plot(energies_with, absorption_with,
               color=COLORS['primary'], linewidth=3.5,
               label='With excitons', zorder=3)
        ax.fill_between(energies_with, absorption_with,
                       alpha=0.3, color=COLORS['primary'])
        
        # Mark bright exciton peaks
        if exciton_energies is not None and oscillator_strengths is not None:
            bright_threshold = 0.01 * oscillator_strengths.max() if oscillator_strengths.max() > 0 else 0.01
            
            # Add vertical lines for bright excitons
            bright_count = 0
            for i, (E_exc, f_osc) in enumerate(zip(exciton_energies, oscillator_strengths)):
                if f_osc > bright_threshold and energies_with.min() <= E_exc <= energies_with.max():
                    # Vertical line with gradient
                    ax.axvline(x=E_exc, ymin=0, ymax=0.9,
                             color=COLORS['secondary'], linestyle=':',
                             linewidth=2, alpha=0.5)
                    
                    if bright_count < 3:  # Label first 3
                        idx = np.argmin(np.abs(energies_with - E_exc))
                        y_pos = absorption_with[idx]
                        
                        # Add peak label with arrow
                        ax.annotate(f'S{bright_count+1}\n{E_exc:.3f} eV',
                                   xy=(E_exc, y_pos),
                                   xytext=(E_exc + 0.15, y_pos + 0.15),
                                   fontsize=12, fontweight='bold',
                                   color=COLORS['secondary'],
                                   bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white',
                                           edgecolor=COLORS['secondary'],
                                           alpha=0.9),
                                   arrowprops=dict(arrowstyle='->',
                                                 color=COLORS['secondary'],
                                                 linewidth=2))
                    bright_count += 1
        
        # Calculate binding energy
        if len(energies_without) > 0 and len(energies_with) > 0:
            threshold = 0.01
            onset_without = energies_without[absorption_without > threshold][0] if np.any(absorption_without > threshold) else 0
            onset_with = energies_with[absorption_with > threshold][0] if np.any(absorption_with > threshold) else 0
            
            if onset_without > 0 and onset_with > 0:
                binding_energy = onset_without - onset_with
                
                # Add binding energy arrow
                y_arrow = 0.5
                ax.annotate('', xy=(onset_with, y_arrow), 
                           xytext=(onset_without, y_arrow),
                           arrowprops=dict(arrowstyle='<->', 
                                         color=COLORS['accent'],
                                         linewidth=3))
                ax.text((onset_with + onset_without)/2, y_arrow + 0.05,
                       f'Eb = {binding_energy:.3f} eV',
                       ha='center', va='bottom', fontsize=14,
                       fontweight='bold', color=COLORS['accent'])
        
        # Add theory explanation
        theory_text = (
            'Excitonic effects:\n'
            '• Red-shift absorption edge\n'
            '• Create sharp peaks\n'
            '• Enhance oscillator strength'
        )
        add_text_box(ax, theory_text, loc='upper right',
                    facecolor=COLORS['light'], edgecolor=COLORS['dark'])
        
        # Styling
        ax.set_xlabel('Photon Energy (eV)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Absorption (arb. units)', fontsize=16, fontweight='bold')
        ax.set_title('Optical Absorption: Effect of Electron-Hole Interaction',
                    fontsize=18, fontweight='bold', pad=20)
        
        # Grid and limits
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        if len(energies_with) > 0:
            x_min = max(0, energies_with[0])
            x_max = min(energies_with[-1], energies_with[0] + 3.0)
            ax.set_xlim(x_min, x_max)
        
        ax.set_ylim(0, 1.1)
        
        # Legend
        legend = ax.legend(loc='upper left', frameon=True,
                          fancybox=True, shadow=True, fontsize=14)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        save_figure(fig, filepath, dpi=300)
        logger.info(f"Saved absorption comparison to {filepath}")
        
        return fig
    
    def plot_exciton_wavefunction_beautiful(self, R_grid: np.ndarray,
                                          wavefunction: np.ndarray,
                                          state_index: int = 0,
                                          energy: Optional[float] = None,
                                          save_name: Optional[str] = None) -> Figure:
        """Plot beautiful exciton wavefunction."""
        if save_name is None:
            save_name = f"exciton_wavefunction_S{state_index}.png"
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                       gridspec_kw={'height_ratios': [1, 1.2]})
        
        # Extract position coordinate
        if R_grid.shape[1] >= 1:
            x_coords = R_grid[:, 0]
        else:
            x_coords = np.arange(len(wavefunction))
        
        # Top panel: Real and imaginary parts
        ax1.plot(x_coords, np.real(wavefunction), 
                color=COLORS['primary'], linewidth=2.5, 
                label='Real part', alpha=0.9)
        ax1.plot(x_coords, np.imag(wavefunction), 
                color=COLORS['secondary'], linewidth=2.5, 
                linestyle='--', label='Imaginary part', alpha=0.9)
        
        # Fill between for real part
        ax1.fill_between(x_coords, 0, np.real(wavefunction),
                        where=(np.real(wavefunction) > 0),
                        alpha=0.3, color=COLORS['primary'])
        ax1.fill_between(x_coords, 0, np.real(wavefunction),
                        where=(np.real(wavefunction) < 0),
                        alpha=0.3, color=COLORS['secondary'])
        
        ax1.set_ylabel('Ψ(R)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Exciton Wavefunction S{state_index}' + 
                     (f' (E = {energy:.3f} eV)' if energy else ''),
                     fontsize=16, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', fontsize=12)
        ax1.set_xlim(x_coords[0], x_coords[-1])
        
        # Bottom panel: Probability density
        density = np.abs(wavefunction)**2
        
        # Create gradient fill
        ax2.plot(x_coords, density, color=COLORS['purple'], 
                linewidth=3, label='|Ψ(R)|²')
        
        # Gradient fill under curve
        ax2.fill_between(x_coords, density, alpha=0.4,
                        color=COLORS['purple'])
        
        # Add characteristic length scale
        max_idx = np.argmax(density)
        half_max = density.max() / 2
        indices = np.where(density > half_max)[0]
        
        if len(indices) > 1:
            fwhm = x_coords[indices[-1]] - x_coords[indices[0]]
            
            # Add FWHM indicator
            ax2.hlines(half_max, x_coords[indices[0]], x_coords[indices[-1]],
                      colors=COLORS['accent'], linewidth=3)
            ax2.text(x_coords[max_idx], half_max * 1.1,
                    f'FWHM = {fwhm:.1f} Å',
                    ha='center', va='bottom', fontsize=12,
                    fontweight='bold', color=COLORS['accent'],
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white', alpha=0.9))
        
        # Calculate and display exciton radius
        density_norm = density / np.sum(density)
        r_avg = np.sum(np.abs(x_coords) * density_norm)
        
        radius_text = f'⟨|R|⟩ = {r_avg:.1f} Å'
        add_text_box(ax2, radius_text, loc='upper right',
                    facecolor=COLORS['light'], edgecolor=COLORS['dark'])
        
        ax2.set_xlabel('Electron-hole separation R (Å)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('|Ψ(R)|²', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(x_coords[0], x_coords[-1])
        ax2.set_ylim(0, None)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(self.output_dir, save_name)
        save_figure(fig, filepath, dpi=300)
        logger.info(f"Saved wavefunction plot to {filepath}")
        
        return fig
    
    def _calculate_kpath_distance(self, kpoints: np.ndarray) -> np.ndarray:
        """Calculate cumulative distance along k-path."""
        kdist = np.zeros(len(kpoints))
        for i in range(1, len(kpoints)):
            kdist[i] = kdist[i-1] + np.linalg.norm(kpoints[i] - kpoints[i-1])
        return kdist
    
    def _add_kpoint_labels(self, ax, kdist: np.ndarray, labels: List[str]):
        """Add high-symmetry point labels to band plot."""
        n_labels = len(labels)
        label_positions = np.linspace(kdist[0], kdist[-1], n_labels)
        
        for pos, label in zip(label_positions, labels):
            ax.axvline(x=pos, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            
        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels, fontsize=14, fontweight='bold')