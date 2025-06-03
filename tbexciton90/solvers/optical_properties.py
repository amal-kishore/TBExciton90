"""Optical properties calculations including oscillator strengths."""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OpticalProperties:
    """Calculate optical properties from BSE solutions."""
    
    def __init__(self, tb_model, bse_solver):
        """
        Initialize optical properties calculator.
        
        Args:
            tb_model: TightBindingModel instance
            bse_solver: BSESolver instance
        """
        self.tb_model = tb_model
        self.bse_solver = bse_solver
        
    def compute_oscillator_strengths(self, kpoints: np.ndarray,
                                   eigenvectors: np.ndarray,
                                   exciton_wavefunctions: np.ndarray,
                                   Q: np.ndarray = None) -> np.ndarray:
        """
        Compute oscillator strengths for exciton states.
        
        f_S = |<0|P|S>|^2 where P is the momentum operator
        
        For Q=0 (Gamma point), only s-like excitons are bright.
        
        Args:
            kpoints: k-points array
            eigenvectors: Electronic eigenvectors
            exciton_wavefunctions: Exciton wavefunctions
            Q: Exciton center-of-mass momentum (default: Gamma)
            
        Returns:
            Array of oscillator strengths
        """
        if Q is None:
            Q = np.zeros(3)
            
        num_excitons = exciton_wavefunctions.shape[1]
        num_kpts = len(kpoints)
        num_val = self.bse_solver.num_valence
        num_cond = self.bse_solver.num_conduction
        
        oscillator_strengths = np.zeros(num_excitons)
        
        # Reshape exciton wavefunctions
        for s in range(num_excitons):
            psi_s = exciton_wavefunctions[:, s].reshape(
                (num_kpts, num_cond, num_val)
            )
            
            # Compute momentum matrix elements
            # Simplified: assume dipole approximation at Q=0
            if np.allclose(Q, 0):
                # For direct gap semiconductors at Gamma
                # Bright excitons have s-like envelope function
                # Check symmetry of wavefunction
                wf_sum = np.sum(np.abs(psi_s)**2)
                
                # Simple criterion: uniform phase gives bright exciton
                phase_coherence = np.abs(np.sum(psi_s))**2 / wf_sum
                
                oscillator_strengths[s] = phase_coherence
            else:
                # Finite Q: more complex selection rules
                oscillator_strengths[s] = 0.0
                
        # Normalize
        if oscillator_strengths.max() > 0:
            oscillator_strengths /= oscillator_strengths.max()
            
        return oscillator_strengths
    
    def compute_absorption_with_interaction(self, 
                                          exciton_energies: np.ndarray,
                                          oscillator_strengths: np.ndarray,
                                          energy_range: Tuple[float, float] = None,
                                          broadening: float = 0.05,
                                          num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical absorption spectrum with electron-hole interaction.
        
        Args:
            exciton_energies: Exciton energies
            oscillator_strengths: Oscillator strengths
            energy_range: Energy range for spectrum
            broadening: Lorentzian broadening in eV
            num_points: Number of energy points
            
        Returns:
            energies: Energy grid
            absorption: Absorption spectrum
        """
        if energy_range is None:
            e_min = max(0, exciton_energies[0] - 1.0)
            e_max = exciton_energies[-1] + 1.0
            energy_range = (e_min, e_max)
            
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        absorption = np.zeros_like(energies)
        
        # Add contribution from each bright exciton
        for E_exc, f_osc in zip(exciton_energies, oscillator_strengths):
            if f_osc > 1e-6:  # Only bright excitons
                # Lorentzian lineshape
                lorentzian = broadening / ((energies - E_exc)**2 + broadening**2)
                absorption += f_osc * lorentzian
                
        # Normalize to maximum
        if absorption.max() > 0:
            absorption /= absorption.max()
            
        return energies, absorption
    
    def compute_absorption_without_interaction(self,
                                             eigenvalues: np.ndarray,
                                             num_valence: int,
                                             energy_range: Tuple[float, float] = None,
                                             broadening: float = 0.05,
                                             num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical absorption spectrum without electron-hole interaction.
        This is the independent particle approximation.
        
        Args:
            eigenvalues: Electronic band energies
            num_valence: Number of valence bands
            energy_range: Energy range for spectrum
            broadening: Gaussian broadening in eV
            num_points: Number of energy points
            
        Returns:
            energies: Energy grid
            absorption: Absorption spectrum
        """
        # Calculate all possible transitions
        transitions = []
        for ik in range(len(eigenvalues)):
            for ic in range(num_valence, eigenvalues.shape[1]):
                for iv in range(num_valence):
                    energy = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                    transitions.append(energy)
                    
        transitions = np.array(transitions)
        
        if energy_range is None:
            e_min = max(0, transitions.min() - 0.5)
            e_max = transitions.max() + 0.5
            energy_range = (e_min, e_max)
            
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        absorption = np.zeros_like(energies)
        
        # Joint density of states
        for trans_e in transitions:
            # Gaussian broadening for independent particles
            gaussian = np.exp(-((energies - trans_e) / broadening)**2)
            absorption += gaussian
            
        # Normalize
        if absorption.max() > 0:
            absorption /= absorption.max()
            
        return energies, absorption
    
    def identify_bright_excitons(self, oscillator_strengths: np.ndarray,
                               threshold: float = 0.1) -> np.ndarray:
        """
        Identify optically bright exciton states.
        
        Args:
            oscillator_strengths: Array of oscillator strengths
            threshold: Minimum oscillator strength for bright exciton
            
        Returns:
            Indices of bright exciton states
        """
        bright_indices = np.where(oscillator_strengths > threshold)[0]
        
        logger.info(f"Found {len(bright_indices)} bright excitons out of {len(oscillator_strengths)} total states")
        
        return bright_indices