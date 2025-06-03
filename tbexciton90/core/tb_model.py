"""Tight-binding model implementation with GPU support."""

import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class TightBindingModel:
    """
    Tight-binding model from Wannier90 data with optional GPU acceleration.
    """
    
    def __init__(self, parser, use_gpu: bool = False):
        """
        Initialize tight-binding model.
        
        Args:
            parser: Wannier90Parser instance with parsed data
            use_gpu: Whether to use GPU acceleration if available
        """
        self.parser = parser
        self.num_wann = parser.num_wann
        self.use_gpu = use_gpu and HAS_CUPY
        
        if self.use_gpu:
            logger.info("GPU acceleration enabled with CuPy")
            self.xp = cp
        else:
            if use_gpu and not HAS_CUPY:
                logger.warning("GPU requested but CuPy not available, using CPU")
            self.xp = np
            
        # Precompute data for GPU if needed
        self._prepare_gpu_data()
        
    def _prepare_gpu_data(self):
        """Prepare data structures for efficient GPU computation."""
        if not self.use_gpu:
            return
            
        # Convert HR data to GPU-friendly format
        R_vectors = []
        H_values = []
        indices = []
        
        for key, value in self.parser._hr_dict.items():
            R_vectors.append(key[0])
            indices.append([key[1], key[2]])
            H_values.append(value)
        
        self.gpu_R = cp.asarray(R_vectors, dtype=cp.float64)
        self.gpu_indices = cp.asarray(indices, dtype=cp.int32)
        self.gpu_H_values = cp.asarray(H_values, dtype=cp.complex128)
        
    def construct_hamiltonian(self, k: Union[np.ndarray, list]) -> np.ndarray:
        """
        Construct tight-binding Hamiltonian at k-point.
        
        H_{nm}(k) = sum_R H_{nm}(R) * exp(i * k.R)
        
        Args:
            k: k-point in reciprocal lattice units
            
        Returns:
            Complex Hamiltonian matrix
        """
        if self.use_gpu:
            return self._construct_hamiltonian_gpu(k)
        else:
            return self._construct_hamiltonian_cpu(k)
    
    def _construct_hamiltonian_cpu(self, k: Union[np.ndarray, list]) -> np.ndarray:
        """CPU implementation of Hamiltonian construction."""
        H = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        k_array = np.asarray(k)
        
        for key, value in self.parser._hr_dict.items():
            R, m, n = key
            phase = np.exp(2j * np.pi * np.dot(k_array, R))
            H[n, m] += value * phase
            
        return H
    
    def _construct_hamiltonian_gpu(self, k: Union[np.ndarray, list]) -> np.ndarray:
        """GPU implementation of Hamiltonian construction."""
        k_gpu = cp.asarray(k, dtype=cp.float64)
        H_gpu = cp.zeros((self.num_wann, self.num_wann), dtype=cp.complex128)
        
        # Compute all phases at once
        phases = cp.exp(2j * cp.pi * cp.dot(self.gpu_R, k_gpu))
        
        # Fill Hamiltonian matrix
        for i in range(len(self.gpu_H_values)):
            m, n = self.gpu_indices[i]
            H_gpu[n, m] += self.gpu_H_values[i] * phases[i]
        
        return cp.asnumpy(H_gpu)
    
    def compute_bands(self, kpoints: np.ndarray, 
                     return_eigenvectors: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute electronic band structure at given k-points.
        
        Args:
            kpoints: Array of k-points
            return_eigenvectors: Whether to return eigenvectors
            
        Returns:
            eigenvalues: Band energies
            eigenvectors: Band eigenstates (if requested)
        """
        num_kpts = len(kpoints)
        eigenvalues = np.zeros((num_kpts, self.num_wann))
        
        if return_eigenvectors:
            eigenvectors = np.zeros((num_kpts, self.num_wann, self.num_wann), dtype=complex)
        else:
            eigenvectors = None
        
        if self.use_gpu:
            return self._compute_bands_gpu(kpoints, eigenvalues, eigenvectors)
        else:
            return self._compute_bands_cpu(kpoints, eigenvalues, eigenvectors)
    
    def _compute_bands_cpu(self, kpoints: np.ndarray, 
                          eigenvalues: np.ndarray, 
                          eigenvectors: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """CPU implementation of band structure calculation."""
        from scipy.linalg import eigh
        
        for ik, k in enumerate(kpoints):
            H = self.construct_hamiltonian(k)
            
            if eigenvectors is not None:
                eigvals, eigvecs = eigh(H)
                eigenvalues[ik] = eigvals
                eigenvectors[ik] = eigvecs
            else:
                eigvals = eigh(H, eigvals_only=True)
                eigenvalues[ik] = eigvals
                
        return eigenvalues, eigenvectors
    
    def _compute_bands_gpu(self, kpoints: np.ndarray,
                          eigenvalues: np.ndarray,
                          eigenvectors: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """GPU implementation of band structure calculation."""
        # Process in batches for memory efficiency
        batch_size = min(100, len(kpoints))
        
        for batch_start in range(0, len(kpoints), batch_size):
            batch_end = min(batch_start + batch_size, len(kpoints))
            batch_kpoints = kpoints[batch_start:batch_end]
            
            # Compute Hamiltonians for batch
            H_batch = []
            for k in batch_kpoints:
                H = self.construct_hamiltonian(k)
                H_batch.append(H)
            
            # Convert to GPU array
            H_batch_gpu = cp.asarray(H_batch, dtype=cp.complex128)
            
            # Batch diagonalization
            for i, ik in enumerate(range(batch_start, batch_end)):
                if eigenvectors is not None:
                    eigvals, eigvecs = cp.linalg.eigh(H_batch_gpu[i])
                    eigenvalues[ik] = cp.asnumpy(eigvals)
                    eigenvectors[ik] = cp.asnumpy(eigvecs)
                else:
                    eigvals = cp.linalg.eigvalsh(H_batch_gpu[i])
                    eigenvalues[ik] = cp.asnumpy(eigvals)
                    
        return eigenvalues, eigenvectors
    
    def compute_berry_phase(self, k_path: np.ndarray, band_indices: list) -> float:
        """
        Compute Berry phase along a closed k-path.
        
        Args:
            k_path: Array of k-points forming a closed loop
            band_indices: Indices of bands to include
            
        Returns:
            Berry phase in units of 2π
        """
        _, eigenvectors = self.compute_bands(k_path, return_eigenvectors=True)
        
        berry_phase = 0.0
        num_kpts = len(k_path)
        
        for n in band_indices:
            for ik in range(num_kpts):
                ik_next = (ik + 1) % num_kpts
                overlap = np.vdot(eigenvectors[ik, :, n], eigenvectors[ik_next, :, n])
                berry_phase += np.angle(overlap)
                
        return berry_phase / (2 * np.pi)
    
    def compute_velocity(self, k: Union[np.ndarray, list], 
                        band_index: int,
                        dk: float = 1e-5) -> np.ndarray:
        """
        Compute group velocity v = ∇_k E(k) for a given band.
        
        Args:
            k: k-point
            band_index: Band index
            dk: Small k-space displacement for numerical derivative
            
        Returns:
            Group velocity vector
        """
        velocity = np.zeros(3)
        
        for i in range(3):
            k_plus = np.array(k)
            k_minus = np.array(k)
            k_plus[i] += dk
            k_minus[i] -= dk
            
            E_plus = self.compute_bands(k_plus[np.newaxis, :], return_eigenvectors=False)[0][0, band_index]
            E_minus = self.compute_bands(k_minus[np.newaxis, :], return_eigenvectors=False)[0][0, band_index]
            
            velocity[i] = (E_plus - E_minus) / (2 * dk)
            
        return velocity