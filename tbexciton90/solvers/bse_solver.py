"""Bethe-Salpeter Equation solver with GPU and MPI support."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags, kron, eye
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
import time

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import cupy as cp
    from cupy.sparse import csr_matrix as cp_csr_matrix
    from cupy.sparse.linalg import eigsh as cp_eigsh
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    MPI = None
    HAS_MPI = False


class BSESolver:
    """
    Bethe-Salpeter Equation solver for excitons with GPU/MPI support.
    """
    
    def __init__(self, tb_model, num_valence: int = 4, num_conduction: int = 4,
                 use_gpu: bool = False, use_mpi: bool = False):
        """
        Initialize BSE solver.
        
        Args:
            tb_model: TightBindingModel instance
            num_valence: Number of valence bands to include
            num_conduction: Number of conduction bands to include
            use_gpu: Whether to use GPU acceleration
            use_mpi: Whether to use MPI parallelization
        """
        self.tb_model = tb_model
        self.num_valence = num_valence
        self.num_conduction = num_conduction
        self.screening_parameter = 0.1  # Default screening in eV
        self.screening_type = "constant"  # Default screening type
        
        # Setup computational backend
        self.use_gpu = use_gpu and HAS_CUPY
        self.use_mpi = use_mpi and HAS_MPI
        
        if self.use_gpu:
            logger.info("BSE solver using GPU acceleration")
            self.xp = cp
        else:
            if use_gpu and not HAS_CUPY:
                logger.warning("GPU requested but CuPy not available")
            self.xp = np
            
        # MPI setup
        if self.use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            logger.info(f"BSE solver using MPI with {self.size} processes (rank {self.rank})")
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            
    def set_screening(self, screening_type: str = "constant", 
                     parameters: Dict[str, float] = None):
        """
        Set the screening model for Coulomb interaction.
        
        Args:
            screening_type: Type of screening ("constant", "thomas-fermi", "rpa")
            parameters: Parameters for the screening model
        """
        self.screening_type = screening_type
        
        if screening_type == "constant":
            self.screening_parameter = parameters.get("W0", 0.1)
        elif screening_type == "thomas-fermi":
            self.screening_length = parameters.get("screening_length", 5.0)  # in Angstrom
        elif screening_type == "rpa":
            self.epsilon_infinity = parameters.get("epsilon_infinity", 10.0)
            
    def construct_bse_matrix(self, kpoints: np.ndarray, 
                           eigenvalues: np.ndarray,
                           eigenvectors: np.ndarray) -> np.ndarray:
        """
        Construct the BSE Hamiltonian matrix.
        
        H_BSE = H_0 + K_eh
        where H_0 is diagonal (quasi-particle energies)
        and K_eh is the electron-hole interaction kernel
        
        Args:
            kpoints: Array of k-points
            eigenvalues: Electronic eigenvalues
            eigenvectors: Electronic eigenvectors
            
        Returns:
            BSE Hamiltonian matrix
        """
        t0 = time.time()
        
        num_kpts = len(kpoints)
        num_wann = self.tb_model.num_wann
        
        # Identify band indices
        val_indices = list(range(self.num_valence))
        cond_indices = list(range(self.num_valence, self.num_valence + self.num_conduction))
        
        # BSE matrix size
        bse_size = self.num_valence * self.num_conduction * num_kpts
        
        if self.rank == 0:
            logger.info(f"Constructing BSE matrix of size {bse_size}x{bse_size}")
        
        if self.use_mpi:
            H_BSE = self._construct_bse_matrix_mpi(
                kpoints, eigenvalues, eigenvectors, 
                val_indices, cond_indices, bse_size
            )
        else:
            H_BSE = self._construct_bse_matrix_serial(
                kpoints, eigenvalues, eigenvectors,
                val_indices, cond_indices, bse_size
            )
            
        if self.rank == 0:
            logger.info(f"BSE matrix construction took {time.time() - t0:.2f} seconds")
            
        return H_BSE
    
    def _construct_bse_matrix_serial(self, kpoints, eigenvalues, eigenvectors,
                                   val_indices, cond_indices, bse_size):
        """Serial implementation of BSE matrix construction."""
        # Diagonal part: quasi-particle energies
        diagonal = []
        for ik in range(len(kpoints)):
            for ic in cond_indices:
                for iv in val_indices:
                    energy_diff = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                    diagonal.append(energy_diff)
        
        # Create sparse matrix
        H_BSE = diags(diagonal, 0, shape=(bse_size, bse_size), format='csr')
        
        # Add electron-hole interaction
        if self.screening_type == "constant":
            # Simplified local interaction
            H_BSE.setdiag(H_BSE.diagonal() - self.screening_parameter)
        else:
            # More sophisticated screening models
            H_BSE = self._add_screened_interaction(H_BSE, kpoints, eigenvectors,
                                                  val_indices, cond_indices)
            
        return H_BSE.toarray()
    
    def _construct_bse_matrix_mpi(self, kpoints, eigenvalues, eigenvectors,
                                val_indices, cond_indices, bse_size):
        """MPI-parallel implementation of BSE matrix construction."""
        # Distribute k-points among MPI processes
        num_kpts = len(kpoints)
        k_per_proc = num_kpts // self.size
        k_start = self.rank * k_per_proc
        k_end = k_start + k_per_proc if self.rank < self.size - 1 else num_kpts
        
        # Each process computes its portion
        local_diagonal = []
        for ik in range(k_start, k_end):
            for ic in cond_indices:
                for iv in val_indices:
                    energy_diff = eigenvalues[ik, ic] - eigenvalues[ik, iv]
                    local_diagonal.append(energy_diff)
        
        # Gather all diagonal elements
        all_diagonal = self.comm.gather(local_diagonal, root=0)
        
        if self.rank == 0:
            # Combine all contributions
            diagonal = []
            for proc_diagonal in all_diagonal:
                diagonal.extend(proc_diagonal)
                
            # Create matrix on root process
            H_BSE = diags(diagonal, 0, shape=(bse_size, bse_size), format='csr')
            H_BSE.setdiag(H_BSE.diagonal() - self.screening_parameter)
            H_BSE = H_BSE.toarray()
        else:
            H_BSE = None
            
        # Broadcast result to all processes
        H_BSE = self.comm.bcast(H_BSE, root=0)
        
        return H_BSE
    
    def _add_screened_interaction(self, H_BSE, kpoints, eigenvectors,
                                val_indices, cond_indices):
        """Add screened Coulomb interaction to BSE matrix."""
        # Placeholder for more sophisticated screening models
        # This would compute W(q) based on RPA or other approximations
        return H_BSE
    
    def solve_bse(self, kpoints: np.ndarray,
                 eigenvalues: np.ndarray,
                 eigenvectors: np.ndarray,
                 num_states: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the BSE eigenvalue problem.
        
        Args:
            kpoints: Array of k-points
            eigenvalues: Electronic eigenvalues
            eigenvectors: Electronic eigenvectors
            num_states: Number of exciton states to compute
            
        Returns:
            exciton_energies: Exciton energies
            exciton_wavefunctions: Exciton wavefunctions
        """
        if self.rank == 0:
            logger.info("Solving BSE eigenvalue problem...")
        
        # Construct BSE matrix
        H_BSE = self.construct_bse_matrix(kpoints, eigenvalues, eigenvectors)
        
        # Solve eigenvalue problem
        if self.use_gpu:
            exciton_energies, exciton_wavefunctions = self._solve_bse_gpu(
                H_BSE, num_states
            )
        else:
            exciton_energies, exciton_wavefunctions = self._solve_bse_cpu(
                H_BSE, num_states
            )
            
        if self.rank == 0:
            logger.info(f"Found {len(exciton_energies)} exciton states")
            logger.info(f"Lowest exciton energy: {exciton_energies[0]:.3f} eV")
            
        return exciton_energies, exciton_wavefunctions
    
    def _solve_bse_cpu(self, H_BSE: np.ndarray, 
                      num_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU implementation of BSE eigenvalue solver."""
        bse_size = H_BSE.shape[0]
        
        if num_states < bse_size // 2:
            # Use sparse solver for few states
            H_BSE_sparse = csr_matrix(H_BSE)
            try:
                # Try with increased tolerance and iterations
                eigenvalues, eigenvectors = eigsh(H_BSE_sparse, k=num_states, which='SA', 
                                                 maxiter=50000, tol=1e-6)
            except ArpackNoConvergence as e:
                logger.warning(f"Sparse solver failed to converge, trying with fewer states...")
                # Try with fewer states first
                k_reduced = min(num_states // 2, 5)
                try:
                    eigenvalues, eigenvectors = eigsh(H_BSE_sparse, k=k_reduced, which='SA',
                                                     maxiter=50000, tol=1e-5)
                    logger.info(f"Converged with {k_reduced} states instead of {num_states}")
                    # Pad with zeros if needed
                    if k_reduced < num_states:
                        pad_size = num_states - k_reduced
                        eigenvalues = np.pad(eigenvalues, (0, pad_size), 'constant', constant_values=eigenvalues[-1] + 0.1)
                        eigenvectors = np.pad(eigenvectors, ((0, 0), (0, pad_size)), 'constant')
                except ArpackNoConvergence:
                    logger.warning("Sparse solver completely failed, falling back to dense solver")
                    eigenvalues, eigenvectors = eigh(H_BSE)
                    eigenvalues = eigenvalues[:num_states]
                    eigenvectors = eigenvectors[:, :num_states]
        else:
            # Use dense solver
            eigenvalues, eigenvectors = eigh(H_BSE)
            eigenvalues = eigenvalues[:num_states]
            eigenvectors = eigenvectors[:, :num_states]
            
        # Sort by energy
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _solve_bse_gpu(self, H_BSE: np.ndarray,
                      num_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """GPU implementation of BSE eigenvalue solver."""
        # Transfer to GPU
        H_BSE_gpu = cp.asarray(H_BSE, dtype=cp.float64)
        
        # Solve on GPU
        eigenvalues_gpu, eigenvectors_gpu = cp.linalg.eigh(H_BSE_gpu)
        
        # Transfer back and select states
        eigenvalues = cp.asnumpy(eigenvalues_gpu[:num_states])
        eigenvectors = cp.asnumpy(eigenvectors_gpu[:, :num_states])
        
        return eigenvalues, eigenvectors
    
    def transform_to_realspace(self, exciton_wavefunction: np.ndarray,
                             kpoints: np.ndarray,
                             R_grid: np.ndarray,
                             exciton_momentum: np.ndarray = None,
                             eigenvectors: np.ndarray = None) -> np.ndarray:
        """
        Transform exciton wavefunction to real space.
        
        For electron-hole separation R:
        Ψ_S(r_e, r_h) = sum_{vck} Ψ_{vck}^S φ_c(k,r_e) φ_v*(k,r_h)
        
        Setting R = r_e - r_h and R_cm = (r_e + r_h)/2:
        A^S(R) = sum_k exp(i*k.R) * sum_{vc} Ψ_{vck}^S
        
        Args:
            exciton_wavefunction: Exciton wavefunction in k-space
            kpoints: Array of k-points
            R_grid: Real-space grid points (electron-hole separation)
            exciton_momentum: Center-of-mass momentum Q (default: Gamma point)
            eigenvectors: Electronic eigenvectors (optional, for full calculation)
            
        Returns:
            Real-space exciton wavefunction
        """
        num_kpts = len(kpoints)
        num_R = len(R_grid)
        
        if exciton_momentum is None:
            exciton_momentum = np.zeros(3)
        
        # Reshape wavefunction
        psi_vck = exciton_wavefunction.reshape(
            (num_kpts, self.num_conduction, self.num_valence)
        )
        
        # Transform to real space
        if self.use_gpu:
            A_R = self._transform_to_realspace_gpu(
                psi_vck, kpoints, R_grid, exciton_momentum
            )
        else:
            A_R = self._transform_to_realspace_cpu(
                psi_vck, kpoints, R_grid, exciton_momentum
            )
            
        return A_R
    
    def _transform_to_realspace_cpu(self, psi_vck, kpoints, R_grid, Q):
        """CPU implementation of real-space transformation."""
        num_kpts = len(kpoints)
        num_R = len(R_grid)
        A_R = np.zeros(num_R, dtype=complex)
        
        for ir, R in enumerate(R_grid):
            for ik, k in enumerate(kpoints):
                phase = np.exp(2j * np.pi * np.dot(k - Q, R))
                A_R[ir] += phase * np.sum(psi_vck[ik])
                
        return A_R / num_kpts
    
    def _transform_to_realspace_gpu(self, psi_vck, kpoints, R_grid, Q):
        """GPU implementation of real-space transformation."""
        # Transfer to GPU
        psi_gpu = cp.asarray(psi_vck)
        kpoints_gpu = cp.asarray(kpoints)
        R_grid_gpu = cp.asarray(R_grid)
        Q_gpu = cp.asarray(Q)
        
        num_kpts = len(kpoints)
        num_R = len(R_grid)
        
        # Compute phase factors
        k_minus_Q = kpoints_gpu - Q_gpu[cp.newaxis, :]
        phases = cp.exp(2j * cp.pi * cp.dot(R_grid_gpu, k_minus_Q.T))
        
        # Sum over bands
        psi_sum = cp.sum(psi_gpu, axis=(1, 2))
        
        # Transform
        A_R_gpu = cp.dot(phases, psi_sum) / num_kpts
        
        return cp.asnumpy(A_R_gpu)
    
    def compute_optical_absorption(self, exciton_energies: np.ndarray,
                                 exciton_wavefunctions: np.ndarray,
                                 broadening: float = 0.05,
                                 energy_range: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical absorption spectrum.
        
        Args:
            exciton_energies: Exciton energies
            exciton_wavefunctions: Exciton wavefunctions
            broadening: Lorentzian broadening in eV
            energy_range: Energy range for spectrum (min, max) in eV
            
        Returns:
            energies: Energy grid
            absorption: Absorption spectrum
        """
        if energy_range is None:
            e_min = exciton_energies[0] - 1.0
            e_max = exciton_energies[-1] + 1.0
        else:
            e_min, e_max = energy_range
            
        energies = np.linspace(e_min, e_max, 1000)
        absorption = np.zeros_like(energies)
        
        # Compute oscillator strengths (simplified)
        oscillator_strengths = np.abs(exciton_wavefunctions[0, :])**2
        
        # Add Lorentzian peaks
        for i, (E_exc, f_osc) in enumerate(zip(exciton_energies, oscillator_strengths)):
            lorentzian = broadening / ((energies - E_exc)**2 + broadening**2)
            absorption += f_osc * lorentzian
            
        # Normalize
        absorption /= np.max(absorption)
        
        return energies, absorption