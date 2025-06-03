#!/usr/bin/env python3
"""
Example using MPI parallelization with exciton-wannier90.

To run:
    mpirun -np 4 python mpi_example.py
"""

import numpy as np
from exciton_wannier90 import Wannier90Parser, TightBindingModel, BSESolver
from exciton_wannier90.utils import ParallelManager

# Initialize parallel manager with MPI
parallel = ParallelManager(use_mpi=True)

if parallel.rank == 0:
    print(f"Running with {parallel.size} MPI processes")

# Parse files (only on rank 0)
if parallel.rank == 0:
    parser = Wannier90Parser()
    parser.parse_hr_file('silicon_hr.dat')
    parser.parse_kpt_file('silicon_band.kpt')
else:
    parser = None

# Broadcast parser data to all ranks
parser = parallel.broadcast(parser)

# Create TB model
tb_model = TightBindingModel(parser)

# Distribute k-points among processes
start_k, end_k = parallel.distribute_work(len(parser.kpoints))
local_kpoints = parser.kpoints[start_k:end_k]

if parallel.rank == 0:
    print(f"Process {parallel.rank}: computing {len(local_kpoints)} k-points")

# Compute local bands
local_eigenvalues = []
for k in local_kpoints:
    H = tb_model.construct_hamiltonian(k)
    eigvals = np.linalg.eigvalsh(H)
    local_eigenvalues.append(eigvals)

# Gather results
all_eigenvalues = parallel.gather_results(local_eigenvalues)

if parallel.rank == 0:
    # Combine results
    eigenvalues = np.vstack([vals for proc_vals in all_eigenvalues for vals in proc_vals])
    
    # Analyze
    num_valence = 4
    bandgap = eigenvalues[:, num_valence].min() - eigenvalues[:, num_valence-1].max()
    print(f"\nBand gap: {bandgap:.3f} eV")
    
    # Setup BSE solver with MPI
    bse_solver = BSESolver(tb_model, use_mpi=True)
    
    # Note: Full BSE with MPI would distribute matrix construction
    # This is a simplified example

# Synchronize before exit
parallel.barrier()