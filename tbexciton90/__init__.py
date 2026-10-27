"""TBExciton90 — Bethe-Salpeter solver in a Wannier / tight-binding basis.

Submodules
----------
potentials  screened e-h interaction kernels (RK, extended RK)
bse         BSE Hamiltonian construction, diagonalisation, absorption,
            finite-Q exciton dispersion
io          Wannier90 / QE readers, HDF5 output
analysis    2D Wannier-equation reference solver
backends    numpy / cupy backend switch
"""

__version__ = "0.2.0"
