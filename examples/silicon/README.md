# Silicon Example

This directory contains Wannier90 output files for bulk silicon, which can be used to test TBExciton90.

## Files

- `silicon_hr.dat` - Tight-binding Hamiltonian in Wannier basis
- `silicon_band.kpt` - k-points for band structure calculation
- `silicon_centres.xyz` - Wannier function centers
- `silicon_band.dat` - Band energies from Wannier90
- `silicon.win` - Original Wannier90 input file

## Usage

Run TBExciton90 on these files:

```bash
# From the examples/silicon directory
tbx90 compute --input silicon_hr.dat --kpoints silicon_band.kpt

# Or with Python
python ../basic_example.py
```

## Expected Results

- Band gap: ~1.1 eV (indirect gap)
- Exciton binding energy: ~15 meV
- Lowest bright exciton: ~1.085 eV

These are test calculations with a minimal k-point grid and should not be considered converged results.