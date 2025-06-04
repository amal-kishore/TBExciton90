#!/usr/bin/env python3
"""Command-line interface for exciton-wannier90."""

import click
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np

from .core.parser import Wannier90Parser
from .core.tb_model import TightBindingModel
from .solvers.bse_solver import BSESolver
from .solvers.optical_properties import OpticalProperties
from .utils.config import Config
from .utils.parallel import ParallelManager
from .visualization.plotter import ExcitonPlotter

logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def main():
    """Exciton-Wannier90: Compute exciton properties from Wannier90 outputs."""
    pass


@main.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file (YAML)')
@click.option('--input', '-i', type=click.Path(exists=True),
              help='Wannier90 HR file')
@click.option('--kpoints', '-k', type=click.Path(exists=True),
              help='K-points file')
@click.option('--centres', type=click.Path(exists=True),
              help='Wannier centres file')
@click.option('--gpu', is_flag=True, help='Use GPU acceleration')
@click.option('--mpi', is_flag=True, help='Use MPI parallelization')
@click.option('--output-dir', '-o', default='./results',
              help='Output directory')
@click.option('--num-valence', type=int, default=4,
              help='Number of valence bands')
@click.option('--num-conduction', type=int, default=4,
              help='Number of conduction bands')
@click.option('--num-excitons', type=int, default=10,
              help='Number of exciton states to compute')
@click.option('--screening', type=float, default=0.1,
              help='Screening parameter (eV)')
@click.option('--plot/--no-plot', default=True,
              help='Generate plots')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def compute(config, input, kpoints, centres, gpu, mpi, output_dir,
           num_valence, num_conduction, num_excitons, screening,
           plot, verbose):
    """Compute exciton properties from Wannier90 outputs."""
    
    # Setup configuration
    cfg = Config(config)
    
    # Override config with command-line options
    if input:
        cfg.set('input.hr_file', input)
    if kpoints:
        cfg.set('input.kpt_file', kpoints)
    if centres:
        cfg.set('input.centres_file', centres)
    cfg.set('solver.use_gpu', gpu)
    cfg.set('solver.use_mpi', mpi)
    cfg.set('output.output_dir', output_dir)
    cfg.set('model.num_valence', num_valence)
    cfg.set('model.num_conduction', num_conduction)
    cfg.set('solver.num_exciton_states', num_excitons)
    cfg.set('model.screening_parameter', screening)
    cfg.set('output.plot_results', plot)
    
    if verbose:
        cfg.set('advanced.verbosity', 'debug')
        
    # Setup logging
    cfg.setup_logging()
    
    # Validate configuration
    if not cfg.validate():
        click.echo("Configuration validation failed!", err=True)
        sys.exit(1)
        
    # Create output directory
    cfg.create_output_directory()
    
    # Initialize parallel manager
    parallel = ParallelManager(use_gpu=gpu, use_mpi=mpi)
    
    if parallel.rank == 0:
        click.echo("="*60)
        click.echo("Exciton-Wannier90 Calculation")
        click.echo("="*60)
        cfg.print_config()
        
        # Print memory info
        mem_info = parallel.get_memory_info()
        if 'gpu' in mem_info:
            click.echo(f"\nGPU Memory: {mem_info['gpu']['free_gb']:.1f}/{mem_info['gpu']['total_gb']:.1f} GB available")
        if 'cpu' in mem_info:
            click.echo(f"CPU Memory: {mem_info['cpu']['free_gb']:.1f}/{mem_info['cpu']['total_gb']:.1f} GB available")
    
    try:
        # Run calculation
        results = run_calculation(cfg, parallel)
        
        # Save results
        if parallel.rank == 0:
            save_results(results, cfg)
            
            # Generate plots
            if cfg.get('output.plot_results'):
                generate_plots(results, cfg)
                
            click.echo("\n" + "="*60)
            click.echo("Calculation completed successfully!")
            click.echo("="*60)
            
    except Exception as e:
        logger.error(f"Calculation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        parallel.cleanup()


def run_calculation(config: Config, parallel: ParallelManager) -> dict:
    """Run the main exciton calculation."""
    t_start = time.time()
    results = {}
    
    # Step 1: Parse Wannier90 files
    if parallel.rank == 0:
        click.echo("\n1. Parsing Wannier90 outputs...")
        
    parser = Wannier90Parser()
    parser.parse_hr_file(config.get('input.hr_file'))
    parser.parse_kpt_file(config.get('input.kpt_file'))
    
    centres_file = config.get('input.centres_file')
    if centres_file and os.path.exists(centres_file):
        parser.parse_centres_file(centres_file)
        
    win_file = config.get('input.win_file')
    if win_file and os.path.exists(win_file):
        parser.parse_win_file(win_file)
    
    # Step 2: Setup tight-binding model
    if parallel.rank == 0:
        click.echo("\n2. Constructing tight-binding model...")
        
    tb_model = TightBindingModel(parser, use_gpu=config.get('solver.use_gpu'))
    
    # Step 3: Compute band structure
    if parallel.rank == 0:
        click.echo("\n3. Computing electronic band structure...")
        
    t_bands = time.time()
    eigenvalues, eigenvectors = tb_model.compute_bands(parser.kpoints)
    
    if parallel.rank == 0:
        click.echo(f"   Band calculation took {time.time() - t_bands:.2f} seconds")
        
        # Analyze bands
        num_valence = config.get('model.num_valence')
        vbm = eigenvalues[:, num_valence-1].max()
        cbm = eigenvalues[:, num_valence].min()
        bandgap = cbm - vbm
        
        click.echo(f"\n   Electronic structure:")
        click.echo(f"   - VBM: {vbm:.3f} eV")
        click.echo(f"   - CBM: {cbm:.3f} eV")
        click.echo(f"   - Band gap: {bandgap:.3f} eV")
    
    # Step 4: Solve BSE
    if parallel.rank == 0:
        click.echo("\n4. Solving Bethe-Salpeter Equation...")
        
    bse_solver = BSESolver(
        tb_model,
        num_valence=config.get('model.num_valence'),
        num_conduction=config.get('model.num_conduction'),
        use_gpu=config.get('solver.use_gpu'),
        use_mpi=config.get('solver.use_mpi')
    )
    
    # Set screening
    bse_solver.screening_parameter = config.get('model.screening_parameter')
    
    t_bse = time.time()
    exciton_energies, exciton_wavefunctions = bse_solver.solve_bse(
        parser.kpoints, eigenvalues, eigenvectors,
        num_states=config.get('solver.num_exciton_states')
    )
    
    if parallel.rank == 0:
        click.echo(f"   BSE calculation took {time.time() - t_bse:.2f} seconds")
        
        # Analyze excitons
        exciton_binding = bandgap - exciton_energies[0]
        click.echo(f"\n   Exciton properties:")
        click.echo(f"   - Lowest exciton energy: {exciton_energies[0]:.3f} eV")
        click.echo(f"   - Exciton binding energy: {exciton_binding:.3f} eV")
        click.echo(f"   - Optical gap: {exciton_energies[0]:.3f} eV")
    
    # Step 5: Compute optical properties
    if parallel.rank == 0:
        click.echo("\n5. Computing optical properties...")
        
    # Initialize optical properties calculator
    optical = OpticalProperties(tb_model, bse_solver)
    
    # Compute oscillator strengths
    oscillator_strengths = optical.compute_oscillator_strengths(
        parser.kpoints, eigenvectors, exciton_wavefunctions
    )
    
    # Identify bright excitons
    bright_indices = optical.identify_bright_excitons(oscillator_strengths)
    if parallel.rank == 0:
        click.echo(f"   Found {len(bright_indices)} bright excitons")
        if len(bright_indices) > 0:
            for i, idx in enumerate(bright_indices[:3]):
                click.echo(f"   - S{i+1}: {exciton_energies[idx]:.3f} eV (f={oscillator_strengths[idx]:.3f})")
    
    # Compute absorption with e-h interaction
    absorption_energies, absorption = optical.compute_absorption_with_interaction(
        exciton_energies, oscillator_strengths
    )
    
    # Compute absorption without e-h interaction
    absorption_energies_no_int, absorption_no_int = optical.compute_absorption_without_interaction(
        eigenvalues, config.get('model.num_valence')
    )
    
    
    # Collect results
    results = {
        'kpoints': parser.kpoints,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'exciton_energies': exciton_energies,
        'exciton_wavefunctions': exciton_wavefunctions,
        'absorption_energies': absorption_energies,
        'absorption': absorption,
        'absorption_energies_no_int': absorption_energies_no_int,
        'absorption_no_int': absorption_no_int,
        'oscillator_strengths': oscillator_strengths,
        'bright_indices': bright_indices,
        'bandgap': bandgap if parallel.rank == 0 else None,
        'exciton_binding': exciton_binding if parallel.rank == 0 else None,
    }
    
    if parallel.rank == 0:
        click.echo(f"\nTotal calculation time: {time.time() - t_start:.2f} seconds")
    
    return results



def save_results(results: dict, config: Config):
    """Save calculation results."""
    import h5py
    import numpy as np
    
    output_dir = config.get('output.output_dir')
    
    # Save to HDF5
    output_file = os.path.join(output_dir, 'exciton_results.h5')
    with h5py.File(output_file, 'w') as f:
        # Save arrays
        for key in ['kpoints', 'eigenvalues', 'exciton_energies', 
                   'exciton_wavefunctions', 'absorption_energies', 'absorption']:
            if key in results and results[key] is not None:
                f.create_dataset(key, data=results[key])
                
        # Save scalars
        if results.get('bandgap') is not None:
            f.attrs['bandgap'] = results['bandgap']
        if results.get('exciton_binding') is not None:
            f.attrs['exciton_binding'] = results['exciton_binding']
            
    click.echo(f"\nResults saved to {output_file}")
    
    # Also save key results as text
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Exciton Calculation Summary\n")
        f.write("="*60 + "\n\n")
        
        if results.get('bandgap') is not None:
            f.write(f"Electronic band gap: {results['bandgap']:.3f} eV\n")
        if results.get('exciton_binding') is not None:
            f.write(f"Exciton binding energy: {results['exciton_binding']:.3f} eV\n")
        if 'exciton_energies' in results:
            f.write(f"Optical gap: {results['exciton_energies'][0]:.3f} eV\n\n")
            
            f.write("Lowest 10 exciton energies (eV):\n")
            for i, E in enumerate(results['exciton_energies'][:10]):
                f.write(f"  State {i}: {E:.4f}\n")
                
    click.echo(f"Summary saved to {summary_file}")


def generate_plots(results: dict, config: Config):
    """Generate visualization plots."""
    output_dir = config.get('output.output_dir')
    plotter = ExcitonPlotter(output_dir)
    
    click.echo("\nGenerating plots...")
    
    # Band structure
    if 'eigenvalues' in results:
        plotter.plot_band_structure(
            results['kpoints'], 
            results['eigenvalues'],
            config.get('model.num_valence')
        )
        click.echo("  - Band structure: band_structure.png")
    
    # Exciton spectrum with bright/dark distinction
    if 'exciton_energies' in results and 'oscillator_strengths' in results:
        plotter.plot_exciton_spectrum(
            results['exciton_energies'], 
            results['oscillator_strengths']
        )
        click.echo("  - Exciton spectrum: exciton_spectrum.png")
    
    # Absorption spectrum comparison
    if 'absorption' in results and 'absorption_no_int' in results:
        plotter.plot_optical_absorption_comparison(
            results['absorption_energies'],
            results['absorption'],
            results['absorption_energies_no_int'],
            results['absorption_no_int'],
            results.get('exciton_energies'),
            results.get('oscillator_strengths')
        )
        click.echo("  - Absorption comparison: optical_absorption_comparison.png")
    


@main.command()
@click.option('--example', type=click.Choice(['minimal', 'gpu', 'mpi', 'full']),
              default='minimal', help='Type of example configuration')
@click.option('--output', '-o', default='config.yaml',
              help='Output configuration file')
def generate_config(example, output):
    """Generate example configuration file."""
    
    examples = {
        'minimal': {
            'input': {
                'hr_file': 'silicon_hr.dat',
                'kpt_file': 'silicon_band.kpt',
            },
            'model': {
                'num_valence': 4,
                'num_conduction': 4,
            }
        },
        'gpu': {
            'input': {
                'hr_file': 'silicon_hr.dat',
                'kpt_file': 'silicon_band.kpt',
                'centres_file': 'silicon_centres.xyz',
            },
            'model': {
                'num_valence': 4,
                'num_conduction': 4,
                'screening_type': 'constant',
                'screening_parameter': 0.1,
            },
            'solver': {
                'use_gpu': True,
                'num_exciton_states': 50,
            },
            'output': {
                'save_bands': True,
                'save_excitons': True,
                'plot_results': True,
            }
        },
        'mpi': {
            'input': {
                'hr_file': 'silicon_hr.dat',
                'kpt_file': 'silicon_band.kpt',
            },
            'model': {
                'num_valence': 8,
                'num_conduction': 8,
            },
            'solver': {
                'use_mpi': True,
                'num_exciton_states': 100,
            }
        },
        'full': {
            'input': {
                'hr_file': 'silicon_hr.dat',
                'kpt_file': 'silicon_band.kpt',
                'centres_file': 'silicon_centres.xyz',
                'win_file': 'silicon.win',
            },
            'model': {
                'num_valence': 8,
                'num_conduction': 8,
                'screening_type': 'thomas-fermi',
                'screening_parameter': 0.1,
            },
            'solver': {
                'use_gpu': True,
                'use_mpi': True,
                'num_exciton_states': 200,
                'sparse_threshold': 1000,
            },
            'output': {
                'save_bands': True,
                'save_excitons': True,
                'save_wavefunctions': True,
                'plot_results': True,
                'output_dir': './results',
                'file_format': 'hdf5',
            },
            'advanced': {
                'gpu_batch_size': 100,
                'mpi_load_balance': 'auto',
                'verbosity': 'info',
            }
        }
    }
    
    config = Config()
    config.config = examples[example]
    config.save_to_file(output)
    
    click.echo(f"Generated {example} configuration: {output}")


@main.command()
def test():
    """Run basic tests to verify installation."""
    click.echo("Running installation tests...")
    
    # Test imports
    tests_passed = True
    
    try:
        import numpy
        click.echo("✓ NumPy imported successfully")
    except ImportError:
        click.echo("✗ NumPy import failed", err=True)
        tests_passed = False
        
    try:
        import scipy
        click.echo("✓ SciPy imported successfully")
    except ImportError:
        click.echo("✗ SciPy import failed", err=True)
        tests_passed = False
        
    try:
        import matplotlib
        click.echo("✓ Matplotlib imported successfully")
    except ImportError:
        click.echo("✗ Matplotlib import failed", err=True)
        tests_passed = False
        
    # Test optional dependencies
    try:
        import cupy
        click.echo("✓ CuPy (GPU support) available")
    except ImportError:
        click.echo("ℹ CuPy not available (GPU support disabled)")
        
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        click.echo(f"✓ MPI available (size={comm.Get_size()})")
    except ImportError:
        click.echo("ℹ mpi4py not available (MPI support disabled)")
        
    if tests_passed:
        click.echo("\nAll required dependencies are installed correctly!")
    else:
        click.echo("\nSome required dependencies are missing!", err=True)
        sys.exit(1)


@main.command()
@click.option('--results-dir', '-r', default='./results',
              help='Directory containing calculation results')
@click.option('--output-dir', '-o', default='./plots',
              help='Output directory for plots')
@click.option('--plot-type', '-t', 
              type=click.Choice(['bands', 'excitons', 'absorption', 'all']),
              default='all', help='Type of plot to generate')
def plot(results_dir, output_dir, plot_type):
    """Generate plots from existing calculation results."""
    import h5py
    from .visualization.plotter import ExcitonPlotter
    
    # Check if results exist
    results_file = os.path.join(results_dir, 'exciton_results.h5')
    if not os.path.exists(results_file):
        click.echo(f"Error: Results file not found at {results_file}", err=True)
        click.echo("Run 'tbx90 compute' first to generate results.", err=True)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize plotter (always high quality)
    plotter = ExcitonPlotter(output_dir=output_dir, style="publication")
    
    click.echo(f"Loading results from {results_file}...")
    
    # Load results
    with h5py.File(results_file, 'r') as f:
        results = {
            'kpoints': f['kpoints'][:],
            'eigenvalues': f['eigenvalues'][:],
            'exciton_energies': f['exciton_energies'][:],
            'oscillator_strengths': f['oscillator_strengths'][:],
            'num_valence': f.attrs['num_valence'],
            'num_conduction': f.attrs['num_conduction']
        }
        
        # Load optional data
        if 'absorption_energies' in f:
            results['absorption_energies'] = f['absorption_energies'][:]
            results['absorption_spectrum'] = f['absorption_spectrum'][:]
            results['absorption_no_interaction'] = f['absorption_no_interaction'][:]
        
        if 'exciton_wavefunctions' in f:
            results['exciton_wavefunctions'] = f['exciton_wavefunctions'][:]
    
    click.echo(f"Generating {plot_type} plots...")
    
    # Generate requested plots
    if plot_type in ['bands', 'all']:
        click.echo("  - Band structure")
        plotter.plot_band_structure(
            results['kpoints'], 
            results['eigenvalues'], 
            results['num_valence']
        )
    
    if plot_type in ['excitons', 'all']:
        click.echo("  - Exciton spectrum")
        plotter.plot_exciton_spectrum(
            results['exciton_energies'],
            results['oscillator_strengths']
        )
    
    if plot_type in ['absorption', 'all'] and 'absorption_energies' in results:
        click.echo("  - Optical absorption")
        plotter.plot_optical_absorption_comparison(
            results['absorption_energies'],
            results['absorption_spectrum'],
            results['absorption_no_interaction']
        )
    
    
    click.echo(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()