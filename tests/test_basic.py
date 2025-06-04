"""Basic tests for TBExciton90."""

import pytest
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    from tbexciton90 import Wannier90Parser, TightBindingModel, BSESolver
    from tbexciton90.solvers import OpticalProperties
    from tbexciton90.utils import Config, ParallelManager
    from tbexciton90.visualization import ExcitonPlotter, AdvancedExcitonPlotter
    assert True

def test_config():
    """Test configuration system."""
    from tbexciton90.utils import Config
    config = Config()
    assert config.get('model.screening_parameter') == 0.1
    assert config.get('output.output_dir') == './results'

def test_plot_style():
    """Test plotting style system."""
    from tbexciton90.visualization.plot_style import COLORS, PALETTES, set_publication_style
    assert 'primary' in COLORS
    assert 'default' in PALETTES
    # This should not raise an exception
    set_publication_style()

def test_parallel_manager():
    """Test parallel manager."""
    from tbexciton90.utils import ParallelManager
    pm = ParallelManager(use_gpu=False, use_mpi=False)
    assert pm.rank == 0
    assert pm.size == 1

if __name__ == "__main__":
    test_imports()
    test_config()
    test_plot_style()
    test_parallel_manager()
    print("✅ All basic tests passed!")