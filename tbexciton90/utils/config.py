"""Configuration management for exciton calculations."""

import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for exciton calculations."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        # Default configuration
        self.config = {
            'input': {
                'hr_file': 'silicon_hr.dat',
                'kpt_file': 'silicon_band.kpt',
                'centres_file': 'silicon_centres.xyz',
                'win_file': None,
            },
            'model': {
                'num_valence': 4,
                'num_conduction': 4,
                'screening_type': 'constant',
                'screening_parameter': 0.1,
            },
            'solver': {
                'use_gpu': False,
                'use_mpi': False,
                'num_exciton_states': 10,
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
                'memory_limit_gb': None,
                'verbosity': 'info',
            }
        }
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
            
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(config_file):
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            return
            
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f)
            
        # Update configuration with user values
        self._update_nested_dict(self.config, user_config)
        logger.info(f"Loaded configuration from {config_file}")
        
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML file."""
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved configuration to {config_file}")
        
    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
                
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'solver.use_gpu')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config_dict = self.config
        
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
            
        config_dict[keys[-1]] = value
        
    def validate(self) -> bool:
        """Validate configuration."""
        # Check required input files
        required_files = ['hr_file', 'kpt_file']
        for file_key in required_files:
            file_path = self.get(f'input.{file_key}')
            if file_path and not os.path.exists(file_path):
                logger.error(f"Required file not found: {file_path}")
                return False
                
        # Check numerical parameters
        if self.get('model.num_valence', 0) <= 0:
            logger.error("Number of valence bands must be positive")
            return False
            
        if self.get('model.num_conduction', 0) <= 0:
            logger.error("Number of conduction bands must be positive")
            return False
            
        return True
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("\nCurrent Configuration:")
        print("=" * 60)
        self._print_nested_dict(self.config)
        print("=" * 60)
        
    def _print_nested_dict(self, d: Dict, indent: int = 0) -> None:
        """Recursively print nested dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_nested_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
                
    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        verbosity = self.get('advanced.verbosity', 'info')
        level_map = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
        }
        
        level = level_map.get(verbosity, logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def create_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        output_dir = self.get('output.output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")