"""Parser for Wannier90 output files."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Wannier90Parser:
    """Parser for Wannier90 output files with optimized data structures."""
    
    def __init__(self):
        self.hr_data = None
        self.kpoints = None
        self.centres = None
        self.num_wann = None
        self.num_kpts = None
        self.lattice_vectors = None
        self._hr_dict = None  # Optimized lookup structure
        
    def parse_hr_file(self, filename: str) -> None:
        """
        Parse the Hamiltonian in real space from _hr.dat file.
        
        Args:
            filename: Path to the _hr.dat file
        """
        logger.info(f"Parsing {filename}...")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        idx = 1  # Skip header comment
        
        # Read number of Wannier functions
        self.num_wann = int(lines[idx].strip())
        idx += 1
        
        # Read number of Wigner-Seitz grid points
        num_ws = int(lines[idx].strip())
        idx += 1
        
        # Read degeneracy of each Wigner-Seitz grid point
        degeneracy = []
        while len(degeneracy) < num_ws:
            degeneracy.extend(map(int, lines[idx].split()))
            idx += 1
        
        # Parse Hamiltonian elements
        hr_data = []
        self._hr_dict = {}  # For fast lookup
        
        for i in range(idx, len(lines)):
            parts = lines[i].split()
            if len(parts) == 7:
                R = tuple(map(int, parts[:3]))
                m = int(parts[3]) - 1  # Convert to 0-based
                n = int(parts[4]) - 1
                hr_real = float(parts[5])
                hr_imag = float(parts[6])
                
                value = hr_real + 1j * hr_imag
                deg = degeneracy[len(hr_data) // (self.num_wann * self.num_wann)]
                
                hr_data.append({
                    'R': np.array(R),
                    'm': m,
                    'n': n,
                    'value': value,
                    'degeneracy': deg
                })
                
                # Store in dictionary for fast lookup
                key = (R, m, n)
                self._hr_dict[key] = value / deg
        
        self.hr_data = hr_data
        logger.info(f"  Found {self.num_wann} Wannier functions")
        logger.info(f"  Parsed {len(hr_data)} hopping elements")
        
    def parse_kpt_file(self, filename: str) -> None:
        """
        Parse k-points from _band.kpt file.
        
        Args:
            filename: Path to the _band.kpt file
        """
        logger.info(f"Parsing {filename}...")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        self.num_kpts = int(lines[0].strip())
        kpoints = []
        
        for i in range(1, self.num_kpts + 1):
            parts = lines[i].split()
            kpt = [float(parts[0]), float(parts[1]), float(parts[2])]
            kpoints.append(kpt)
        
        self.kpoints = np.array(kpoints)
        logger.info(f"  Found {self.num_kpts} k-points")
        
    def parse_centres_file(self, filename: str) -> None:
        """
        Parse Wannier function centres from _centres.xyz file.
        
        Args:
            filename: Path to the _centres.xyz file
        """
        logger.info(f"Parsing {filename}...")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        centres = []
        for line in lines[2:]:  # Skip atom count and comment
            parts = line.split()
            if len(parts) >= 4:
                centres.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        self.centres = np.array(centres)
        logger.info(f"  Found {len(centres)} Wannier centres")
        
    def parse_win_file(self, filename: str) -> None:
        """
        Parse lattice vectors and other parameters from .win file.
        
        Args:
            filename: Path to the .win file
        """
        logger.info(f"Parsing {filename}...")
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find unit_cell_cart block
        in_cell_block = False
        lattice_vectors = []
        
        for line in lines:
            if "begin unit_cell_cart" in line.lower():
                in_cell_block = True
                continue
            elif "end unit_cell_cart" in line.lower():
                in_cell_block = False
                continue
                
            if in_cell_block:
                parts = line.split()
                if len(parts) >= 3:
                    lattice_vectors.append([float(x) for x in parts[:3]])
        
        if lattice_vectors:
            self.lattice_vectors = np.array(lattice_vectors)
            logger.info("  Parsed lattice vectors")
        
    def get_hr_element(self, R: Tuple[int, int, int], m: int, n: int) -> complex:
        """
        Get Hamiltonian matrix element H_{mn}(R) with fast lookup.
        
        Args:
            R: Real-space vector as tuple
            m, n: Orbital indices
            
        Returns:
            Complex hopping element
        """
        return self._hr_dict.get((R, m, n), 0.0 + 0.0j)
    
    def save_parsed_data(self, filename: str) -> None:
        """Save parsed data to HDF5 file for faster loading."""
        import h5py
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('num_wann', data=self.num_wann)
            f.create_dataset('num_kpts', data=self.num_kpts)
            f.create_dataset('kpoints', data=self.kpoints)
            
            if self.centres is not None:
                f.create_dataset('centres', data=self.centres)
            
            if self.lattice_vectors is not None:
                f.create_dataset('lattice_vectors', data=self.lattice_vectors)
            
            # Store HR data efficiently
            R_list = []
            mn_list = []
            values = []
            
            for key, value in self._hr_dict.items():
                R_list.append(key[0])
                mn_list.append([key[1], key[2]])
                values.append(value)
            
            f.create_dataset('hr_R', data=np.array(R_list))
            f.create_dataset('hr_mn', data=np.array(mn_list))
            f.create_dataset('hr_values', data=np.array(values))
    
    def load_parsed_data(self, filename: str) -> None:
        """Load parsed data from HDF5 file."""
        import h5py
        
        with h5py.File(filename, 'r') as f:
            self.num_wann = int(f['num_wann'][()])
            self.num_kpts = int(f['num_kpts'][()])
            self.kpoints = f['kpoints'][:]
            
            if 'centres' in f:
                self.centres = f['centres'][:]
                
            if 'lattice_vectors' in f:
                self.lattice_vectors = f['lattice_vectors'][:]
            
            # Reconstruct HR dictionary
            R_list = f['hr_R'][:]
            mn_list = f['hr_mn'][:]
            values = f['hr_values'][:]
            
            self._hr_dict = {}
            for i in range(len(values)):
                key = (tuple(R_list[i]), mn_list[i, 0], mn_list[i, 1])
                self._hr_dict[key] = values[i]