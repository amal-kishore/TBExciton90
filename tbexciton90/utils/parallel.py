"""Parallel execution utilities for GPU and MPI."""

import numpy as np
import logging
from typing import Optional, Callable, Any, List
import os

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import cupy as cp
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


class ParallelManager:
    """Manager for parallel execution with GPU and MPI support."""
    
    def __init__(self, use_gpu: bool = False, use_mpi: bool = False):
        """
        Initialize parallel manager.
        
        Args:
            use_gpu: Whether to use GPU if available
            use_mpi: Whether to use MPI if available
        """
        self.use_gpu = use_gpu and HAS_CUPY
        self.use_mpi = use_mpi and HAS_MPI
        
        # GPU setup
        if self.use_gpu:
            self.gpu_id = self._setup_gpu()
            self.xp = cp
            logger.info(f"Using GPU device {self.gpu_id}")
        else:
            self.xp = np
            if use_gpu and not HAS_CUPY:
                logger.warning("GPU requested but CuPy not available")
                
        # MPI setup
        if self.use_mpi:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            logger.info(f"MPI initialized: rank {self.rank}/{self.size}")
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
            if use_mpi and not HAS_MPI:
                logger.warning("MPI requested but mpi4py not available")
                
    def _setup_gpu(self) -> int:
        """Setup GPU device."""
        # Check for CUDA_VISIBLE_DEVICES
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        if cuda_devices:
            # Use first available device
            device_id = 0
        else:
            # Use device based on MPI rank if using MPI
            if self.use_mpi:
                num_devices = cp.cuda.runtime.getDeviceCount()
                device_id = self.rank % num_devices
            else:
                device_id = 0
                
        cp.cuda.Device(device_id).use()
        return device_id
    
    def get_memory_info(self) -> dict:
        """Get memory information for current device."""
        info = {}
        
        if self.use_gpu:
            meminfo = cp.cuda.MemoryPool().get_limit()
            free_mem = cp.cuda.runtime.memGetInfo()[0]
            total_mem = cp.cuda.runtime.memGetInfo()[1]
            
            info['gpu'] = {
                'free_gb': free_mem / 1e9,
                'total_gb': total_mem / 1e9,
                'used_gb': (total_mem - free_mem) / 1e9,
                'device_id': self.gpu_id
            }
            
        # CPU memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            info['cpu'] = {
                'free_gb': mem.available / 1e9,
                'total_gb': mem.total / 1e9,
                'used_gb': mem.used / 1e9
            }
        except ImportError:
            pass
            
        return info
    
    def to_device(self, array: np.ndarray) -> Any:
        """Transfer array to appropriate device."""
        if self.use_gpu:
            return cp.asarray(array)
        return array
    
    def from_device(self, array: Any) -> np.ndarray:
        """Transfer array from device to CPU."""
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def alloc_array(self, shape: tuple, dtype: type = np.float64) -> Any:
        """Allocate array on appropriate device."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def distribute_work(self, total_items: int) -> tuple:
        """
        Distribute work items among MPI processes.
        
        Args:
            total_items: Total number of work items
            
        Returns:
            (start_idx, end_idx) for this process
        """
        if not self.use_mpi:
            return 0, total_items
            
        items_per_proc = total_items // self.size
        remainder = total_items % self.size
        
        if self.rank < remainder:
            start = self.rank * (items_per_proc + 1)
            end = start + items_per_proc + 1
        else:
            start = self.rank * items_per_proc + remainder
            end = start + items_per_proc
            
        return start, end
    
    def gather_results(self, local_data: Any, root: int = 0) -> Optional[List]:
        """
        Gather results from all MPI processes.
        
        Args:
            local_data: Data from this process
            root: Root process for gathering
            
        Returns:
            Gathered data (only on root process)
        """
        if not self.use_mpi:
            return [local_data]
            
        gathered = self.comm.gather(local_data, root=root)
        return gathered if self.rank == root else None
    
    def broadcast(self, data: Any, root: int = 0) -> Any:
        """
        Broadcast data from root to all processes.
        
        Args:
            data: Data to broadcast (only needed on root)
            root: Root process
            
        Returns:
            Broadcasted data
        """
        if not self.use_mpi:
            return data
            
        return self.comm.bcast(data, root=root)
    
    def reduce_sum(self, local_value: Any, root: int = 0) -> Optional[Any]:
        """
        Sum reduction across all processes.
        
        Args:
            local_value: Local value to sum
            root: Root process for result
            
        Returns:
            Sum of all values (only on root)
        """
        if not self.use_mpi:
            return local_value
            
        if self.use_gpu:
            # Convert to CPU for MPI
            local_value = self.from_device(local_value)
            
        result = self.comm.reduce(local_value, op=MPI.SUM, root=root)
        
        if self.rank == root and self.use_gpu:
            result = self.to_device(result)
            
        return result if self.rank == root else None
    
    def barrier(self) -> None:
        """Synchronization barrier."""
        if self.use_mpi:
            self.comm.Barrier()
            
    def parallel_for(self, func: Callable, items: List[Any], 
                    gather: bool = True) -> Optional[List]:
        """
        Execute function in parallel over items.
        
        Args:
            func: Function to execute
            items: List of items to process
            gather: Whether to gather results
            
        Returns:
            Results (gathered if requested)
        """
        # Distribute items
        start, end = self.distribute_work(len(items))
        local_items = items[start:end]
        
        # Process local items
        local_results = []
        for item in local_items:
            result = func(item)
            local_results.append(result)
            
        # Gather if requested
        if gather:
            return self.gather_results(local_results)
        return local_results
    
    def get_optimal_batch_size(self, array_shape: tuple, 
                              dtype: type = np.complex128) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            array_shape: Shape of arrays to process
            dtype: Data type of arrays
            
        Returns:
            Optimal batch size
        """
        # Estimate memory per item
        bytes_per_element = np.dtype(dtype).itemsize
        elements_per_item = np.prod(array_shape[1:]) if len(array_shape) > 1 else 1
        bytes_per_item = bytes_per_element * elements_per_item
        
        if self.use_gpu:
            # Get available GPU memory
            free_mem = cp.cuda.runtime.memGetInfo()[0]
            # Use 80% of available memory
            available_mem = 0.8 * free_mem
        else:
            # Assume 4GB available for CPU
            available_mem = 4e9
            
        # Calculate batch size
        batch_size = int(available_mem / bytes_per_item)
        
        # Ensure reasonable limits
        batch_size = max(1, min(batch_size, 1000))
        
        return batch_size
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.use_gpu:
            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            
        if self.use_mpi:
            self.barrier()