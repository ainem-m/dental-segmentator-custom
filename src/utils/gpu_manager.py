"""
GPU Manager for Dental Segmentator Application.

This module provides comprehensive GPU management functionality including:
- CUDA availability detection
- GPU memory management
- Device selection and switching
- Memory usage monitoring
- GPU resource allocation
"""

import os
import logging
import psutil
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from contextlib import contextmanager

# Try to import PyTorch with proper error handling
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

from .logging_manager import get_logging_manager


class DeviceType(Enum):
    """Device type enumeration."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class GPUInfo:
    """
    Data structure for GPU information.
    
    Attributes:
        device_id: GPU device ID
        name: GPU name
        memory_total: Total memory in MB
        memory_allocated: Currently allocated memory in MB
        memory_free: Free memory in MB
        compute_capability: Compute capability version
        is_available: Whether GPU is available for use
    """
    
    def __init__(
        self,
        device_id: int,
        name: str,
        memory_total: float,
        memory_allocated: float,
        memory_free: float,
        compute_capability: Optional[Tuple[int, int]] = None,
        is_available: bool = True
    ):
        self.device_id = device_id
        self.name = name
        self.memory_total = memory_total
        self.memory_allocated = memory_allocated
        self.memory_free = memory_free
        self.compute_capability = compute_capability
        self.is_available = is_available
    
    def __repr__(self) -> str:
        return (f"GPUInfo(id={self.device_id}, name='{self.name}', "
                f"memory={self.memory_allocated}/{self.memory_total}MB)")


class GPUManager:
    """
    GPU resource manager for the dental segmentator application.
    
    Provides functionality for:
    - GPU detection and enumeration
    - Memory management and monitoring
    - Device selection based on availability
    - Resource allocation and cleanup
    """
    
    def __init__(self, memory_fraction: float = 0.8, reserve_memory_mb: float = 500.0):
        """
        Initialize GPU manager.
        
        Args:
            memory_fraction: Maximum fraction of GPU memory to use (0.0 to 1.0)
            reserve_memory_mb: Memory to reserve for system (MB)
        """
        self.memory_fraction = max(0.1, min(1.0, memory_fraction))
        self.reserve_memory_mb = reserve_memory_mb
        self.logger = get_logging_manager().get_logger(
            "utils.gpu_manager",
            component="gpu_manager"
        )
        
        # Initialize PyTorch if available
        self.pytorch_available = PYTORCH_AVAILABLE
        if not self.pytorch_available:
            self.logger.warning("PyTorch not available, GPU functionality disabled")
        
        # Cache GPU information
        self._gpu_info_cache = None
        self._last_cache_update = 0
        self._cache_ttl = 30  # Cache TTL in seconds
        
        self.logger.info(f"GPU Manager initialized (PyTorch available: {self.pytorch_available})")
    
    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available
        """
        if not self.pytorch_available:
            return False
        return torch.cuda.is_available()
    
    def is_mps_available(self) -> bool:
        """
        Check if MPS (Metal Performance Shaders) is available.
        
        Returns:
            True if MPS is available (macOS with Apple Silicon)
        """
        if not self.pytorch_available:
            return False
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    def get_device_count(self) -> int:
        """
        Get number of available CUDA devices.
        
        Returns:
            Number of CUDA devices (0 if CUDA not available)
        """
        if not self.is_cuda_available():
            return 0
        return torch.cuda.device_count()
    
    def get_gpu_info(self, device_id: Optional[int] = None, force_refresh: bool = False) -> List[GPUInfo]:
        """
        Get detailed information about available GPUs.
        
        Args:
            device_id: Specific device ID to query (None for all devices)
            force_refresh: Force refresh of cached information
            
        Returns:
            List of GPUInfo objects
        """
        import time
        
        # Check cache validity
        current_time = time.time()
        if (not force_refresh and 
            self._gpu_info_cache is not None and 
            (current_time - self._last_cache_update) < self._cache_ttl):
            gpu_infos = self._gpu_info_cache
        else:
            # Refresh cache
            gpu_infos = self._collect_gpu_info()
            self._gpu_info_cache = gpu_infos
            self._last_cache_update = current_time
        
        # Filter by device_id if specified
        if device_id is not None:
            gpu_infos = [info for info in gpu_infos if info.device_id == device_id]
        
        return gpu_infos
    
    def _collect_gpu_info(self) -> List[GPUInfo]:
        """Collect GPU information from the system."""
        gpu_infos = []
        
        if not self.is_cuda_available():
            return gpu_infos
        
        try:
            for device_id in range(self.get_device_count()):
                # Get device properties
                device_props = torch.cuda.get_device_properties(device_id)
                name = device_props.name
                memory_total = device_props.total_memory / (1024 ** 2)  # Convert to MB
                compute_capability = (device_props.major, device_props.minor)
                
                # Get memory usage
                torch.cuda.set_device(device_id)
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                memory_free = memory_total - memory_allocated
                
                gpu_info = GPUInfo(
                    device_id=device_id,
                    name=name,
                    memory_total=memory_total,
                    memory_allocated=memory_allocated,
                    memory_free=memory_free,
                    compute_capability=compute_capability,
                    is_available=True
                )
                
                gpu_infos.append(gpu_info)
                
        except Exception as e:
            self.logger.error(f"Failed to collect GPU information: {e}")
        
        return gpu_infos
    
    def select_best_device(
        self, 
        required_memory_mb: Optional[float] = None,
        preferred_device: Optional[str] = None
    ) -> str:
        """
        Select the best available device for processing.
        
        Args:
            required_memory_mb: Minimum required memory in MB
            preferred_device: Preferred device ('cpu', 'cuda', 'mps', 'auto')
            
        Returns:
            Selected device string ('cpu', 'cuda:X', or 'mps')
        """
        # Handle explicit CPU request
        if preferred_device == "cpu":
            self.logger.info("CPU explicitly requested")
            return "cpu"
        
        # Handle explicit MPS request
        if preferred_device == "mps":
            if self.is_mps_available():
                self.logger.info("MPS explicitly requested and available")
                return "mps"
            else:
                self.logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
        
        # Handle explicit CUDA request
        if preferred_device and preferred_device.startswith("cuda"):
            if self.is_cuda_available():
                self.logger.info(f"CUDA device {preferred_device} explicitly requested")
                return preferred_device
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        
        # Auto selection logic
        if preferred_device == "auto" or preferred_device is None:
            # Priority: CUDA > MPS > CPU
            if self.is_cuda_available():
                # Select best CUDA device
                gpu_infos = self.get_gpu_info()
                if gpu_infos:
                    # Filter GPUs by memory requirement
                    suitable_gpus = []
                    for gpu_info in gpu_infos:
                        available_memory = gpu_info.memory_free - self.reserve_memory_mb
                        max_usable_memory = gpu_info.memory_total * self.memory_fraction
                        usable_memory = min(available_memory, max_usable_memory)
                        
                        if required_memory_mb is None or usable_memory >= required_memory_mb:
                            suitable_gpus.append((gpu_info, usable_memory))
                    
                    if suitable_gpus:
                        # Select GPU with most free memory
                        best_gpu, best_memory = max(suitable_gpus, key=lambda x: x[1])
                        device_string = f"cuda:{best_gpu.device_id}"
                        self.logger.info(f"Selected CUDA device: {device_string} "
                                        f"({best_gpu.name}, {best_memory:.1f}MB usable)")
                        return device_string
            
            # Fall back to MPS if CUDA not available/suitable
            if self.is_mps_available():
                self.logger.info("Selected MPS device (Apple Silicon GPU)")
                return "mps"
            
            # Final fallback to CPU
            self.logger.info("No suitable GPU found, using CPU")
            return "cpu"
        
        # Default fallback
        self.logger.info("Using CPU as default device")
        return "cpu"
    
    def get_memory_info(self, device: Optional[str] = None) -> Dict[str, float]:
        """
        Get memory information for a specific device.
        
        Args:
            device: Device string (e.g., 'cuda:0', 'mps') or None for current device
            
        Returns:
            Dictionary with memory information in MB
        """
        if device == "mps":
            # MPS memory information is limited in PyTorch
            if self.is_mps_available():
                try:
                    # MPS doesn't provide detailed memory statistics like CUDA
                    # Return estimated values based on system memory
                    import psutil
                    system_memory = psutil.virtual_memory()
                    
                    # Estimate MPS uses a portion of system memory
                    estimated_total = system_memory.total / (1024 ** 2) * 0.7  # Assume 70% available for MPS
                    estimated_free = system_memory.available / (1024 ** 2) * 0.7
                    
                    return {
                        'total': estimated_total,
                        'allocated': 0.0,  # Not available for MPS
                        'cached': 0.0,     # Not available for MPS
                        'free': estimated_free
                    }
                except Exception as e:
                    self.logger.error(f"Failed to get MPS memory info: {e}")
            
            return {
                'total': 0.0,
                'allocated': 0.0,
                'cached': 0.0,
                'free': 0.0
            }
        
        if not self.is_cuda_available() or device == "cpu":
            return {
                'total': 0.0,
                'allocated': 0.0,
                'cached': 0.0,
                'free': 0.0
            }
        
        try:
            if device is not None and device.startswith("cuda"):
                device_id = int(device.split(':')[1]) if ':' in device else 0
            else:
                device_id = torch.cuda.current_device()
            
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
            memory_cached = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
            
            # Get total memory from device properties
            device_props = torch.cuda.get_device_properties(device_id)
            memory_total = device_props.total_memory / (1024 ** 2)
            memory_free = memory_total - memory_cached
            
            return {
                'total': memory_total,
                'allocated': memory_allocated,
                'cached': memory_cached,
                'free': memory_free
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory info for device {device}: {e}")
            return {'total': 0.0, 'allocated': 0.0, 'cached': 0.0, 'free': 0.0}
    
    @contextmanager
    def device_context(self, device: str):
        """
        Context manager for device operations.
        
        Args:
            device: Device string to set as current
            
        Yields:
            The device string
        """
        if device == "cpu":
            yield device
            return
        
        if device == "mps":
            # MPS doesn't need explicit device setting like CUDA
            if self.is_mps_available():
                self.logger.debug(f"Using MPS device")
                yield device
            else:
                self.logger.warning("MPS not available, falling back to CPU")
                yield "cpu"
            return
        
        if not self.is_cuda_available() or not device.startswith("cuda"):
            yield device
            return
        
        try:
            # Parse device ID for CUDA
            device_id = int(device.split(':')[1]) if ':' in device else 0
            
            # Store current device
            current_device = torch.cuda.current_device()
            
            # Set new device
            torch.cuda.set_device(device_id)
            self.logger.debug(f"Set device to {device}")
            
            yield device
            
        finally:
            # Restore original device
            if self.is_cuda_available() and 'current_device' in locals():
                torch.cuda.set_device(current_device)
                self.logger.debug(f"Restored device to cuda:{current_device}")
    
    def clear_cache(self, device: Optional[str] = None):
        """
        Clear GPU memory cache.
        
        Args:
            device: Device to clear cache for (None for all devices)
        """
        try:
            if device is None:
                # Clear cache for all available devices
                if self.is_cuda_available():
                    torch.cuda.empty_cache()
                    self.logger.info("Cleared CUDA memory cache for all devices")
                
                if self.is_mps_available():
                    # MPS cache clearing (if available)
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                        self.logger.info("Cleared MPS memory cache")
                
            elif device == "mps":
                if self.is_mps_available() and hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                    self.logger.info(f"Cleared MPS memory cache")
                    
            elif device.startswith("cuda") and self.is_cuda_available():
                with self.device_context(device):
                    torch.cuda.empty_cache()
                self.logger.info(f"Cleared CUDA memory cache for {device}")
                    
        except Exception as e:
            self.logger.error(f"Failed to clear GPU cache for {device}: {e}")
    
    def estimate_memory_requirement(
        self, 
        input_shape: Tuple[int, ...], 
        dtype_size: int = 4,
        safety_factor: float = 2.0
    ) -> float:
        """
        Estimate memory requirement for a given input shape.
        
        Args:
            input_shape: Shape of input tensor
            dtype_size: Size of data type in bytes (4 for float32)
            safety_factor: Safety factor for memory estimation
            
        Returns:
            Estimated memory requirement in MB
        """
        # Calculate input memory
        input_elements = 1
        for dim in input_shape:
            input_elements *= dim
        input_memory_mb = (input_elements * dtype_size) / (1024 ** 2)
        
        # Apply safety factor to account for intermediate results, gradients, etc.
        estimated_memory_mb = input_memory_mb * safety_factor
        
        self.logger.debug(f"Estimated memory requirement: {estimated_memory_mb:.1f}MB "
                         f"for input shape {input_shape}")
        
        return estimated_memory_mb
    
    def monitor_memory_usage(self) -> Dict[str, Any]:
        """
        Monitor current memory usage across all devices.
        
        Returns:
            Dictionary with memory usage statistics
        """
        stats = {
            'timestamp': self._get_timestamp(),
            'system_memory': self._get_system_memory_info(),
            'gpu_memory': {}
        }
        
        if self.is_cuda_available():
            for device_id in range(self.get_device_count()):
                device = f"cuda:{device_id}"
                stats['gpu_memory'][device] = self.get_memory_info(device)
        
        return stats
    
    def _get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total / (1024 ** 2),  # MB
                'available': memory.available / (1024 ** 2),
                'used': memory.used / (1024 ** 2),
                'percent': memory.percent
            }
        except Exception as e:
            self.logger.error(f"Failed to get system memory info: {e}")
            return {'total': 0.0, 'available': 0.0, 'used': 0.0, 'percent': 0.0}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_device_compatibility(self, device: str) -> bool:
        """
        Validate if a device is compatible with the current system.
        
        Args:
            device: Device string to validate
            
        Returns:
            True if device is compatible
        """
        if device == "cpu":
            return True
        
        if device == "mps":
            return self.is_mps_available()
        
        if device.startswith("cuda"):
            if not self.is_cuda_available():
                return False
            try:
                device_id = int(device.split(':')[1]) if ':' in device else 0
                return 0 <= device_id < self.get_device_count()
            except (ValueError, IndexError):
                return False
        
        return False


# Global GPU manager instance
_gpu_manager = None


def get_gpu_manager(
    memory_fraction: float = 0.8, 
    reserve_memory_mb: float = 500.0
) -> GPUManager:
    """
    Get or create the global GPU manager instance.
    
    Args:
        memory_fraction: Maximum fraction of GPU memory to use
        reserve_memory_mb: Memory to reserve for system (MB)
        
    Returns:
        GPUManager instance
    """
    global _gpu_manager
    
    if _gpu_manager is None:
        _gpu_manager = GPUManager(
            memory_fraction=memory_fraction,
            reserve_memory_mb=reserve_memory_mb
        )
    
    return _gpu_manager


def setup_gpu_environment(device: str = "auto") -> str:
    """
    Setup GPU environment for the application.
    
    Args:
        device: Device preference ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Selected device string
    """
    gpu_manager = get_gpu_manager()
    
    if device == "auto":
        selected_device = gpu_manager.select_best_device()
    else:
        selected_device = device
        if not gpu_manager.validate_device_compatibility(device):
            gpu_manager.logger.warning(f"Device {device} not compatible, falling back to CPU")
            selected_device = "cpu"
    
    # Set environment variables for optimal performance
    if selected_device != "cpu":
        if selected_device.startswith("cuda") and gpu_manager.is_cuda_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_device.split(":")[1] if ":" in selected_device else "0"
            
            # Set cuDNN benchmark for consistent input sizes
            if PYTORCH_AVAILABLE:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
        elif selected_device == "mps" and gpu_manager.is_mps_available():
            # MPS-specific optimizations
            if PYTORCH_AVAILABLE:
                # Set MPS allocator settings if available
                try:
                    if hasattr(torch.backends.mps, 'set_allocator_strategy'):
                        torch.backends.mps.set_allocator_strategy('large_pool')
                except Exception:
                    # Ignore if not available
                    pass
    
    gpu_manager.logger.info(f"GPU environment setup complete, using device: {selected_device}")
    return selected_device