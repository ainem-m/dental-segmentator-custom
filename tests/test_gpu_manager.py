"""
Unit tests for GPU Manager functionality.
"""

import pytest
import tempfile
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.gpu_manager import (
    GPUManager,
    GPUInfo,
    DeviceType,
    get_gpu_manager,
    setup_gpu_environment
)


class TestGPUManager:
    """Test cases for GPUManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gpu_manager = GPUManager(memory_fraction=0.8, reserve_memory_mb=200.0)
    
    def test_gpu_manager_initialization(self):
        """Test GPU manager initialization."""
        assert self.gpu_manager is not None
        assert self.gpu_manager.memory_fraction == 0.8
        assert self.gpu_manager.reserve_memory_mb == 200.0
        assert hasattr(self.gpu_manager, 'pytorch_available')
        assert hasattr(self.gpu_manager, 'logger')
    
    def test_cuda_availability_check(self):
        """Test CUDA availability checking."""
        # This test will pass regardless of CUDA availability
        cuda_available = self.gpu_manager.is_cuda_available()
        assert isinstance(cuda_available, bool)
    
    def test_mps_availability_check(self):
        """Test MPS availability checking."""
        # This test will pass regardless of MPS availability
        mps_available = self.gpu_manager.is_mps_available()
        assert isinstance(mps_available, bool)
    
    def test_device_count(self):
        """Test getting device count."""
        device_count = self.gpu_manager.get_device_count()
        assert isinstance(device_count, int)
        assert device_count >= 0
    
    def test_gpu_info_collection(self):
        """Test GPU information collection."""
        gpu_infos = self.gpu_manager.get_gpu_info()
        assert isinstance(gpu_infos, list)
        
        for gpu_info in gpu_infos:
            assert isinstance(gpu_info, GPUInfo)
            assert gpu_info.device_id >= 0
            assert isinstance(gpu_info.name, str)
            assert gpu_info.memory_total >= 0
            assert gpu_info.memory_allocated >= 0
            assert gpu_info.memory_free >= 0
    
    def test_device_selection_cpu_only(self):
        """Test device selection when requesting CPU only."""
        selected_device = self.gpu_manager.select_best_device(preferred_device="cpu")
        assert selected_device == "cpu"
    
    def test_device_selection_auto(self):
        """Test automatic device selection."""
        selected_device = self.gpu_manager.select_best_device(preferred_device="auto")
        assert isinstance(selected_device, str)
        assert selected_device in ["cpu"] or selected_device.startswith("cuda:")
    
    def test_memory_info_cpu(self):
        """Test getting memory info for CPU (should return zeros)."""
        memory_info = self.gpu_manager.get_memory_info("cpu")
        assert isinstance(memory_info, dict)
        assert all(key in memory_info for key in ['total', 'allocated', 'cached', 'free'])
        
        # For CPU, GPU memory info should be zero
        if not self.gpu_manager.is_cuda_available():
            assert all(memory_info[key] == 0.0 for key in memory_info.keys())
    
    def test_device_context_manager_cpu(self):
        """Test device context manager with CPU."""
        with self.gpu_manager.device_context("cpu") as device:
            assert device == "cpu"
    
    def test_cache_clearing(self):
        """Test GPU cache clearing (should not raise errors)."""
        # This should not raise errors regardless of CUDA availability
        self.gpu_manager.clear_cache()
        self.gpu_manager.clear_cache("cpu")
    
    def test_memory_requirement_estimation(self):
        """Test memory requirement estimation."""
        input_shape = (1, 1, 128, 128, 64)
        estimated_memory = self.gpu_manager.estimate_memory_requirement(
            input_shape=input_shape,
            dtype_size=4,
            safety_factor=2.0
        )
        
        assert isinstance(estimated_memory, float)
        assert estimated_memory > 0
        
        # Check calculation: 1*1*128*128*64*4 bytes = 4MB * 2.0 = 8MB
        expected_memory = (128 * 128 * 64 * 4) / (1024 ** 2) * 2.0
        assert abs(estimated_memory - expected_memory) < 0.1
    
    def test_device_compatibility_validation(self):
        """Test device compatibility validation."""
        # CPU should always be compatible
        assert self.gpu_manager.validate_device_compatibility("cpu") is True
        
        # MPS compatibility depends on system
        mps_compatible = self.gpu_manager.validate_device_compatibility("mps")
        assert isinstance(mps_compatible, bool)
        
        # Invalid device strings should be incompatible
        assert self.gpu_manager.validate_device_compatibility("invalid") is False
        assert self.gpu_manager.validate_device_compatibility("cuda:999") is False
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        stats = self.gpu_manager.monitor_memory_usage()
        
        assert isinstance(stats, dict)
        assert 'timestamp' in stats
        assert 'system_memory' in stats
        assert 'gpu_memory' in stats
        
        # Check system memory info
        sys_mem = stats['system_memory']
        assert all(key in sys_mem for key in ['total', 'available', 'used', 'percent'])
        assert all(isinstance(sys_mem[key], (int, float)) for key in sys_mem.keys())
    
    def test_gpu_info_caching(self):
        """Test GPU info caching mechanism."""
        # Get info twice - second call should use cache
        start_time = time.time()
        gpu_infos1 = self.gpu_manager.get_gpu_info()
        mid_time = time.time()
        gpu_infos2 = self.gpu_manager.get_gpu_info()
        end_time = time.time()
        
        # Results should be identical (cached)
        assert len(gpu_infos1) == len(gpu_infos2)
        
        # Force refresh
        gpu_infos3 = self.gpu_manager.get_gpu_info(force_refresh=True)
        assert len(gpu_infos1) == len(gpu_infos3)


class TestGlobalGPUManager:
    """Test cases for global GPU manager functions."""
    
    def test_get_gpu_manager_singleton(self):
        """Test that get_gpu_manager returns singleton."""
        manager1 = get_gpu_manager()
        manager2 = get_gpu_manager()
        assert manager1 is manager2
    
    def test_setup_gpu_environment(self):
        """Test GPU environment setup."""
        # Test auto selection
        device1 = setup_gpu_environment("auto")
        assert isinstance(device1, str)
        assert device1 in ["cpu"] or device1.startswith("cuda:")
        
        # Test explicit CPU
        device2 = setup_gpu_environment("cpu")
        assert device2 == "cpu"


class TestGPUInfo:
    """Test cases for GPUInfo data structure."""
    
    def test_gpu_info_creation(self):
        """Test GPUInfo object creation."""
        gpu_info = GPUInfo(
            device_id=0,
            name="Test GPU",
            memory_total=8192.0,
            memory_allocated=2048.0,
            memory_free=6144.0,
            compute_capability=(7, 5),
            is_available=True
        )
        
        assert gpu_info.device_id == 0
        assert gpu_info.name == "Test GPU"
        assert gpu_info.memory_total == 8192.0
        assert gpu_info.memory_allocated == 2048.0
        assert gpu_info.memory_free == 6144.0
        assert gpu_info.compute_capability == (7, 5)
        assert gpu_info.is_available is True
    
    def test_gpu_info_repr(self):
        """Test GPUInfo string representation."""
        gpu_info = GPUInfo(
            device_id=0,
            name="Test GPU",
            memory_total=8192.0,
            memory_allocated=2048.0,
            memory_free=6144.0
        )
        
        repr_str = repr(gpu_info)
        assert "GPUInfo" in repr_str
        assert "id=0" in repr_str
        assert "name='Test GPU'" in repr_str
        assert "2048.0/8192.0MB" in repr_str


class TestDeviceType:
    """Test cases for DeviceType enum."""
    
    def test_device_type_enum(self):
        """Test DeviceType enum values."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.AUTO.value == "auto"


# Integration tests
class TestGPUManagerIntegration:
    """Integration tests for GPU manager with other components."""
    
    def test_gpu_manager_with_segmentator_integration(self):
        """Test GPU manager integration potential."""
        gpu_manager = get_gpu_manager()
        
        # Test typical segmentation workflow
        device = gpu_manager.select_best_device(required_memory_mb=1024.0)
        assert isinstance(device, str)
        
        # Test memory estimation for typical volume
        memory_needed = gpu_manager.estimate_memory_requirement(
            input_shape=(1, 1, 512, 512, 128),
            dtype_size=4,
            safety_factor=3.0
        )
        assert memory_needed > 0
        
        # Test device context
        with gpu_manager.device_context(device):
            memory_stats = gpu_manager.monitor_memory_usage()
            assert isinstance(memory_stats, dict)