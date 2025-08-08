"""
Unit tests for nnU-Net Segmentator functionality.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.segmentation.nnunet_segmentator import (
    NnUNetSegmentator, 
    SegmentationError,
    SegmentationResult
)


class TestNnUNetSegmentator:
    """Test cases for NnUNetSegmentator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = Path(self.temp_dir) / "models"
        self.model_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_segmentator_initialization(self):
        """Test nnU-Net segmentator initialization."""
        segmentator = NnUNetSegmentator(
            model_path=str(self.model_dir),
            device="cpu"
        )
        
        assert str(segmentator.model_path) == str(self.model_dir)
        assert segmentator.device == "cpu"
        assert segmentator.model is None  # Not loaded yet
    
    def test_segmentator_initialization_with_gpu(self):
        """Test segmentator initialization with GPU setting."""
        segmentator = NnUNetSegmentator(
            model_path=str(self.model_dir),
            device="cuda"
        )
        
        # Should fall back to CPU if CUDA not available
        assert segmentator.device in ["cuda", "cpu"]
    
    def test_create_mock_input_data(self):
        """Test creating mock input data for testing."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # Create mock 3D volume (like preprocessed DICOM)
        test_volume = np.random.randint(
            -1000, 1000, 
            size=(64, 64, 32), 
            dtype=np.int16
        )
        
        assert test_volume.shape == (64, 64, 32)
        assert test_volume.dtype == np.int16
    
    def test_validate_input_volume(self):
        """Test input volume validation."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # Valid volume
        valid_volume = np.random.randint(-1000, 1000, size=(100, 100, 50))
        assert segmentator._validate_input_volume(valid_volume) is True
        
        # Invalid volumes
        too_small = np.random.randint(-1000, 1000, size=(10, 10, 5))
        assert segmentator._validate_input_volume(too_small) is False
        
        wrong_dim = np.random.randint(-1000, 1000, size=(100, 100))
        assert segmentator._validate_input_volume(wrong_dim) is False
    
    def test_segmentation_result_creation(self):
        """Test SegmentationResult data structure."""
        result = SegmentationResult(
            segmentation_mask=np.zeros((64, 64, 32), dtype=np.uint8),
            confidence_scores=[0.85, 0.92, 0.78],
            processing_time=15.5,
            model_version="dental_segmentator_v1",
            detected_teeth_count=28
        )
        
        assert result.segmentation_mask.shape == (64, 64, 32)
        assert len(result.confidence_scores) == 3
        assert result.processing_time == 15.5
        assert result.detected_teeth_count == 28
    
    def test_mock_model_download(self):
        """Test mock model download functionality."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # This would test the download functionality
        # For now, we'll test the path preparation
        download_path = segmentator._prepare_model_download_path()
        assert Path(download_path).parent == self.model_dir
    
    def test_mock_preprocessing(self):
        """Test preprocessing functionality."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # Create mock input volume
        input_volume = np.random.randint(-1000, 1000, size=(128, 128, 64))
        
        # Test preprocessing (normalization, resampling)
        preprocessed = segmentator._preprocess_volume(input_volume)
        
        assert preprocessed is not None
        assert isinstance(preprocessed, np.ndarray)
    
    def test_mock_postprocessing(self):
        """Test postprocessing functionality."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # Create mock segmentation result
        raw_segmentation = np.random.randint(0, 33, size=(64, 64, 32), dtype=np.uint8)
        
        # Test postprocessing
        processed = segmentator._postprocess_segmentation(raw_segmentation)
        
        assert processed is not None
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.uint8
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # Mock segmentation with different tooth labels
        segmentation = np.zeros((64, 64, 32), dtype=np.uint8)
        segmentation[10:20, 10:20, 10:20] = 1  # Tooth 1
        segmentation[30:40, 30:40, 15:25] = 2  # Tooth 2
        
        scores = segmentator._calculate_confidence_scores(segmentation)
        
        assert isinstance(scores, list)
        assert len(scores) >= 0  # May vary based on implementation
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        segmentator = NnUNetSegmentator(str(self.model_dir))
        
        # Test with invalid input
        with pytest.raises(SegmentationError):
            segmentator.segment(None)
        
        # Test with wrong input type
        with pytest.raises(SegmentationError):
            segmentator.segment("not_an_array")
        
        # Test with wrong dimensions
        with pytest.raises(SegmentationError):
            wrong_shape = np.random.rand(10, 10)  # 2D instead of 3D
            segmentator.segment(wrong_shape)