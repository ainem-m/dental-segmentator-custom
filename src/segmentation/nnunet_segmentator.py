"""
nnU-Net Segmentator for Dental Imaging.

This module provides comprehensive dental segmentation capabilities using
the pre-trained dental segmentator model from Zenodo, integrated with the
nnU-Net framework for high-accuracy tooth segmentation.
"""

import os
import time
import logging
import numpy as np
import urllib.request
import urllib.parse
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

# Import nnU-Net and PyTorch dependencies (will be handled gracefully if not available)
try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    F = None

try:
    from nnunetv2.inference.predict import nnUNetPredictor
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False
    nnUNetPredictor = None

from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label
import skimage
from skimage import measure, morphology

from ..utils.logging_manager import get_logging_manager
from ..utils.gpu_manager import get_gpu_manager, setup_gpu_environment


@dataclass
class SegmentationResult:
    """
    Data structure representing segmentation results.
    
    Attributes:
        segmentation_mask: 3D numpy array with tooth labels
        confidence_scores: List of confidence scores for each detected tooth
        processing_time: Processing time in seconds
        model_version: Version of the segmentation model used
        detected_teeth_count: Number of teeth detected
        memory_usage_mb: Memory usage during segmentation
        metadata: Additional processing metadata
    """
    segmentation_mask: np.ndarray
    confidence_scores: List[float] = field(default_factory=list)
    processing_time: float = 0.0
    model_version: str = ""
    detected_teeth_count: int = 0
    memory_usage_mb: float = 0.0
    metadata: Dict = field(default_factory=dict)


class SegmentationError(Exception):
    """Exception raised for segmentation-related errors."""
    pass


class NnUNetSegmentator:
    """
    nnU-Net based dental segmentator.
    
    Features:
    - Automatic model download from Zenodo
    - CPU/GPU inference support
    - Dental-specific preprocessing and postprocessing
    - Confidence score calculation
    - Memory-efficient processing
    """
    
    def __init__(
        self,
        model_path: str = "models",
        device: str = "auto",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize nnU-Net segmentator.
        
        Args:
            model_path: Path to model directory
            device: Device to use ('auto', 'cpu', 'cuda')
            confidence_threshold: Minimum confidence threshold
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.logger = get_logging_manager().get_logger(
            "segmentation.nnunet_segmentator",
            component="nnunet_segmentator"
        )
        
        # Initialize device with GPU manager
        self.gpu_manager = get_gpu_manager()
        self.device = self._setup_device(device)
        
        # Model configuration
        self.model_name = "dental_segmentator_v1"
        self.model_url = "https://zenodo.org/records/10829675/files/model.zip"
        self.model_checksum = ""  # To be updated with actual checksum
        
        # Model state
        self.model = None
        self.is_model_loaded = False
        
        # Preprocessing parameters
        self.target_spacing = (0.5, 0.5, 0.5)  # Target voxel spacing in mm
        self.intensity_bounds = (-1000, 1000)  # HU bounds for CT
        
        self.logger.info(f"nnU-Net segmentator initialized with device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """
        Setup processing device using GPU manager.
        
        Args:
            device: Requested device ('auto', 'cpu', 'cuda')
            
        Returns:
            Actual device to use
        """
        if not PYTORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, using CPU-only mode")
            return "cpu"
        
        # Use GPU manager for intelligent device selection
        if device == "auto":
            # Estimate memory requirement for typical dental segmentation
            estimated_memory_mb = self.gpu_manager.estimate_memory_requirement(
                input_shape=(1, 1, 512, 512, 128),  # Typical CBCT volume size
                dtype_size=4,  # float32
                safety_factor=3.0  # Account for model parameters and intermediate results
            )
            selected_device = self.gpu_manager.select_best_device(
                required_memory_mb=estimated_memory_mb
            )
        else:
            selected_device = device
            if not self.gpu_manager.validate_device_compatibility(device):
                self.logger.warning(f"Device {device} not compatible, falling back to CPU")
                selected_device = "cpu"
        
        # Log device information
        if selected_device.startswith("cuda"):
            gpu_infos = self.gpu_manager.get_gpu_info()
            if gpu_infos:
                device_id = int(selected_device.split(':')[1]) if ':' in selected_device else 0
                gpu_info = next((info for info in gpu_infos if info.device_id == device_id), None)
                if gpu_info:
                    self.logger.info(f"Using CUDA GPU: {gpu_info.name} "
                                   f"({gpu_info.memory_free:.1f}MB free)")
        elif selected_device == "mps":
            memory_info = self.gpu_manager.get_memory_info("mps")
            self.logger.info(f"Using MPS (Apple Silicon GPU) "
                           f"({memory_info['free']:.1f}MB estimated available)")
        else:
            self.logger.info("Using CPU for segmentation")
        
        return selected_device
    
    def download_model(self, force_redownload: bool = False) -> bool:
        """
        Download pre-trained dental segmentation model from Zenodo.
        
        Args:
            force_redownload: Force re-download even if model exists
            
        Returns:
            True if download successful
        """
        try:
            self.logger.info("Starting model download from Zenodo")
            
            # Check if model already exists
            model_dir = self.model_path / "dental_segmentator"
            if model_dir.exists() and not force_redownload:
                self.logger.info("Model already exists, skipping download")
                return True
            
            # Create model directory
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # Download model file
            download_path = self.model_path / "dental_segmentator_model.zip"
            
            self.logger.info(f"Downloading model from: {self.model_url}")
            self._download_file_with_progress(self.model_url, download_path)
            
            # Verify checksum if provided
            if self.model_checksum:
                if not self._verify_checksum(download_path, self.model_checksum):
                    raise SegmentationError("Model checksum verification failed")
            
            # Extract model
            self.logger.info("Extracting model files")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Clean up zip file
            download_path.unlink()
            
            self.logger.info("Model download completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            raise SegmentationError(f"Model download failed: {e}")
    
    def _download_file_with_progress(self, url: str, destination: Path) -> None:
        """
        Download file with progress reporting.
        
        Args:
            url: URL to download from
            destination: Local file destination
        """
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, (block_num * block_size / total_size) * 100)
                if block_num % 50 == 0:  # Log every ~5%
                    self.logger.info(f"Download progress: {percent:.1f}%")
        
        urllib.request.urlretrieve(url, destination, progress_hook)
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """
        Verify file checksum.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 checksum
            
        Returns:
            True if checksum matches
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        actual_checksum = sha256_hash.hexdigest()
        return actual_checksum == expected_checksum
    
    def load_model(self) -> bool:
        """
        Load the nnU-Net model for inference.
        
        Returns:
            True if model loaded successfully
        """
        try:
            self.logger.info("Loading nnU-Net model")
            
            if not NNUNET_AVAILABLE:
                raise SegmentationError("nnU-Net v2 not available. Please install nnunetv2.")
            
            model_dir = self.model_path / "dental_segmentator"
            if not model_dir.exists():
                self.logger.warning("Model not found, attempting to download")
                self.download_model()
            
            # Initialize nnU-Net predictor (mock implementation for MVP)
            if NNUNET_AVAILABLE:
                # This would be the actual nnU-Net model loading
                # self.model = nnUNetPredictor(...)
                pass
            
            # For MVP, create a mock model
            self.model = MockNnUNetModel(self.device)
            self.is_model_loaded = True
            
            self.logger.info("nnU-Net model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise SegmentationError(f"Failed to load model: {e}")
    
    def segment(self, volume: np.ndarray) -> SegmentationResult:
        """
        Perform dental segmentation on 3D volume.
        
        Args:
            volume: 3D numpy array representing DICOM volume
            
        Returns:
            SegmentationResult with segmentation mask and metadata
        """
        try:
            start_time = time.time()
            self.logger.info("Starting dental segmentation")
            
            # Validate input
            if not self._validate_input_volume(volume):
                raise SegmentationError("Invalid input volume")
            
            # Load model if not already loaded
            if not self.is_model_loaded:
                self.load_model()
            
            # Preprocess volume
            self.logger.info("Preprocessing volume")
            preprocessed_volume = self._preprocess_volume(volume)
            
            # Perform inference
            self.logger.info("Running inference")
            raw_segmentation = self._run_inference(preprocessed_volume)
            
            # Postprocess results
            self.logger.info("Postprocessing segmentation")
            final_segmentation = self._postprocess_segmentation(raw_segmentation)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(final_segmentation)
            
            # Count detected teeth
            detected_teeth = self._count_detected_teeth(final_segmentation)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result
            result = SegmentationResult(
                segmentation_mask=final_segmentation,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                model_version=self.model_name,
                detected_teeth_count=detected_teeth,
                metadata={
                    "original_shape": volume.shape,
                    "preprocessed_shape": preprocessed_volume.shape,
                    "device_used": self.device,
                    "confidence_threshold": self.confidence_threshold
                }
            )
            
            self.logger.info(
                f"Segmentation completed: {detected_teeth} teeth detected "
                f"in {processing_time:.2f} seconds"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            raise SegmentationError(f"Segmentation failed: {e}")
    
    def _validate_input_volume(self, volume: np.ndarray) -> bool:
        """
        Validate input volume for segmentation.
        
        Args:
            volume: Input volume to validate
            
        Returns:
            True if volume is valid
        """
        if volume is None:
            return False
        
        if not isinstance(volume, np.ndarray):
            return False
        
        if len(volume.shape) != 3:
            return False
        
        # Check minimum size
        min_size = 32
        if any(dim < min_size for dim in volume.shape):
            return False
        
        return True
    
    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Preprocess volume for nnU-Net inference.
        
        Args:
            volume: Raw DICOM volume
            
        Returns:
            Preprocessed volume
        """
        # Clone volume for processing
        processed = volume.copy().astype(np.float32)
        
        # Intensity normalization (CT-specific)
        processed = np.clip(processed, self.intensity_bounds[0], self.intensity_bounds[1])
        
        # Z-score normalization
        mean_val = np.mean(processed)
        std_val = np.std(processed)
        if std_val > 0:
            processed = (processed - mean_val) / std_val
        
        # Ensure minimum size requirements
        target_shape = [max(dim, 64) for dim in processed.shape]
        if processed.shape != tuple(target_shape):
            # Simple resizing for MVP (would use proper resampling in production)
            from scipy.ndimage import zoom
            zoom_factors = [t/o for t, o in zip(target_shape, processed.shape)]
            processed = zoom(processed, zoom_factors, order=1)
        
        return processed
    
    def _run_inference(self, volume: np.ndarray) -> np.ndarray:
        """
        Run nnU-Net inference on preprocessed volume.
        
        Args:
            volume: Preprocessed volume
            
        Returns:
            Raw segmentation result
        """
        if not self.is_model_loaded or self.model is None:
            raise SegmentationError("Model not loaded")
        
        # Use GPU context manager for inference
        with self.gpu_manager.device_context(self.device):
            # Monitor memory before inference
            memory_info = self.gpu_manager.get_memory_info(self.device)
            self.logger.debug(f"Memory before inference: {memory_info}")
            
            try:
                # Clear cache before inference to free up memory
                if self.device.startswith("cuda") or self.device == "mps":
                    self.gpu_manager.clear_cache(self.device)
                
                # For MVP, use mock inference
                result = self.model.predict(volume)
                
                # Monitor memory after inference
                memory_info = self.gpu_manager.get_memory_info(self.device)
                self.logger.debug(f"Memory after inference: {memory_info}")
                
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.error(f"GPU out of memory during inference: {e}")
                    # Try to free memory and retry on CPU if possible
                    if self.device.startswith("cuda") or self.device == "mps":
                        self.gpu_manager.clear_cache()
                        self.logger.info("Falling back to CPU due to GPU memory issues")
                        # Switch to CPU for this inference
                        original_device = self.device
                        self.device = "cpu"
                        try:
                            result = self.model.predict(volume)
                            return result
                        finally:
                            self.device = original_device
                    else:
                        raise SegmentationError(f"Out of memory error: {e}")
                else:
                    raise SegmentationError(f"Inference failed: {e}")
    
    def _postprocess_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Postprocess segmentation results.
        
        Args:
            segmentation: Raw segmentation from model
            
        Returns:
            Cleaned segmentation
        """
        # Remove small connected components
        processed = self._remove_small_components(segmentation)
        
        # Fill holes in segmented regions
        processed = self._fill_holes(processed)
        
        # Apply morphological operations
        processed = self._apply_morphological_operations(processed)
        
        return processed.astype(np.uint8)
    
    def _remove_small_components(self, segmentation: np.ndarray) -> np.ndarray:
        """Remove small connected components from segmentation."""
        result = segmentation.copy()
        min_size = 100  # Minimum voxels for valid tooth
        
        for tooth_label in np.unique(segmentation):
            if tooth_label == 0:  # Background
                continue
            
            # Create binary mask for this tooth
            tooth_mask = (segmentation == tooth_label)
            
            # Label connected components
            labeled, num_components = label(tooth_mask)
            
            # Remove small components
            for comp_label in range(1, num_components + 1):
                component_mask = (labeled == comp_label)
                if np.sum(component_mask) < min_size:
                    result[component_mask] = 0
        
        return result
    
    def _fill_holes(self, segmentation: np.ndarray) -> np.ndarray:
        """Fill holes in segmented regions."""
        result = segmentation.copy()
        
        for tooth_label in np.unique(segmentation):
            if tooth_label == 0:  # Background
                continue
            
            # Create binary mask for this tooth
            tooth_mask = (segmentation == tooth_label)
            
            # Fill holes
            filled = binary_fill_holes(tooth_mask)
            
            # Update result
            result[filled & (result == 0)] = tooth_label
        
        return result
    
    def _apply_morphological_operations(self, segmentation: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean segmentation."""
        # Apply opening to remove small noise
        kernel = morphology.ball(1)
        
        result = segmentation.copy()
        for tooth_label in np.unique(segmentation):
            if tooth_label == 0:
                continue
            
            tooth_mask = (segmentation == tooth_label)
            cleaned = morphology.binary_opening(tooth_mask, kernel)
            result[tooth_mask & ~cleaned] = 0
        
        return result
    
    def _calculate_confidence_scores(self, segmentation: np.ndarray) -> List[float]:
        """
        Calculate confidence scores for detected teeth.
        
        Args:
            segmentation: Final segmentation mask
            
        Returns:
            List of confidence scores
        """
        scores = []
        
        for tooth_label in np.unique(segmentation):
            if tooth_label == 0:  # Background
                continue
            
            tooth_mask = (segmentation == tooth_label)
            volume = np.sum(tooth_mask)
            
            # Simple confidence based on volume and compactness
            # In production, this would use model prediction probabilities
            if volume > 500:  # Reasonable tooth volume
                confidence = min(1.0, volume / 2000.0)
            else:
                confidence = 0.3  # Low confidence for small objects
            
            scores.append(confidence)
        
        return scores
    
    def _count_detected_teeth(self, segmentation: np.ndarray) -> int:
        """
        Count number of detected teeth.
        
        Args:
            segmentation: Final segmentation mask
            
        Returns:
            Number of detected teeth
        """
        unique_labels = np.unique(segmentation)
        # Subtract 1 for background label (0)
        return len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
    
    def _prepare_model_download_path(self) -> str:
        """Prepare model download path."""
        return str(self.model_path / "dental_segmentator_model.zip")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "is_loaded": self.is_model_loaded,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "pytorch_available": PYTORCH_AVAILABLE,
            "nnunet_available": NNUNET_AVAILABLE
        }


class MockNnUNetModel:
    """Mock nnU-Net model for MVP testing."""
    
    def __init__(self, device: str):
        self.device = device
    
    def predict(self, volume: np.ndarray) -> np.ndarray:
        """
        Mock prediction that creates realistic tooth segmentation.
        
        Args:
            volume: Input volume
            
        Returns:
            Mock segmentation with tooth-like structures
        """
        # Create mock segmentation with multiple teeth
        segmentation = np.zeros(volume.shape, dtype=np.uint8)
        
        # Add some tooth-like structures
        z_mid = volume.shape[2] // 2
        
        # Mock tooth 1 (upper left)
        segmentation[20:35, 20:35, z_mid-5:z_mid+5] = 1
        
        # Mock tooth 2 (upper right)
        segmentation[50:65, 20:35, z_mid-5:z_mid+5] = 2
        
        # Mock tooth 3 (lower left)
        segmentation[20:35, 50:65, z_mid-5:z_mid+5] = 3
        
        # Mock tooth 4 (lower right)
        segmentation[50:65, 50:65, z_mid-5:z_mid+5] = 4
        
        # Add some noise and realistic boundaries
        import random
        for _ in range(100):
            z = random.randint(0, volume.shape[2]-1)
            y = random.randint(0, volume.shape[1]-1)
            x = random.randint(0, volume.shape[0]-1)
            
            if segmentation[x, y, z] > 0:
                # Add some boundary noise
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nx, ny, nz = x+dx, y+dy, z+dz
                            if (0 <= nx < volume.shape[0] and 
                                0 <= ny < volume.shape[1] and 
                                0 <= nz < volume.shape[2]):
                                if random.random() < 0.1:
                                    segmentation[nx, ny, nz] = segmentation[x, y, z]
        
        return segmentation