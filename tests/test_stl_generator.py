"""
Unit tests for STL Generator functionality.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.generators.stl_generator import (
    STLGenerator,
    STLGenerationError,
    MeshInfo
)


class TestSTLGenerator:
    """Test cases for STLGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_segmentation(self) -> np.ndarray:
        """Create a test segmentation with simple geometric shapes."""
        segmentation = np.zeros((64, 64, 32), dtype=np.uint8)
        
        # Create a cube-like tooth
        segmentation[20:40, 20:40, 10:22] = 1
        
        # Create a cylinder-like tooth
        center_y, center_x = 45, 45
        radius = 8
        for z in range(15, 25):
            for y in range(64):
                for x in range(64):
                    if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                        segmentation[x, y, z] = 2
        
        return segmentation
    
    def test_stl_generator_initialization(self):
        """Test STL generator initialization."""
        generator = STLGenerator(
            output_directory=str(self.output_dir),
            spacing=(0.5, 0.5, 0.5)
        )
        
        assert generator.output_directory == Path(str(self.output_dir))
        assert generator.spacing == (0.5, 0.5, 0.5)
        assert generator.smoothing_enabled is True
        assert generator.simplification_enabled is False
    
    def test_stl_generator_with_custom_settings(self):
        """Test STL generator with custom settings."""
        generator = STLGenerator(
            output_directory=str(self.output_dir),
            spacing=(0.25, 0.25, 1.0),
            smoothing_enabled=False,
            simplification_enabled=True,
            simplification_ratio=0.5
        )
        
        assert generator.spacing == (0.25, 0.25, 1.0)
        assert generator.smoothing_enabled is False
        assert generator.simplification_enabled is True
        assert generator.simplification_ratio == 0.5
    
    def test_validate_segmentation_valid(self):
        """Test segmentation validation with valid input."""
        generator = STLGenerator(str(self.output_dir))
        segmentation = self.create_test_segmentation()
        
        assert generator._validate_segmentation(segmentation) is True
    
    def test_validate_segmentation_invalid(self):
        """Test segmentation validation with invalid inputs."""
        generator = STLGenerator(str(self.output_dir))
        
        # None input
        assert generator._validate_segmentation(None) is False
        
        # Wrong dimensions
        wrong_dim = np.zeros((64, 64), dtype=np.uint8)
        assert generator._validate_segmentation(wrong_dim) is False
        
        # Empty segmentation
        empty = np.zeros((64, 64, 32), dtype=np.uint8)
        assert generator._validate_segmentation(empty) is False
    
    def test_extract_tooth_labels(self):
        """Test extraction of unique tooth labels."""
        generator = STLGenerator(str(self.output_dir))
        segmentation = self.create_test_segmentation()
        
        labels = generator._extract_tooth_labels(segmentation)
        
        assert isinstance(labels, list)
        assert 1 in labels
        assert 2 in labels
        assert 0 not in labels  # Background should be excluded
    
    def test_extract_single_tooth(self):
        """Test extraction of single tooth from segmentation."""
        generator = STLGenerator(str(self.output_dir))
        segmentation = self.create_test_segmentation()
        
        tooth_mask = generator._extract_single_tooth(segmentation, tooth_label=1)
        
        assert isinstance(tooth_mask, np.ndarray)
        assert tooth_mask.dtype == bool
        assert np.any(tooth_mask)  # Should contain some True values
        assert tooth_mask.shape == segmentation.shape
    
    def test_generate_mesh_from_mask(self):
        """Test mesh generation from binary mask."""
        generator = STLGenerator(str(self.output_dir))
        segmentation = self.create_test_segmentation()
        tooth_mask = generator._extract_single_tooth(segmentation, tooth_label=1)
        
        # Test spacing (typical CT scan spacing)
        spacing = (0.5, 0.5, 0.5)
        vertices, faces = generator._generate_mesh_from_mask(tooth_mask, spacing)
        
        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert vertices.shape[1] == 3  # 3D coordinates
        assert faces.shape[1] == 3     # Triangular faces
        assert len(vertices) > 0
        assert len(faces) > 0
    
    def test_create_mesh_object(self):
        """Test creation of trimesh object."""
        generator = STLGenerator(str(self.output_dir))
        
        # Create simple test mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        
        mesh = generator._create_mesh_object(vertices, faces)
        
        assert mesh is not None
        # Basic mesh properties would be checked here if trimesh was available
    
    def test_validate_mesh(self):
        """Test mesh validation."""
        generator = STLGenerator(str(self.output_dir))
        
        # Create simple test mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
        
        mock_mesh = type('MockMesh', (), {
            'vertices': vertices,
            'faces': faces,
            'is_watertight': True,
            'is_winding_consistent': True
        })()
        
        mesh_info = generator._validate_mesh(mock_mesh, tooth_label=1)
        
        assert isinstance(mesh_info, MeshInfo)
        assert mesh_info.tooth_label == 1
        assert mesh_info.vertex_count == 4
        assert mesh_info.face_count == 4
        assert mesh_info.is_watertight is True
    
    def test_generate_filename(self):
        """Test STL filename generation."""
        generator = STLGenerator(str(self.output_dir))
        
        filename = generator._generate_filename("case_001", 15)
        assert filename == "case_001_tooth_15.stl"
        
        filename = generator._generate_filename("patient_xyz", 3)
        assert filename == "patient_xyz_tooth_3.stl"
    
    def test_mesh_info_dataclass(self):
        """Test MeshInfo data structure."""
        mesh_info = MeshInfo(
            tooth_label=8,
            vertex_count=1024,
            face_count=2048,
            volume=125.5,
            surface_area=85.2,
            is_watertight=True,
            quality_score=0.85
        )
        
        assert mesh_info.tooth_label == 8
        assert mesh_info.vertex_count == 1024
        assert mesh_info.face_count == 2048
        assert mesh_info.volume == 125.5
        assert mesh_info.is_watertight is True
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        generator = STLGenerator(str(self.output_dir))
        
        # Test with None segmentation
        with pytest.raises(STLGenerationError):
            generator.generate_stl_files(None, "test_case")
        
        # Test with invalid segmentation
        with pytest.raises(STLGenerationError):
            invalid_seg = np.zeros((10, 10), dtype=np.uint8)
            generator.generate_stl_files(invalid_seg, "test_case")
        
        # Test with empty segmentation
        with pytest.raises(STLGenerationError):
            empty_seg = np.zeros((64, 64, 32), dtype=np.uint8)
            generator.generate_stl_files(empty_seg, "test_case")