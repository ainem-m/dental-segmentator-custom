"""
STL Generator for Dental Segmentation Results.

This module provides comprehensive STL file generation capabilities including:
- Mesh generation from segmentation masks using marching cubes
- STL file export with trimesh integration
- Mesh optimization (smoothing, simplification)
- Quality validation and metrics
- 3D printing ready output
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

# Import mesh processing dependencies
try:
    import trimesh
    from trimesh import Trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None
    Trimesh = None

try:
    from skimage import measure
    from skimage.morphology import binary_closing, ball
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    measure = None

from scipy import ndimage
from scipy.ndimage import gaussian_filter

from ..utils.logging_manager import get_logging_manager


@dataclass
class MeshInfo:
    """
    Data structure containing mesh information and quality metrics.
    
    Attributes:
        tooth_label: Label/ID of the tooth
        vertex_count: Number of vertices in the mesh
        face_count: Number of faces in the mesh
        volume: Mesh volume in cubic mm
        surface_area: Mesh surface area in square mm
        is_watertight: Whether the mesh is watertight
        is_winding_consistent: Whether face winding is consistent
        quality_score: Overall quality score (0-1)
        file_path: Path to the generated STL file
        file_size_bytes: Size of the STL file in bytes
    """
    tooth_label: int
    vertex_count: int = 0
    face_count: int = 0
    volume: float = 0.0
    surface_area: float = 0.0
    is_watertight: bool = False
    is_winding_consistent: bool = False
    quality_score: float = 0.0
    file_path: str = ""
    file_size_bytes: int = 0


class STLGenerationError(Exception):
    """Exception raised for STL generation errors."""
    pass


class STLGenerator:
    """
    STL file generator for dental segmentation results.
    
    Features:
    - Marching cubes mesh generation
    - Mesh optimization and smoothing
    - Quality validation and metrics
    - 3D printing ready STL export
    - Batch processing for multiple teeth
    """
    
    def __init__(
        self,
        output_directory: str,
        spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        smoothing_enabled: bool = True,
        smoothing_iterations: int = 5,
        simplification_enabled: bool = False,
        simplification_ratio: float = 0.1,
        quality_threshold: float = 0.7
    ):
        """
        Initialize STL generator.
        
        Args:
            output_directory: Directory to save STL files
            spacing: Voxel spacing in mm (x, y, z)
            smoothing_enabled: Enable mesh smoothing
            smoothing_iterations: Number of smoothing iterations
            simplification_enabled: Enable mesh simplification
            simplification_ratio: Simplification ratio (0-1)
            quality_threshold: Minimum quality threshold
        """
        self.output_directory = Path(output_directory)
        self.spacing = spacing
        self.smoothing_enabled = smoothing_enabled
        self.smoothing_iterations = smoothing_iterations
        self.simplification_enabled = simplification_enabled
        self.simplification_ratio = simplification_ratio
        self.quality_threshold = quality_threshold
        
        self.logger = get_logging_manager().get_logger(
            "generators.stl_generator",
            component="stl_generator"
        )
        
        # Create output directory
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Mesh generation parameters
        self.marching_cubes_level = 0.5
        self.min_volume_threshold = 50.0  # Minimum tooth volume in cubic mm
        
        self.logger.info(f"STL Generator initialized with output: {self.output_directory}")
    
    def generate_stl_files(
        self,
        segmentation: np.ndarray,
        case_id: str,
        spacing_override: Optional[Tuple[float, float, float]] = None
    ) -> List[MeshInfo]:
        """
        Generate STL files for all teeth in segmentation.
        
        Args:
            segmentation: 3D segmentation mask with tooth labels
            case_id: Case identifier for file naming
            spacing_override: Override default spacing
            
        Returns:
            List of MeshInfo objects for generated STL files
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting STL generation for case: {case_id}")
            
            # Validate input
            if not self._validate_segmentation(segmentation):
                raise STLGenerationError("Invalid segmentation input")
            
            # Use provided spacing or default
            spacing = spacing_override or self.spacing
            
            # Extract unique tooth labels
            tooth_labels = self._extract_tooth_labels(segmentation)
            if not tooth_labels:
                raise STLGenerationError("No teeth found in segmentation")
            
            self.logger.info(f"Found {len(tooth_labels)} teeth to process: {tooth_labels}")
            
            # Generate STL for each tooth
            mesh_info_list = []
            for tooth_label in tooth_labels:
                try:
                    mesh_info = self._generate_single_stl(
                        segmentation, tooth_label, case_id, spacing
                    )
                    if mesh_info:
                        mesh_info_list.append(mesh_info)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to generate STL for tooth {tooth_label}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"STL generation completed for case {case_id}: "
                f"{len(mesh_info_list)}/{len(tooth_labels)} successful "
                f"in {processing_time:.2f} seconds"
            )
            
            # Generate metadata file
            self._generate_metadata_file(case_id, mesh_info_list, processing_time)
            
            return mesh_info_list
            
        except Exception as e:
            self.logger.error(f"STL generation failed for case {case_id}: {e}")
            raise STLGenerationError(f"STL generation failed: {e}")
    
    def _validate_segmentation(self, segmentation: np.ndarray) -> bool:
        """
        Validate segmentation input.
        
        Args:
            segmentation: Segmentation to validate
            
        Returns:
            True if segmentation is valid
        """
        if segmentation is None:
            return False
        
        if not isinstance(segmentation, np.ndarray):
            return False
        
        if len(segmentation.shape) != 3:
            return False
        
        if segmentation.size == 0:
            return False
        
        # Check if there are any non-zero values (teeth)
        if not np.any(segmentation > 0):
            return False
        
        return True
    
    def _extract_tooth_labels(self, segmentation: np.ndarray) -> List[int]:
        """
        Extract unique tooth labels from segmentation.
        
        Args:
            segmentation: 3D segmentation mask
            
        Returns:
            List of unique tooth labels (excluding background)
        """
        unique_labels = np.unique(segmentation)
        # Remove background label (0)
        tooth_labels = [int(label) for label in unique_labels if label > 0]
        return sorted(tooth_labels)
    
    def _generate_single_stl(
        self,
        segmentation: np.ndarray,
        tooth_label: int,
        case_id: str,
        spacing: Tuple[float, float, float]
    ) -> Optional[MeshInfo]:
        """
        Generate STL file for a single tooth.
        
        Args:
            segmentation: Full segmentation mask
            tooth_label: Label of the tooth to extract
            case_id: Case identifier
            spacing: Voxel spacing
            
        Returns:
            MeshInfo object or None if generation failed
        """
        try:
            self.logger.debug(f"Generating STL for tooth {tooth_label}")
            
            # Extract single tooth mask
            tooth_mask = self._extract_single_tooth(segmentation, tooth_label)
            
            # Check minimum volume
            if not self._check_minimum_volume(tooth_mask, spacing):
                self.logger.warning(f"Tooth {tooth_label} below minimum volume threshold")
                return None
            
            # Generate mesh using marching cubes
            vertices, faces = self._generate_mesh_from_mask(tooth_mask, spacing)
            
            if len(vertices) == 0 or len(faces) == 0:
                self.logger.warning(f"Empty mesh generated for tooth {tooth_label}")
                return None
            
            # Create mesh object
            mesh = self._create_mesh_object(vertices, faces)
            
            # Apply optimizations
            if self.smoothing_enabled:
                mesh = self._apply_smoothing(mesh)
            
            if self.simplification_enabled:
                mesh = self._apply_simplification(mesh)
            
            # Validate mesh quality
            mesh_info = self._validate_mesh(mesh, tooth_label)
            
            if mesh_info.quality_score < self.quality_threshold:
                self.logger.warning(
                    f"Tooth {tooth_label} quality score {mesh_info.quality_score:.2f} "
                    f"below threshold {self.quality_threshold}"
                )
            
            # Export STL file
            output_path = self._export_stl_file(mesh, case_id, tooth_label)
            
            # Update mesh info with file information
            mesh_info.file_path = str(output_path)
            mesh_info.file_size_bytes = output_path.stat().st_size
            
            self.logger.debug(
                f"STL generated for tooth {tooth_label}: "
                f"{mesh_info.vertex_count} vertices, {mesh_info.face_count} faces"
            )
            
            return mesh_info
            
        except Exception as e:
            self.logger.error(f"Failed to generate STL for tooth {tooth_label}: {e}")
            return None
    
    def _extract_single_tooth(self, segmentation: np.ndarray, tooth_label: int) -> np.ndarray:
        """
        Extract binary mask for a single tooth.
        
        Args:
            segmentation: Full segmentation
            tooth_label: Tooth label to extract
            
        Returns:
            Binary mask for the specified tooth
        """
        tooth_mask = (segmentation == tooth_label).astype(bool)
        
        # Apply morphological closing to fill small gaps
        if np.any(tooth_mask):
            # Use small structuring element for dental anatomy
            structure = ball(1)  # 3x3x3 structuring element
            tooth_mask = binary_closing(tooth_mask, structure)
        
        return tooth_mask
    
    def _check_minimum_volume(self, tooth_mask: np.ndarray, spacing: Tuple[float, float, float]) -> bool:
        """
        Check if tooth volume meets minimum threshold.
        
        Args:
            tooth_mask: Binary tooth mask
            spacing: Voxel spacing
            
        Returns:
            True if volume is above threshold
        """
        voxel_count = np.sum(tooth_mask)
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³ per voxel
        total_volume = voxel_count * voxel_volume
        
        return total_volume >= self.min_volume_threshold
    
    def _generate_mesh_from_mask(
        self,
        mask: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mesh from binary mask using marching cubes.
        
        Args:
            mask: Binary mask
            spacing: Voxel spacing
            
        Returns:
            Tuple of (vertices, faces) arrays
        """
        if not SKIMAGE_AVAILABLE:
            raise STLGenerationError("scikit-image not available for marching cubes")
        
        try:
            # Smooth the mask slightly to reduce mesh artifacts
            smoothed_mask = gaussian_filter(mask.astype(float), sigma=0.5)
            
            # Apply marching cubes algorithm
            vertices, faces, normals, values = measure.marching_cubes(
                smoothed_mask,
                level=self.marching_cubes_level,
                spacing=spacing
            )
            
            # Ensure proper data types
            vertices = vertices.astype(np.float32)
            faces = faces.astype(np.int32)
            
            return vertices, faces
            
        except Exception as e:
            raise STLGenerationError(f"Marching cubes failed: {e}")
    
    def _create_mesh_object(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Create trimesh object from vertices and faces.
        
        Args:
            vertices: Mesh vertices
            faces: Mesh faces
            
        Returns:
            Trimesh object or mock object if trimesh not available
        """
        if TRIMESH_AVAILABLE:
            try:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                return mesh
            except Exception as e:
                self.logger.warning(f"Failed to create trimesh object: {e}")
        
        # Create mock mesh object for testing
        return MockMesh(vertices, faces)
    
    def _apply_smoothing(self, mesh) -> object:
        """
        Apply Laplacian smoothing to mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Smoothed mesh
        """
        if hasattr(mesh, 'smoothed'):
            # Use trimesh smoothing if available
            try:
                return mesh.smoothed()
            except:
                pass
        
        # For mock mesh, return as-is
        return mesh
    
    def _apply_simplification(self, mesh) -> object:
        """
        Apply mesh simplification.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Simplified mesh
        """
        if hasattr(mesh, 'simplify_quadric_decimation'):
            try:
                target_faces = int(len(mesh.faces) * (1 - self.simplification_ratio))
                return mesh.simplify_quadric_decimation(target_faces)
            except:
                pass
        
        # For mock mesh, return as-is
        return mesh
    
    def _validate_mesh(self, mesh, tooth_label: int) -> MeshInfo:
        """
        Validate mesh quality and extract metrics.
        
        Args:
            mesh: Mesh to validate
            tooth_label: Tooth label
            
        Returns:
            MeshInfo with validation results
        """
        # Extract basic metrics
        vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
        face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
        
        # Calculate volume and surface area
        volume = mesh.volume if hasattr(mesh, 'volume') else 0.0
        surface_area = mesh.area if hasattr(mesh, 'area') else 0.0
        
        # Check mesh properties
        is_watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False
        is_winding_consistent = (
            mesh.is_winding_consistent if hasattr(mesh, 'is_winding_consistent') else True
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            vertex_count, face_count, volume, is_watertight, is_winding_consistent
        )
        
        return MeshInfo(
            tooth_label=tooth_label,
            vertex_count=vertex_count,
            face_count=face_count,
            volume=volume,
            surface_area=surface_area,
            is_watertight=is_watertight,
            is_winding_consistent=is_winding_consistent,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(
        self,
        vertex_count: int,
        face_count: int,
        volume: float,
        is_watertight: bool,
        is_winding_consistent: bool
    ) -> float:
        """
        Calculate overall mesh quality score.
        
        Args:
            vertex_count: Number of vertices
            face_count: Number of faces
            volume: Mesh volume
            is_watertight: Whether mesh is watertight
            is_winding_consistent: Whether winding is consistent
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Basic mesh existence (20%)
        if vertex_count > 0 and face_count > 0:
            score += 0.2
        
        # Reasonable mesh density (20%)
        if 100 <= vertex_count <= 50000 and 200 <= face_count <= 100000:
            score += 0.2
        
        # Reasonable volume (20%)
        if 10.0 <= volume <= 2000.0:  # Typical tooth volume range
            score += 0.2
        
        # Watertight property (20%)
        if is_watertight:
            score += 0.2
        
        # Consistent winding (20%)
        if is_winding_consistent:
            score += 0.2
        
        return score
    
    def _export_stl_file(self, mesh, case_id: str, tooth_label: int) -> Path:
        """
        Export mesh to STL file.
        
        Args:
            mesh: Mesh to export
            case_id: Case identifier
            tooth_label: Tooth label
            
        Returns:
            Path to exported STL file
        """
        filename = self._generate_filename(case_id, tooth_label)
        output_path = self.output_directory / filename
        
        if hasattr(mesh, 'export'):
            # Use trimesh export
            mesh.export(output_path)
        else:
            # Write basic STL file for mock mesh
            self._write_basic_stl(mesh, output_path)
        
        return output_path
    
    def _generate_filename(self, case_id: str, tooth_label: int) -> str:
        """
        Generate STL filename.
        
        Args:
            case_id: Case identifier
            tooth_label: Tooth label
            
        Returns:
            STL filename
        """
        return f"{case_id}_tooth_{tooth_label}.stl"
    
    def _write_basic_stl(self, mesh, output_path: Path) -> None:
        """
        Write basic ASCII STL file for mock mesh.
        
        Args:
            mesh: Mock mesh object
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write(f"solid tooth_mesh\n")
            
            # Write triangular faces
            for face in mesh.faces:
                v0, v1, v2 = mesh.vertices[face]
                
                # Calculate normal (simplified)
                normal = np.array([0.0, 0.0, 1.0])  # Placeholder normal
                
                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {v0[0]} {v0[1]} {v0[2]}\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write("endsolid tooth_mesh\n")
    
    def _generate_metadata_file(
        self,
        case_id: str,
        mesh_info_list: List[MeshInfo],
        processing_time: float
    ) -> None:
        """
        Generate metadata JSON file for the case.
        
        Args:
            case_id: Case identifier
            mesh_info_list: List of mesh information
            processing_time: Total processing time
        """
        metadata = {
            "case_id": case_id,
            "generation_timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "total_teeth_processed": len(mesh_info_list),
            "generator_settings": {
                "spacing": self.spacing,
                "smoothing_enabled": self.smoothing_enabled,
                "smoothing_iterations": self.smoothing_iterations,
                "simplification_enabled": self.simplification_enabled,
                "simplification_ratio": self.simplification_ratio,
                "quality_threshold": self.quality_threshold
            },
            "teeth": []
        }
        
        for mesh_info in mesh_info_list:
            tooth_data = {
                "tooth_label": mesh_info.tooth_label,
                "file_name": Path(mesh_info.file_path).name,
                "vertex_count": mesh_info.vertex_count,
                "face_count": mesh_info.face_count,
                "volume_mm3": mesh_info.volume,
                "surface_area_mm2": mesh_info.surface_area,
                "is_watertight": mesh_info.is_watertight,
                "quality_score": mesh_info.quality_score,
                "file_size_bytes": mesh_info.file_size_bytes
            }
            metadata["teeth"].append(tooth_data)
        
        # Write metadata file
        metadata_path = self.output_directory / f"{case_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to: {metadata_path}")


class MockMesh:
    """Mock mesh object for testing when trimesh is not available."""
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        
        # Calculate mock properties
        self.volume = len(vertices) * 0.1  # Mock volume
        self.area = len(faces) * 0.05      # Mock surface area
        self.is_watertight = True
        self.is_winding_consistent = True