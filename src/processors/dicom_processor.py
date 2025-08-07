"""
DICOM Processor for Dental Segmentator Application.

This module provides comprehensive DICOM file processing capabilities including:
- DICOM file reading and validation
- DICOM series identification and grouping
- Metadata extraction and verification
- Image data preprocessing and normalization
- NumPy array conversion for downstream processing
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import pydicom
from pydicom.errors import InvalidDicomError
import pydicom.uid

from ..utils.logging_manager import get_logging_manager


@dataclass
class DICOMSeries:
    """
    Data structure representing a DICOM series.
    
    Attributes:
        series_uid: Unique identifier for the series
        study_uid: Study instance UID
        patient_id: Patient identifier (may be anonymized)
        modality: Imaging modality (e.g., 'CT', 'CBCT')
        file_paths: List of file paths in the series
        image_dimensions: Image dimensions as [width, height, slices]
        spacing: Pixel/slice spacing as [x, y, z]
        slice_thickness: Slice thickness in mm
        created_at: Timestamp when series was processed
        metadata: Additional DICOM metadata
    """
    series_uid: str
    study_uid: str
    patient_id: str
    modality: str
    file_paths: List[str] = field(default_factory=list)
    image_dimensions: Tuple[int, int, int] = (0, 0, 0)
    spacing: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    slice_thickness: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


class DICOMProcessingError(Exception):
    """Exception raised for DICOM processing errors."""
    pass


class DICOMProcessor:
    """
    DICOM processor for dental imaging data.
    
    Features:
    - Automatic DICOM file discovery and validation
    - Series identification and grouping
    - Metadata extraction and verification
    - Image preprocessing and normalization
    - Privacy-aware processing (anonymization support)
    """
    
    def __init__(self, anonymize_tags: bool = True):
        """
        Initialize DICOM processor.
        
        Args:
            anonymize_tags: Whether to anonymize personal information
        """
        self.anonymize_tags = anonymize_tags
        self.logger = get_logging_manager().get_logger(
            "processors.dicom_processor",
            component="dicom_processor"
        )
        
        # Supported file extensions
        self.dicom_extensions = {'.dcm', '.dicom', '.DCM', '.DICOM'}
        
        # Required DICOM tags for dental CT processing
        self.required_tags = [
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'Modality',
            'PixelData'
        ]
        
        # Supported modalities for dental imaging
        self.supported_modalities = {
            'CT',      # Computed Tomography
            'CBCT',    # Cone Beam CT
            'DX',      # Digital Radiography
            'IO',      # Intra-oral Radiography
        }
    
    def scan_directory(self, directory_path: str) -> List[DICOMSeries]:
        """
        Scan directory for DICOM files and group into series.
        
        Args:
            directory_path: Path to directory containing DICOM files
            
        Returns:
            List of identified DICOM series
            
        Raises:
            DICOMProcessingError: If directory scanning fails
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise DICOMProcessingError(f"Directory does not exist: {directory_path}")
            
            if not directory.is_dir():
                raise DICOMProcessingError(f"Path is not a directory: {directory_path}")
            
            self.logger.info(f"Scanning directory for DICOM files: {directory_path}")
            
            # Find all potential DICOM files
            dicom_files = self._find_dicom_files(directory)
            if not dicom_files:
                self.logger.warning(f"No DICOM files found in directory: {directory_path}")
                return []
            
            self.logger.info(f"Found {len(dicom_files)} potential DICOM files")
            
            # Read and validate DICOM files
            valid_dicom_data = []
            for file_path in dicom_files:
                try:
                    dicom_data = self._read_dicom_file(file_path)
                    if dicom_data:
                        valid_dicom_data.append((file_path, dicom_data))
                except Exception as e:
                    self.logger.warning(f"Failed to read DICOM file {file_path}: {e}")
                    continue
            
            if not valid_dicom_data:
                raise DICOMProcessingError("No valid DICOM files found")
            
            self.logger.info(f"Successfully read {len(valid_dicom_data)} valid DICOM files")
            
            # Group files into series
            series_list = self._group_into_series(valid_dicom_data)
            
            self.logger.info(f"Identified {len(series_list)} DICOM series")
            return series_list
            
        except Exception as e:
            self.logger.error(f"Failed to scan directory {directory_path}: {e}")
            raise DICOMProcessingError(f"Directory scanning failed: {e}")
    
    def _find_dicom_files(self, directory: Path) -> List[Path]:
        """
        Find all potential DICOM files in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of potential DICOM file paths
        """
        dicom_files = []
        
        # Search recursively for files with DICOM extensions
        for ext in self.dicom_extensions:
            dicom_files.extend(directory.rglob(f"*{ext}"))
        
        # Also check files without extensions (common for DICOM)
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not file_path.suffix:
                # Check if it might be a DICOM file by attempting to read header
                try:
                    with open(file_path, 'rb') as f:
                        # DICOM files have "DICM" at offset 128
                        f.seek(128)
                        if f.read(4) == b'DICM':
                            dicom_files.append(file_path)
                except:
                    continue
        
        return sorted(set(dicom_files))
    
    def _read_dicom_file(self, file_path: Path) -> Optional[pydicom.Dataset]:
        """
        Read and validate a single DICOM file.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            DICOM dataset if valid, None otherwise
            
        Raises:
            DICOMProcessingError: If file reading fails
        """
        try:
            # Read DICOM file
            dicom_dataset = pydicom.dcmread(str(file_path), force=True)
            
            # Validate required tags
            missing_tags = []
            for tag in self.required_tags:
                if not hasattr(dicom_dataset, tag):
                    missing_tags.append(tag)
            
            if missing_tags:
                self.logger.warning(
                    f"DICOM file {file_path} missing required tags: {missing_tags}"
                )
                # For dental CT, we might still be able to process with some missing tags
                if 'PixelData' in missing_tags:
                    return None  # Cannot process without pixel data
            
            # Validate modality
            modality = getattr(dicom_dataset, 'Modality', 'UNKNOWN')
            if modality not in self.supported_modalities:
                self.logger.warning(
                    f"Unsupported modality '{modality}' in file {file_path}. "
                    f"Supported modalities: {self.supported_modalities}"
                )
                # Continue processing anyway - might be a non-standard modality
            
            # Anonymize sensitive tags if requested
            if self.anonymize_tags:
                self._anonymize_dataset(dicom_dataset)
            
            return dicom_dataset
            
        except InvalidDicomError as e:
            self.logger.warning(f"Invalid DICOM file {file_path}: {e}")
            return None
        except Exception as e:
            raise DICOMProcessingError(f"Failed to read DICOM file {file_path}: {e}")
    
    def _anonymize_dataset(self, dataset: pydicom.Dataset) -> None:
        """
        Anonymize sensitive tags in DICOM dataset.
        
        Args:
            dataset: DICOM dataset to anonymize
        """
        # Common patient identification tags to anonymize
        sensitive_tags = [
            'PatientName',
            'PatientID', 
            'PatientBirthDate',
            'PatientSex',
            'PatientAge',
            'InstitutionName',
            'ReferringPhysicianName',
            'PerformingPhysicianName',
            'OperatorsName'
        ]
        
        for tag in sensitive_tags:
            if hasattr(dataset, tag):
                if tag == 'PatientID':
                    # Keep a hash-based anonymous ID for tracking
                    import hashlib
                    original_id = str(getattr(dataset, tag))
                    anonymous_id = hashlib.sha256(original_id.encode()).hexdigest()[:8]
                    setattr(dataset, tag, f"ANON_{anonymous_id}")
                else:
                    setattr(dataset, tag, "ANONYMIZED")
    
    def _group_into_series(self, dicom_data: List[Tuple[Path, pydicom.Dataset]]) -> List[DICOMSeries]:
        """
        Group DICOM files into series based on SeriesInstanceUID.
        
        Args:
            dicom_data: List of (file_path, dataset) tuples
            
        Returns:
            List of DICOM series
        """
        series_dict = {}
        
        for file_path, dataset in dicom_data:
            try:
                # Extract series information
                series_uid = getattr(dataset, 'SeriesInstanceUID', 'UNKNOWN')
                study_uid = getattr(dataset, 'StudyInstanceUID', 'UNKNOWN')
                patient_id = getattr(dataset, 'PatientID', 'UNKNOWN')
                modality = getattr(dataset, 'Modality', 'UNKNOWN')
                
                if series_uid not in series_dict:
                    # Create new series
                    series_dict[series_uid] = DICOMSeries(
                        series_uid=series_uid,
                        study_uid=study_uid,
                        patient_id=patient_id,
                        modality=modality
                    )
                
                # Add file to series
                series_dict[series_uid].file_paths.append(str(file_path))
                
                # Extract metadata from first file in series
                if len(series_dict[series_uid].file_paths) == 1:
                    self._extract_series_metadata(series_dict[series_uid], dataset)
                
            except Exception as e:
                self.logger.warning(f"Failed to process file {file_path} for series grouping: {e}")
                continue
        
        # Sort files within each series and calculate dimensions
        series_list = []
        for series in series_dict.values():
            try:
                self._finalize_series(series)
                series_list.append(series)
            except Exception as e:
                self.logger.warning(f"Failed to finalize series {series.series_uid}: {e}")
                continue
        
        return series_list
    
    def _extract_series_metadata(self, series: DICOMSeries, dataset: pydicom.Dataset) -> None:
        """
        Extract metadata from DICOM dataset for series.
        
        Args:
            series: DICOM series to update
            dataset: DICOM dataset to extract metadata from
        """
        # Extract pixel spacing
        pixel_spacing = getattr(dataset, 'PixelSpacing', [1.0, 1.0])
        if isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 2:
            pixel_spacing_x = float(pixel_spacing[0])
            pixel_spacing_y = float(pixel_spacing[1])
        else:
            pixel_spacing_x = pixel_spacing_y = 1.0
        
        # Extract slice thickness
        slice_thickness = float(getattr(dataset, 'SliceThickness', 1.0))
        series.slice_thickness = slice_thickness
        
        # Store spacing (x, y, z)
        series.spacing = (pixel_spacing_x, pixel_spacing_y, slice_thickness)
        
        # Extract additional metadata
        series.metadata = {
            'StudyDate': getattr(dataset, 'StudyDate', ''),
            'StudyTime': getattr(dataset, 'StudyTime', ''),
            'SeriesDescription': getattr(dataset, 'SeriesDescription', ''),
            'Manufacturer': getattr(dataset, 'Manufacturer', ''),
            'ManufacturerModelName': getattr(dataset, 'ManufacturerModelName', ''),
            'ConvolutionKernel': getattr(dataset, 'ConvolutionKernel', ''),
            'KVP': getattr(dataset, 'KVP', ''),
            'XRayTubeCurrent': getattr(dataset, 'XRayTubeCurrent', ''),
        }
    
    def _finalize_series(self, series: DICOMSeries) -> None:
        """
        Finalize series by sorting files and calculating dimensions.
        
        Args:
            series: DICOM series to finalize
        """
        if not series.file_paths:
            raise DICOMProcessingError(f"Series {series.series_uid} has no files")
        
        # Sort files by InstanceNumber or SliceLocation
        sorted_files = self._sort_series_files(series.file_paths)
        series.file_paths = sorted_files
        
        # Read first file to get image dimensions
        try:
            dataset = pydicom.dcmread(series.file_paths[0], force=True)
            rows = int(getattr(dataset, 'Rows', 0))
            cols = int(getattr(dataset, 'Columns', 0))
            slices = len(series.file_paths)
            
            series.image_dimensions = (cols, rows, slices)
            
        except Exception as e:
            raise DICOMProcessingError(
                f"Failed to determine dimensions for series {series.series_uid}: {e}"
            )
        
        self.logger.debug(
            f"Finalized series {series.series_uid}: "
            f"{series.image_dimensions} voxels, {len(series.file_paths)} slices"
        )
    
    def _sort_series_files(self, file_paths: List[str]) -> List[str]:
        """
        Sort DICOM files in correct anatomical order.
        
        Args:
            file_paths: List of file paths to sort
            
        Returns:
            Sorted list of file paths
        """
        file_info = []
        
        for file_path in file_paths:
            try:
                dataset = pydicom.dcmread(file_path, force=True)
                
                # Try to get instance number first
                instance_number = getattr(dataset, 'InstanceNumber', None)
                if instance_number is not None:
                    sort_key = int(instance_number)
                else:
                    # Fall back to slice location
                    slice_location = getattr(dataset, 'SliceLocation', None)
                    if slice_location is not None:
                        sort_key = float(slice_location)
                    else:
                        # Last resort: use filename
                        sort_key = os.path.basename(file_path)
                
                file_info.append((sort_key, file_path))
                
            except Exception:
                # If we can't read the file, sort by filename
                file_info.append((os.path.basename(file_path), file_path))
        
        # Sort and return file paths
        file_info.sort(key=lambda x: x[0])
        return [file_path for _, file_path in file_info]
    
    def validate_dicom_series(self, series: DICOMSeries) -> bool:
        """
        Validate a DICOM series for processing compatibility.
        
        Args:
            series: DICOM series to validate
            
        Returns:
            True if series is valid for processing
        """
        try:
            self.logger.info(f"Validating DICOM series: {series.series_uid}")
            
            # Check if files exist
            missing_files = []
            for file_path in series.file_paths:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self.logger.error(f"Missing files in series {series.series_uid}: {missing_files}")
                return False
            
            # Check minimum number of slices for 3D processing
            if len(series.file_paths) < 10:
                self.logger.warning(
                    f"Series {series.series_uid} has only {len(series.file_paths)} slices. "
                    "Minimum 10 slices recommended for dental segmentation."
                )
            
            # Check image dimensions
            width, height, slices = series.image_dimensions
            if width < 100 or height < 100:
                self.logger.warning(
                    f"Series {series.series_uid} has small image dimensions: {width}x{height}. "
                    "This may affect segmentation quality."
                )
            
            # Check pixel spacing
            x_spacing, y_spacing, z_spacing = series.spacing
            if x_spacing > 1.0 or y_spacing > 1.0:
                self.logger.warning(
                    f"Series {series.series_uid} has large pixel spacing: {x_spacing}x{y_spacing}mm. "
                    "This may affect segmentation accuracy."
                )
            
            # Check modality compatibility
            if series.modality not in self.supported_modalities:
                self.logger.warning(
                    f"Series {series.series_uid} has unsupported modality: {series.modality}"
                )
            
            self.logger.info(f"Series validation completed: {series.series_uid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Series validation failed for {series.series_uid}: {e}")
            return False
    
    def preprocess_images(self, series: DICOMSeries) -> np.ndarray:
        """
        Load and preprocess DICOM images into NumPy array.
        
        Args:
            series: DICOM series to preprocess
            
        Returns:
            Preprocessed image array with shape (slices, height, width)
            
        Raises:
            DICOMProcessingError: If preprocessing fails
        """
        try:
            self.logger.info(f"Preprocessing images for series: {series.series_uid}")
            
            if not series.file_paths:
                raise DICOMProcessingError("No files in series")
            
            # Read first image to get dimensions and data type
            first_dataset = pydicom.dcmread(series.file_paths[0], force=True)
            pixel_array = first_dataset.pixel_array
            
            # Initialize output array
            slices, height, width = len(series.file_paths), pixel_array.shape[0], pixel_array.shape[1]
            volume = np.zeros((slices, height, width), dtype=np.int16)
            
            # Load all slices
            for i, file_path in enumerate(series.file_paths):
                try:
                    dataset = pydicom.dcmread(file_path, force=True)
                    pixel_data = dataset.pixel_array.astype(np.int16)
                    
                    # Apply rescale slope and intercept if present
                    slope = getattr(dataset, 'RescaleSlope', 1.0)
                    intercept = getattr(dataset, 'RescaleIntercept', 0.0)
                    
                    if slope != 1.0 or intercept != 0.0:
                        pixel_data = pixel_data * slope + intercept
                    
                    volume[i] = pixel_data
                    
                except Exception as e:
                    self.logger.error(f"Failed to process slice {i} from {file_path}: {e}")
                    raise DICOMProcessingError(f"Failed to process slice {i}: {e}")
            
            self.logger.info(
                f"Successfully preprocessed {slices} slices with shape {volume.shape} "
                f"for series {series.series_uid}"
            )
            
            return volume
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed for series {series.series_uid}: {e}")
            raise DICOMProcessingError(f"Image preprocessing failed: {e}")
    
    def get_series_info(self, series: DICOMSeries) -> Dict:
        """
        Get comprehensive information about a DICOM series.
        
        Args:
            series: DICOM series
            
        Returns:
            Dictionary containing series information
        """
        return {
            'series_uid': series.series_uid,
            'study_uid': series.study_uid,
            'patient_id': series.patient_id,
            'modality': series.modality,
            'num_slices': len(series.file_paths),
            'image_dimensions': series.image_dimensions,
            'spacing': series.spacing,
            'slice_thickness': series.slice_thickness,
            'total_files': len(series.file_paths),
            'created_at': series.created_at.isoformat(),
            'metadata': series.metadata
        }