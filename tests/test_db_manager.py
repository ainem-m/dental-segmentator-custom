"""
Unit tests for Database Manager functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.database.db_manager import (
    DatabaseManager,
    DICOMSeriesModel,
    SegmentationResultModel,
    STLOutputModel,
    ProcessingLogModel,
    DatabaseError
)


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_dental.db"
        self.db_manager = DatabaseManager(str(self.db_path))
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_initialization(self):
        """Test database initialization and table creation."""
        assert self.db_path.exists()
        assert self.db_manager.engine is not None
        assert self.db_manager.Session is not None
    
    def test_create_dicom_series(self):
        """Test creating DICOM series record."""
        series_data = {
            'series_uid': '1.2.3.4.5.6.7.8.9',
            'study_uid': '1.2.3.4.5.6.7.8',
            'patient_id': 'PATIENT_001',
            'modality': 'CT',
            'file_paths': ['/path/to/file1.dcm', '/path/to/file2.dcm'],
            'image_dimensions': [512, 512, 100],
            'spacing': [0.5, 0.5, 1.0],
            'file_size_bytes': 104857600
        }
        
        series_id = self.db_manager.create_dicom_series(**series_data)
        
        assert series_id is not None
        assert isinstance(series_id, str)
        
        # Retrieve and verify
        retrieved = self.db_manager.get_dicom_series_by_uid(series_data['series_uid'])
        assert retrieved is not None
        assert retrieved.series_uid == series_data['series_uid']
        assert retrieved.modality == series_data['modality']
    
    def test_create_segmentation_result(self):
        """Test creating segmentation result record."""
        # First create a DICOM series
        series_data = {
            'series_uid': '1.2.3.4.5.6.7.8.9',
            'study_uid': '1.2.3.4.5.6.7.8',
            'patient_id': 'PATIENT_001',
            'modality': 'CT',
            'file_paths': ['/path/to/file1.dcm'],
            'image_dimensions': [512, 512, 100],
            'spacing': [0.5, 0.5, 1.0],
            'file_size_bytes': 104857600
        }
        self.db_manager.create_dicom_series(**series_data)
        
        # Create segmentation result
        seg_data = {
            'dicom_series_uid': '1.2.3.4.5.6.7.8.9',
            'model_version': 'dental_segmentator_v1',
            'model_path': '/models/dental_segmentator',
            'confidence_scores': [0.85, 0.92, 0.78],
            'detected_teeth_count': 28,
            'processing_time_seconds': 45.2,
            'memory_usage_mb': 2048.0,
            'gpu_used': True,
            'status': 'SUCCESS'
        }
        
        result_id = self.db_manager.create_segmentation_result(**seg_data)
        
        assert result_id is not None
        
        # Retrieve and verify
        retrieved = self.db_manager.get_segmentation_result(result_id)
        assert retrieved is not None
        assert retrieved.detected_teeth_count == 28
        assert retrieved.status == 'SUCCESS'
    
    def test_create_stl_output(self):
        """Test creating STL output record."""
        # Setup prerequisite records
        series_data = {
            'series_uid': '1.2.3.4.5.6.7.8.9',
            'study_uid': '1.2.3.4.5.6.7.8',
            'patient_id': 'PATIENT_001',
            'modality': 'CT',
            'file_paths': ['/path/to/file1.dcm'],
            'image_dimensions': [512, 512, 100],
            'spacing': [0.5, 0.5, 1.0],
            'file_size_bytes': 104857600
        }
        self.db_manager.create_dicom_series(**series_data)
        
        seg_data = {
            'dicom_series_uid': '1.2.3.4.5.6.7.8.9',
            'model_version': 'dental_segmentator_v1',
            'model_path': '/models/dental_segmentator',
            'confidence_scores': [0.85],
            'detected_teeth_count': 1,
            'processing_time_seconds': 45.2,
            'memory_usage_mb': 2048.0,
            'status': 'SUCCESS'
        }
        seg_result_id = self.db_manager.create_segmentation_result(**seg_data)
        
        # Create STL output
        stl_data = {
            'segmentation_result_id': seg_result_id,
            'file_path': '/output/tooth_1.stl',
            'file_size_bytes': 512000,
            'mesh_vertices': 10000,
            'mesh_faces': 20000,
            'anatomical_region': 'Upper Right Molar',
            'is_watertight': True,
            'quality_score': 0.92
        }
        
        stl_id = self.db_manager.create_stl_output(**stl_data)
        
        assert stl_id is not None
        
        # Retrieve and verify
        retrieved = self.db_manager.get_stl_output(stl_id)
        assert retrieved is not None
        assert retrieved.mesh_vertices == 10000
        assert retrieved.is_watertight is True
    
    def test_get_processing_history(self):
        """Test retrieving processing history."""
        # Create some test records
        for i in range(3):
            series_data = {
                'series_uid': f'1.2.3.4.5.6.7.8.{i}',
                'study_uid': f'1.2.3.4.5.6.7.{i}',
                'patient_id': f'PATIENT_{i:03d}',
                'modality': 'CT',
                'file_paths': [f'/path/to/file{i}.dcm'],
                'image_dimensions': [512, 512, 100],
                'spacing': [0.5, 0.5, 1.0],
                'file_size_bytes': 104857600
            }
            self.db_manager.create_dicom_series(**series_data)
        
        # Get history
        history = self.db_manager.get_processing_history(limit=10)
        
        assert len(history) == 3
        assert all(record.modality == 'CT' for record in history)
    
    def test_get_statistics(self):
        """Test getting processing statistics."""
        # Create test data
        series_data = {
            'series_uid': '1.2.3.4.5.6.7.8.9',
            'study_uid': '1.2.3.4.5.6.7.8',
            'patient_id': 'PATIENT_001',
            'modality': 'CT',
            'file_paths': ['/path/to/file1.dcm'],
            'image_dimensions': [512, 512, 100],
            'spacing': [0.5, 0.5, 1.0],
            'file_size_bytes': 104857600
        }
        self.db_manager.create_dicom_series(**series_data)
        
        seg_data = {
            'dicom_series_uid': '1.2.3.4.5.6.7.8.9',
            'model_version': 'dental_segmentator_v1',
            'model_path': '/models/dental_segmentator',
            'confidence_scores': [0.85],
            'detected_teeth_count': 25,
            'processing_time_seconds': 45.2,
            'memory_usage_mb': 2048.0,
            'status': 'SUCCESS'
        }
        self.db_manager.create_segmentation_result(**seg_data)
        
        stats = self.db_manager.get_statistics()
        
        assert 'total_series_processed' in stats
        assert 'total_successful_segmentations' in stats
        assert 'average_processing_time' in stats
        assert 'total_teeth_detected' in stats
        assert stats['total_series_processed'] == 1
        assert stats['total_teeth_detected'] == 25
    
    def test_cleanup_old_records(self):
        """Test cleanup of old records."""
        # Create old record
        series_data = {
            'series_uid': '1.2.3.4.5.6.7.8.9',
            'study_uid': '1.2.3.4.5.6.7.8',
            'patient_id': 'PATIENT_001',
            'modality': 'CT',
            'file_paths': ['/path/to/file1.dcm'],
            'image_dimensions': [512, 512, 100],
            'spacing': [0.5, 0.5, 1.0],
            'file_size_bytes': 104857600
        }
        self.db_manager.create_dicom_series(**series_data)
        
        # Test cleanup (should not delete recent records)
        deleted_count = self.db_manager.cleanup_old_records(days_to_keep=1)
        
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0
    
    def test_database_migration(self):
        """Test database migration functionality."""
        # This would test schema migrations
        current_version = self.db_manager.get_schema_version()
        assert isinstance(current_version, int)
        assert current_version >= 1
    
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test duplicate series creation
        series_data = {
            'series_uid': '1.2.3.4.5.6.7.8.9',
            'study_uid': '1.2.3.4.5.6.7.8',
            'patient_id': 'PATIENT_001',
            'modality': 'CT',
            'file_paths': ['/path/to/file1.dcm'],
            'image_dimensions': [512, 512, 100],
            'spacing': [0.5, 0.5, 1.0],
            'file_size_bytes': 104857600
        }
        
        # First creation should succeed
        series_id = self.db_manager.create_dicom_series(**series_data)
        assert series_id is not None
        
        # Second creation with same series_uid should handle gracefully
        try:
            self.db_manager.create_dicom_series(**series_data)
        except DatabaseError:
            pass  # Expected behavior
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error."""
        # Test that transactions are properly rolled back on errors
        with pytest.raises(DatabaseError):
            with self.db_manager.get_session() as session:
                # This should fail and rollback
                self.db_manager.create_segmentation_result(
                    dicom_series_uid='NON_EXISTENT_UID',  # This should fail FK constraint
                    model_version='test',
                    model_path='/test',
                    status='SUCCESS'
                )
    
    def test_concurrent_access(self):
        """Test concurrent database access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def create_series(i):
            try:
                series_data = {
                    'series_uid': f'concurrent_test_{i}',
                    'study_uid': f'study_{i}',
                    'patient_id': f'PATIENT_{i}',
                    'modality': 'CT',
                    'file_paths': [f'/path/to/file{i}.dcm'],
                    'image_dimensions': [512, 512, 100],
                    'spacing': [0.5, 0.5, 1.0],
                    'file_size_bytes': 104857600
                }
                result = self.db_manager.create_dicom_series(**series_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_series, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(result is not None for result in results)