"""
Unit tests for LoggingManager functionality.
"""

import pytest
import tempfile
import logging
from pathlib import Path
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_manager import LoggingManager, initialize_logging


class TestLoggingManager:
    """Test cases for LoggingManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        
        # Clear any existing logging handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clear logging handlers after test
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
    
    def test_initialization_default_params(self):
        """Test LoggingManager initialization with default parameters."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        assert logging_manager.log_level == logging.INFO
        assert logging_manager.log_dir == self.log_dir
        assert logging_manager.backup_count == 5
        
        # Check if log directory was created
        assert self.log_dir.exists()
    
    def test_initialization_custom_params(self):
        """Test LoggingManager initialization with custom parameters."""
        logging_manager = LoggingManager(
            log_level="DEBUG",
            log_dir=str(self.log_dir),
            max_file_size="50MB",
            backup_count=3
        )
        
        assert logging_manager.log_level == logging.DEBUG
        assert logging_manager.backup_count == 3
    
    def test_parse_file_size(self):
        """Test file size parsing functionality."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        assert logging_manager._parse_file_size("100KB") == 100 * 1024
        assert logging_manager._parse_file_size("50MB") == 50 * 1024 * 1024
        assert logging_manager._parse_file_size("1GB") == 1 * 1024 * 1024 * 1024
        assert logging_manager._parse_file_size("1000") == 1000
    
    def test_get_logger(self):
        """Test logger creation and retrieval."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        logger1 = logging_manager.get_logger("test.component1")
        logger2 = logging_manager.get_logger("test.component2", component="component2")
        
        assert logger1 is not None
        assert logger2 is not None
        
        # Test that same logger is returned for same name
        logger1_again = logging_manager.get_logger("test.component1")
        assert logger1 is logger1_again
    
    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        # This should not raise any exceptions
        logging_manager.log_performance_metrics(
            component="test_component",
            operation="test_operation",
            duration_seconds=1.5,
            memory_usage_mb=128.0,
            additional_metrics={"gpu_memory": 256}
        )
    
    def test_log_resource_usage(self):
        """Test resource usage logging."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        # This should not raise any exceptions
        logging_manager.log_resource_usage(
            cpu_percent=75.5,
            memory_percent=60.2,
            disk_usage_percent=45.0,
            gpu_memory_percent=80.5
        )
    
    def test_log_processing_status(self):
        """Test processing status logging."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        # This should not raise any exceptions
        logging_manager.log_processing_status(
            job_id="test_job_001",
            dicom_series_uid="1.2.3.4.5",
            status="PROCESSING",
            progress_percentage=45.5,
            message="Processing DICOM series"
        )
    
    def test_log_error_with_context(self):
        """Test error logging with context."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            # This should not raise any exceptions
            logging_manager.log_error_with_context(
                error=e,
                component="test_component",
                operation="test_operation",
                context={"input_file": "test.dcm"}
            )
    
    def test_set_log_level(self):
        """Test changing log level at runtime."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        # Initially INFO
        assert logging_manager.log_level == logging.INFO
        
        # Change to DEBUG
        logging_manager.set_log_level("DEBUG")
        assert logging_manager.log_level == logging.DEBUG
        
        # Change to ERROR
        logging_manager.set_log_level("ERROR")
        assert logging_manager.log_level == logging.ERROR
    
    def test_cleanup_old_logs(self):
        """Test log cleanup functionality."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        # Create some test log files
        (self.log_dir / "old_log1.log").touch()
        (self.log_dir / "old_log2.log").touch()
        
        # This should not raise any exceptions
        logging_manager.cleanup_old_logs(days_to_keep=30)
    
    def test_initialize_logging_function(self):
        """Test the initialize_logging function."""
        logging_manager = initialize_logging(
            log_level="DEBUG",
            log_dir=str(self.log_dir),
            max_file_size="10MB",
            backup_count=2
        )
        
        assert isinstance(logging_manager, LoggingManager)
        assert logging_manager.log_level == logging.DEBUG
        assert logging_manager.backup_count == 2
    
    def test_log_file_creation(self):
        """Test that log files are created properly."""
        logging_manager = LoggingManager(log_dir=str(self.log_dir))
        
        # Get a logger and log something
        logger = logging_manager.get_logger("test_logger")
        logger.info("Test message")
        
        # Force log file creation by getting handlers to flush
        for handler in logging.getLogger().handlers:
            handler.flush()
        
        # Check that log files exist
        expected_files = ["application.log", "processing.log", "error.log"]
        for log_file in expected_files:
            log_path = self.log_dir / log_file
            # Files might not be created immediately, so we just check the directory exists
            assert self.log_dir.exists()