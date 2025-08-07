"""
Logging Manager for Dental Segmentator Application.

This module provides comprehensive logging functionality including:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- File and console handlers
- Log rotation
- Component-specific logging
- Performance and resource monitoring
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import json


class LoggingManager:
    """
    Centralized logging manager for the dental segmentator application.
    
    Features:
    - Multiple log levels with configurable thresholds
    - File rotation to prevent log files from growing too large
    - Console and file output handlers
    - Component-specific logger creation
    - Structured logging with context information
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: str = "logs",
        max_file_size: str = "100MB",
        backup_count: int = 5,
        log_format: Optional[str] = None
    ):
        """
        Initialize the logging manager.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_dir: Directory to store log files
            max_file_size: Maximum size for log files before rotation
            backup_count: Number of backup log files to keep
            log_format: Custom log format string
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.max_file_size = self._parse_file_size(max_file_size)
        self.backup_count = backup_count
        self.log_format = log_format or (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize loggers
        self._setup_root_logger()
        self._component_loggers: Dict[str, logging.Logger] = {}
    
    def _parse_file_size(self, size_str: str) -> int:
        """
        Parse file size string to bytes.
        
        Args:
            size_str: Size string like "100MB", "1GB", etc.
            
        Returns:
            Size in bytes
        """
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def _setup_root_logger(self) -> None:
        """Setup the root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handlers
        self._add_file_handlers(root_logger, formatter)
    
    def _add_file_handlers(self, logger: logging.Logger, formatter: logging.Formatter) -> None:
        """
        Add rotating file handlers to the logger.
        
        Args:
            logger: Logger instance to add handlers to
            formatter: Logging formatter
        """
        # Main application log
        app_log_path = self.log_dir / "application.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(self.log_level)
        app_handler.setFormatter(formatter)
        logger.addHandler(app_handler)
        
        # Processing-specific log
        processing_log_path = self.log_dir / "processing.log"
        processing_handler = logging.handlers.RotatingFileHandler(
            processing_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        processing_handler.setLevel(logging.INFO)
        processing_handler.setFormatter(formatter)
        
        # Add filter for processing-related logs
        processing_handler.addFilter(self._processing_filter)
        logger.addHandler(processing_handler)
        
        # Error-only log
        error_log_path = self.log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    def _processing_filter(self, record: logging.LogRecord) -> bool:
        """
        Filter for processing-related log messages.
        
        Args:
            record: Log record to filter
            
        Returns:
            True if the record should be logged in processing.log
        """
        processing_components = [
            'dicom_processor', 'nnunet_segmentator', 'stl_generator',
            'processing_engine'
        ]
        return any(component in record.name.lower() for component in processing_components)
    
    def get_logger(self, name: str, component: Optional[str] = None) -> logging.Logger:
        """
        Get or create a component-specific logger.
        
        Args:
            name: Logger name
            component: Component name for categorization
            
        Returns:
            Configured logger instance
        """
        if name in self._component_loggers:
            return self._component_loggers[name]
        
        logger = logging.getLogger(name)
        
        # Add component information to log records
        if component:
            logger = logging.LoggerAdapter(
                logger, 
                extra={'component': component}
            )
        
        self._component_loggers[name] = logger
        return logger
    
    def log_performance_metrics(
        self,
        component: str,
        operation: str,
        duration_seconds: float,
        memory_usage_mb: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics for monitoring and optimization.
        
        Args:
            component: Component name (e.g., 'dicom_processor')
            operation: Operation name (e.g., 'load_dicom_series')
            duration_seconds: Operation duration
            memory_usage_mb: Memory usage in MB
            additional_metrics: Additional metrics to log
        """
        logger = self.get_logger(f"performance.{component}")
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'component': component,
            'operation': operation,
            'duration_seconds': duration_seconds
        }
        
        if memory_usage_mb is not None:
            metrics['memory_usage_mb'] = memory_usage_mb
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        logger.info(f"Performance metrics: {json.dumps(metrics)}")
    
    def log_resource_usage(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_usage_percent: float,
        gpu_memory_percent: Optional[float] = None
    ) -> None:
        """
        Log system resource usage.
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_usage_percent: Disk usage percentage
            gpu_memory_percent: GPU memory usage percentage
        """
        logger = self.get_logger("resource_monitor")
        
        resources = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_usage_percent': disk_usage_percent
        }
        
        if gpu_memory_percent is not None:
            resources['gpu_memory_percent'] = gpu_memory_percent
        
        logger.info(f"Resource usage: {json.dumps(resources)}")
    
    def log_processing_status(
        self,
        job_id: str,
        dicom_series_uid: str,
        status: str,
        progress_percentage: float,
        message: Optional[str] = None
    ) -> None:
        """
        Log processing job status updates.
        
        Args:
            job_id: Unique job identifier
            dicom_series_uid: DICOM series UID being processed
            status: Current status (QUEUED, PROCESSING, COMPLETED, FAILED)
            progress_percentage: Processing progress (0-100)
            message: Optional status message
        """
        logger = self.get_logger("processing_status")
        
        status_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'job_id': job_id,
            'dicom_series_uid': dicom_series_uid,
            'status': status,
            'progress_percentage': progress_percentage
        }
        
        if message:
            status_info['message'] = message
        
        logger.info(f"Processing status: {json.dumps(status_info)}")
    
    def log_error_with_context(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log errors with detailed context information.
        
        Args:
            error: Exception that occurred
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context information
        """
        logger = self.get_logger(f"error.{component}")
        
        error_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'component': component,
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
        
        if context:
            error_info['context'] = context
        
        logger.error(
            f"Error in {component}.{operation}: {error_info}",
            exc_info=True
        )
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up old log files.
        
        Args:
            days_to_keep: Number of days to keep log files
        """
        logger = self.get_logger("log_cleanup")
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        cleaned_count = 0
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Removed old log file: {log_file}")
                except OSError as e:
                    logger.warning(f"Failed to remove log file {log_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old log files")
    
    def set_log_level(self, level: str) -> None:
        """
        Update the log level for all handlers.
        
        Args:
            level: New log level (DEBUG, INFO, WARNING, ERROR)
        """
        new_level = getattr(logging, level.upper())
        self.log_level = new_level
        
        root_logger = logging.getLogger()
        root_logger.setLevel(new_level)
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(new_level)
        
        logger = self.get_logger("logging_manager")
        logger.info(f"Log level changed to {level}")


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """
    Get the global logging manager instance.
    
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def initialize_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_file_size: str = "100MB",
    backup_count: int = 5,
    log_format: Optional[str] = None
) -> LoggingManager:
    """
    Initialize the global logging manager.
    
    Args:
        log_level: Logging level
        log_dir: Log directory
        max_file_size: Maximum log file size
        backup_count: Number of backup files
        log_format: Custom log format
        
    Returns:
        Initialized LoggingManager instance
    """
    global _logging_manager
    _logging_manager = LoggingManager(
        log_level=log_level,
        log_dir=log_dir,
        max_file_size=max_file_size,
        backup_count=backup_count,
        log_format=log_format
    )
    return _logging_manager