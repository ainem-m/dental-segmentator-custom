"""
Custom Exception Classes for Dental Segmentator Application.

This module provides comprehensive exception handling with:
- Hierarchical exception classes for different error types
- Error severity classification
- Recovery hints and suggestions
- Detailed error context preservation
- Integration with logging system
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "LOW"           # Warning level, processing can continue
    MEDIUM = "MEDIUM"     # Error level, current operation fails but others can continue
    HIGH = "HIGH"         # Critical error, processing should stop
    CRITICAL = "CRITICAL" # System-level error, application should terminate


class ErrorCategory(Enum):
    """Error category classification."""
    INPUT_VALIDATION = "INPUT_VALIDATION"
    FILE_IO = "FILE_IO"
    DICOM_PROCESSING = "DICOM_PROCESSING"
    SEGMENTATION = "SEGMENTATION"
    STL_GENERATION = "STL_GENERATION"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    CONFIGURATION = "CONFIGURATION"
    SYSTEM = "SYSTEM"


class DentalSegmentatorError(Exception):
    """
    Base exception class for all dental segmentator errors.
    
    Provides common functionality for error handling, logging,
    and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error message
            severity: Error severity level
            category: Error category
            error_code: Unique error code for programmatic handling
            context: Additional error context information
            recovery_hint: Suggestion for error recovery
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.cause = cause
        self.timestamp = datetime.now()
        
        # Add exception class name to context
        self.context['exception_class'] = self.__class__.__name__
        self.context['timestamp'] = self.timestamp.isoformat()
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        return f"[{self.severity.value}] {self.message}"
    
    def get_full_context(self) -> Dict[str, Any]:
        """
        Get complete error context for logging and debugging.
        
        Returns:
            Dictionary with full error context
        """
        context = {
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'recovery_hint': self.recovery_hint,
            'context': self.context
        }
        
        if self.cause:
            context['caused_by'] = {
                'type': type(self.cause).__name__,
                'message': str(self.cause)
            }
        
        return context
    
    def is_recoverable(self) -> bool:
        """
        Check if error is potentially recoverable.
        
        Returns:
            True if error might be recoverable
        """
        return self.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
    
    def should_continue_processing(self) -> bool:
        """
        Check if processing should continue after this error.
        
        Returns:
            True if processing can continue
        """
        return self.severity == ErrorSeverity.LOW


# Input Validation Errors
class InputValidationError(DentalSegmentatorError):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        invalid_input: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if invalid_input:
            context['invalid_input'] = invalid_input
        if expected_format:
            context['expected_format'] = expected_format
            
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INPUT_VALIDATION,
            recovery_hint="Please check input format and try again",
            context=context,
            **kwargs
        )


# File I/O Errors
class FileIOError(DentalSegmentatorError):
    """Base class for file I/O related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
            
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.FILE_IO,
            context=context,
            **kwargs
        )


class FileNotFoundError(FileIOError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: str, **kwargs):
        super().__init__(
            f"File not found: {file_path}",
            file_path=file_path,
            recovery_hint="Check that the file exists and path is correct",
            **kwargs
        )


class FilePermissionError(FileIOError):
    """Raised when file access is denied due to permissions."""
    
    def __init__(self, file_path: str, operation: str = "access", **kwargs):
        context = kwargs.get('context', {})
        context['operation'] = operation
        
        super().__init__(
            f"Permission denied for {operation} operation on: {file_path}",
            file_path=file_path,
            recovery_hint="Check file permissions and access rights",
            context=context,
            **kwargs
        )


class DiskSpaceError(FileIOError):
    """Raised when insufficient disk space is available."""
    
    def __init__(
        self,
        required_space: Optional[int] = None,
        available_space: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if required_space:
            context['required_space_bytes'] = required_space
        if available_space:
            context['available_space_bytes'] = available_space
            
        super().__init__(
            "Insufficient disk space for operation",
            severity=ErrorSeverity.HIGH,
            recovery_hint="Free up disk space and try again",
            context=context,
            **kwargs
        )


# DICOM Processing Errors
class DICOMProcessingError(DentalSegmentatorError):
    """Base class for DICOM processing errors."""
    
    def __init__(
        self,
        message: str,
        dicom_file: Optional[str] = None,
        series_uid: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if dicom_file:
            context['dicom_file'] = dicom_file
        if series_uid:
            context['series_uid'] = series_uid
            
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DICOM_PROCESSING,
            context=context,
            **kwargs
        )


class InvalidDICOMError(DICOMProcessingError):
    """Raised when DICOM file is invalid or corrupted."""
    
    def __init__(self, dicom_file: str, validation_errors: Optional[List[str]] = None, **kwargs):
        context = kwargs.get('context', {})
        if validation_errors:
            context['validation_errors'] = validation_errors
            
        super().__init__(
            f"Invalid DICOM file: {dicom_file}",
            dicom_file=dicom_file,
            recovery_hint="Check DICOM file integrity and format",
            context=context,
            **kwargs
        )


class UnsupportedModalityError(DICOMProcessingError):
    """Raised when DICOM modality is not supported."""
    
    def __init__(
        self,
        modality: str,
        dicom_file: Optional[str] = None,
        supported_modalities: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['found_modality'] = modality
        if supported_modalities:
            context['supported_modalities'] = supported_modalities
            
        super().__init__(
            f"Unsupported DICOM modality: {modality}",
            dicom_file=dicom_file,
            severity=ErrorSeverity.LOW,  # Can skip this file and continue
            recovery_hint=f"Use supported modalities: {supported_modalities}" if supported_modalities else None,
            context=context,
            **kwargs
        )


class MissingDICOMTagError(DICOMProcessingError):
    """Raised when required DICOM tags are missing."""
    
    def __init__(
        self,
        missing_tags: List[str],
        dicom_file: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['missing_tags'] = missing_tags
        
        super().__init__(
            f"Missing required DICOM tags: {missing_tags}",
            dicom_file=dicom_file,
            recovery_hint="Ensure DICOM file contains all required metadata",
            context=context,
            **kwargs
        )


# Segmentation Errors
class SegmentationError(DentalSegmentatorError):
    """Base class for segmentation-related errors."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if model_name:
            context['model_name'] = model_name
        if input_shape:
            context['input_shape'] = input_shape
            
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SEGMENTATION,
            context=context,
            **kwargs
        )


class ModelNotFoundError(SegmentationError):
    """Raised when segmentation model is not found."""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(
            f"Segmentation model not found: {model_path}",
            recovery_hint="Download the model or check model path configuration",
            context={'model_path': model_path},
            **kwargs
        )


class ModelLoadError(SegmentationError):
    """Raised when model loading fails."""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(
            f"Failed to load segmentation model: {model_path}",
            recovery_hint="Check model file integrity and compatibility",
            context={'model_path': model_path},
            **kwargs
        )


class InferenceError(SegmentationError):
    """Raised when segmentation inference fails."""
    
    def __init__(
        self,
        message: str = "Segmentation inference failed",
        device: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if device:
            context['device'] = device
            
        super().__init__(
            message,
            recovery_hint="Try with different device (CPU/GPU) or reduce input size",
            context=context,
            **kwargs
        )


class GPUOutOfMemoryError(SegmentationError):
    """Raised when GPU runs out of memory."""
    
    def __init__(self, required_memory: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if required_memory:
            context['required_memory_mb'] = required_memory
            
        super().__init__(
            "GPU out of memory during segmentation",
            recovery_hint="Reduce batch size, use CPU, or free GPU memory",
            context=context,
            **kwargs
        )


# STL Generation Errors
class STLGenerationError(DentalSegmentatorError):
    """Base class for STL generation errors."""
    
    def __init__(
        self,
        message: str,
        tooth_label: Optional[int] = None,
        mesh_info: Optional[Dict] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if tooth_label:
            context['tooth_label'] = tooth_label
        if mesh_info:
            context['mesh_info'] = mesh_info
            
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.STL_GENERATION,
            context=context,
            **kwargs
        )


class MeshGenerationError(STLGenerationError):
    """Raised when mesh generation fails."""
    
    def __init__(self, tooth_label: int, reason: str, **kwargs):
        super().__init__(
            f"Mesh generation failed for tooth {tooth_label}: {reason}",
            tooth_label=tooth_label,
            recovery_hint="Check segmentation quality and mesh generation parameters",
            **kwargs
        )


class InvalidMeshError(STLGenerationError):
    """Raised when generated mesh is invalid."""
    
    def __init__(
        self,
        tooth_label: int,
        validation_issues: List[str],
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['validation_issues'] = validation_issues
        
        super().__init__(
            f"Invalid mesh generated for tooth {tooth_label}: {validation_issues}",
            tooth_label=tooth_label,
            recovery_hint="Try different mesh generation parameters or post-processing",
            context=context,
            **kwargs
        )


# Resource Limit Errors
class ResourceLimitError(DentalSegmentatorError):
    """Base class for resource limit errors."""
    
    def __init__(self, message: str, resource_type: str, **kwargs):
        context = kwargs.get('context', {})
        context['resource_type'] = resource_type
        
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RESOURCE_LIMIT,
            context=context,
            **kwargs
        )


class MemoryLimitExceededError(ResourceLimitError):
    """Raised when memory limit is exceeded."""
    
    def __init__(
        self,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if current_usage:
            context['current_usage_mb'] = current_usage
        if limit:
            context['limit_mb'] = limit
            
        message = "Memory limit exceeded"
        if current_usage and limit:
            message += f" ({current_usage:.1f}MB used, {limit:.1f}MB limit)"
            
        super().__init__(
            message,
            resource_type="memory",
            recovery_hint="Reduce processing batch size or increase memory limit",
            context=context,
            **kwargs
        )


class ProcessingTimeoutError(ResourceLimitError):
    """Raised when processing times out."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        elapsed_seconds: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['operation'] = operation
        context['timeout_seconds'] = timeout_seconds
        if elapsed_seconds:
            context['elapsed_seconds'] = elapsed_seconds
            
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            resource_type="time",
            recovery_hint="Increase timeout limit or optimize processing parameters",
            context=context,
            **kwargs
        )


# Configuration Errors
class ConfigurationError(DentalSegmentatorError):
    """Raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value:
            context['config_value'] = str(config_value)
            
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recovery_hint="Check configuration file and correct invalid values",
            context=context,
            **kwargs
        )


# System Errors
class SystemDependencyError(DentalSegmentatorError):
    """Raised when system dependencies are missing."""
    
    def __init__(
        self,
        dependency: str,
        required_version: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context['dependency'] = dependency
        if required_version:
            context['required_version'] = required_version
            
        message = f"Missing system dependency: {dependency}"
        if required_version:
            message += f" (version {required_version} required)"
            
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SYSTEM,
            recovery_hint=f"Install {dependency}" + (f" version {required_version}" if required_version else ""),
            context=context,
            **kwargs
        )


# Database Errors
class DatabaseError(DentalSegmentatorError):
    """Base class for database-related errors."""
    
    def __init__(
        self,
        message: str,
        table: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if table:
            context['table'] = table
        if operation:
            context['operation'] = operation
            
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            recovery_hint="Check database connection and data integrity",
            context=context,
            **kwargs
        )


# Error Handler Utility Functions
def handle_error_with_recovery(
    error: Exception,
    component: str,
    operation: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> bool:
    """
    Handle error with automatic recovery attempts.
    
    Args:
        error: Exception to handle
        component: Component where error occurred
        operation: Operation that failed
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        True if recovery was successful
    """
    from .utils.logging_manager import get_logging_manager
    import time
    
    logger = get_logging_manager().get_logger(f"error_handler.{component}")
    
    if isinstance(error, DentalSegmentatorError):
        logger.error(f"Error in {component}.{operation}: {error.get_full_context()}")
        
        if not error.is_recoverable():
            logger.error(f"Non-recoverable error, stopping operation")
            return False
        
        if error.should_continue_processing():
            logger.warning(f"Low severity error, continuing with processing")
            return True
        
        # Attempt recovery for recoverable errors
        for attempt in range(max_retries):
            logger.info(f"Attempting recovery {attempt + 1}/{max_retries} for {operation}")
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
            # Recovery logic would be implemented here based on error type
            # This is a placeholder for actual recovery mechanisms
            
        logger.error(f"Recovery failed after {max_retries} attempts")
        return False
    
    else:
        # Handle non-custom exceptions
        logger.error(f"Unexpected error in {component}.{operation}: {error}")
        return False


def create_error_summary(errors: List[Exception]) -> Dict[str, Any]:
    """
    Create summary of multiple errors for reporting.
    
    Args:
        errors: List of exceptions
        
    Returns:
        Dictionary with error summary
    """
    summary = {
        'total_errors': len(errors),
        'error_categories': {},
        'severity_breakdown': {},
        'recoverable_errors': 0,
        'critical_errors': 0,
        'most_common_error': None,
        'errors': []
    }
    
    error_types = {}
    
    for error in errors:
        if isinstance(error, DentalSegmentatorError):
            # Count by category
            category = error.category.value
            summary['error_categories'][category] = summary['error_categories'].get(category, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            summary['severity_breakdown'][severity] = summary['severity_breakdown'].get(severity, 0) + 1
            
            # Count recoverable and critical errors
            if error.is_recoverable():
                summary['recoverable_errors'] += 1
            if error.severity == ErrorSeverity.CRITICAL:
                summary['critical_errors'] += 1
            
            # Track error types
            error_type = error.__class__.__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Add to error list
            summary['errors'].append({
                'type': error_type,
                'message': error.message,
                'severity': severity,
                'category': category,
                'timestamp': error.timestamp.isoformat(),
                'recoverable': error.is_recoverable()
            })
        else:
            # Handle non-custom exceptions
            error_type = error.__class__.__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            summary['errors'].append({
                'type': error_type,
                'message': str(error),
                'severity': 'UNKNOWN',
                'category': 'UNKNOWN',
                'recoverable': False
            })
    
    # Find most common error type
    if error_types:
        summary['most_common_error'] = max(error_types.items(), key=lambda x: x[1])
    
    return summary