"""
Main Processing Engine for Dental Segmentator.

This module provides the core processing engine that orchestrates the
complete DICOM to STL conversion pipeline, integrating all components
and managing the end-to-end workflow.
"""

import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from ..processors.dicom_processor import DICOMProcessor, DICOMSeries, DICOMProcessingError
from ..segmentation.nnunet_segmentator import NnUNetSegmentator, SegmentationResult, SegmentationError
from ..generators.stl_generator import STLGenerator, MeshInfo, STLGenerationError
from ..config.config_manager import AppConfig
from ..utils.logging_manager import get_logging_manager


class ProcessingStatus(Enum):
    """Processing job status enumeration."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class ProcessingJob:
    """
    Processing job data structure.
    
    Attributes:
        job_id: Unique job identifier
        dicom_series_uid: DICOM series UID being processed
        input_path: Input DICOM directory path
        output_path: Output STL directory path
        status: Current processing status
        progress_percentage: Processing progress (0-100)
        started_at: Job start timestamp
        completed_at: Job completion timestamp
        error_details: Error details if processing failed
        metadata: Additional job metadata
    """
    job_id: str
    dicom_series_uid: str
    input_path: str
    output_path: str
    status: ProcessingStatus = ProcessingStatus.QUEUED
    progress_percentage: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_details: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """
    Complete processing result.
    
    Attributes:
        job_id: Job identifier
        dicom_series: Processed DICOM series information
        segmentation_result: Segmentation results
        stl_files: List of generated STL files
        total_processing_time: Total processing time in seconds
        memory_usage_peak_mb: Peak memory usage in MB
        status: Final processing status
        error_message: Error message if failed
    """
    job_id: str
    dicom_series: Optional[DICOMSeries] = None
    segmentation_result: Optional[SegmentationResult] = None
    stl_files: List[MeshInfo] = field(default_factory=list)
    total_processing_time: float = 0.0
    memory_usage_peak_mb: float = 0.0
    status: ProcessingStatus = ProcessingStatus.QUEUED
    error_message: Optional[str] = None


class ProcessingError(Exception):
    """Exception raised for processing engine errors."""
    pass


class ProcessingEngine:
    """
    Main processing engine for dental DICOM to STL conversion.
    
    Features:
    - End-to-end processing pipeline
    - Resource monitoring and management
    - Error handling and recovery
    - Progress tracking and reporting
    - Batch processing support
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize processing engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logging_manager().get_logger(
            "engine.processing_engine",
            component="processing_engine"
        )
        
        # Initialize processors
        self.dicom_processor = DICOMProcessor(
            anonymize_tags=config.security.anonymize_dicom_tags
        )
        
        self.segmentator = NnUNetSegmentator(
            model_path=config.models.base_path,
            device="cuda" if config.hardware.gpu_enabled else "cpu",
            confidence_threshold=config.segmentation.confidence_threshold
        )
        
        self.stl_generator = STLGenerator(
            output_directory=config.processing.output_directory,
            smoothing_enabled=True,
            smoothing_iterations=config.segmentation.smoothing_iterations,
            simplification_enabled=config.segmentation.mesh_simplification,
            simplification_ratio=config.segmentation.simplification_ratio
        )
        
        # Processing state
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.completed_jobs: List[ProcessingResult] = []
        
        # Resource monitoring
        self.memory_limit_mb = config.hardware.memory_limit
        self.monitoring_enabled = True
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.logger.info("Processing engine initialized successfully")
    
    def process_directory(self, input_path: str, output_path: str) -> List[ProcessingResult]:
        """
        Process all DICOM series in a directory.
        
        Args:
            input_path: Input directory containing DICOM files
            output_path: Output directory for STL files
            
        Returns:
            List of processing results
        """
        try:
            self.logger.info(f"Starting directory processing: {input_path} -> {output_path}")
            
            # Start resource monitoring
            self._start_resource_monitoring()
            
            # Discover DICOM series
            self.logger.info("Scanning for DICOM series...")
            dicom_series_list = self.dicom_processor.scan_directory(input_path)
            
            if not dicom_series_list:
                raise ProcessingError(f"No DICOM series found in {input_path}")
            
            self.logger.info(f"Found {len(dicom_series_list)} DICOM series to process")
            
            # Process each series
            results = []
            for i, dicom_series in enumerate(dicom_series_list):
                try:
                    self.logger.info(
                        f"Processing series {i+1}/{len(dicom_series_list)}: "
                        f"{dicom_series.series_uid}"
                    )
                    
                    result = self.process_dicom_series(
                        dicom_series=dicom_series,
                        output_path=output_path
                    )
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process series {dicom_series.series_uid}: {e}")
                    
                    # Create failed result
                    failed_result = ProcessingResult(
                        job_id=f"job_{dicom_series.series_uid}",
                        dicom_series=dicom_series,
                        status=ProcessingStatus.FAILED,
                        error_message=str(e)
                    )
                    results.append(failed_result)
                    continue
            
            # Stop resource monitoring
            self._stop_resource_monitoring()
            
            # Generate summary report
            self._generate_processing_report(results, input_path, output_path)
            
            successful_count = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
            self.logger.info(
                f"Directory processing completed: {successful_count}/{len(results)} successful"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Directory processing failed: {e}")
            raise ProcessingError(f"Directory processing failed: {e}")
    
    def process_dicom_series(
        self,
        dicom_series: Optional[DICOMSeries] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single DICOM series through the complete pipeline.
        
        Args:
            dicom_series: Pre-loaded DICOM series (optional)
            input_path: Input DICOM directory (if dicom_series not provided)
            output_path: Output directory for STL files
            
        Returns:
            ProcessingResult with complete processing information
        """
        job_id = f"job_{int(time.time() * 1000)}"
        start_time = time.time()
        memory_monitor = MemoryMonitor()
        
        try:
            self.logger.info(f"Starting DICOM series processing: {job_id}")
            
            # Load DICOM series if not provided
            if dicom_series is None:
                if input_path is None:
                    raise ProcessingError("Either dicom_series or input_path must be provided")
                
                self.logger.info("Loading DICOM series from input path")
                series_list = self.dicom_processor.scan_directory(input_path)
                if not series_list:
                    raise ProcessingError(f"No DICOM series found in {input_path}")
                
                dicom_series = series_list[0]  # Process first series
                self.logger.info(f"Processing series: {dicom_series.series_uid}")
            
            # Create processing job
            job = ProcessingJob(
                job_id=job_id,
                dicom_series_uid=dicom_series.series_uid,
                input_path=input_path or "unknown",
                output_path=output_path or self.config.processing.output_directory,
                started_at=datetime.now()
            )
            
            self._update_job_progress(job, 10.0, ProcessingStatus.PROCESSING)
            
            # Step 1: Validate DICOM series
            self.logger.info("Step 1: Validating DICOM series")
            if not self.dicom_processor.validate_dicom_series(dicom_series):
                raise ProcessingError("DICOM series validation failed")
            
            self._update_job_progress(job, 20.0)
            
            # Step 2: Preprocess DICOM images
            self.logger.info("Step 2: Preprocessing DICOM images")
            volume_data = self.dicom_processor.preprocess_images(dicom_series)
            
            self._update_job_progress(job, 40.0)
            
            # Step 3: Perform segmentation
            self.logger.info("Step 3: Performing dental segmentation")
            segmentation_result = self.segmentator.segment(volume_data)
            
            self._update_job_progress(job, 70.0)
            
            # Step 4: Generate STL files
            self.logger.info("Step 4: Generating STL files")
            case_id = f"case_{dicom_series.patient_id}_{dicom_series.series_uid[:8]}"
            stl_files = self.stl_generator.generate_stl_files(
                segmentation_result.segmentation_mask,
                case_id,
                spacing_override=dicom_series.spacing
            )
            
            self._update_job_progress(job, 90.0)
            
            # Step 5: Finalize processing
            self.logger.info("Step 5: Finalizing processing")
            total_time = time.time() - start_time
            peak_memory = memory_monitor.get_peak_usage()
            
            # Create successful result
            result = ProcessingResult(
                job_id=job_id,
                dicom_series=dicom_series,
                segmentation_result=segmentation_result,
                stl_files=stl_files,
                total_processing_time=total_time,
                memory_usage_peak_mb=peak_memory,
                status=ProcessingStatus.COMPLETED
            )
            
            self._update_job_progress(job, 100.0, ProcessingStatus.COMPLETED)
            
            # Log performance metrics
            self.logger.info(
                f"Processing completed successfully: "
                f"{len(stl_files)} STL files generated in {total_time:.2f}s "
                f"(Peak memory: {peak_memory:.1f}MB)"
            )
            
            # Log detailed metrics
            get_logging_manager().log_performance_metrics(
                component="processing_engine",
                operation="process_dicom_series",
                duration_seconds=total_time,
                memory_usage_mb=peak_memory,
                additional_metrics={
                    "dicom_series_uid": dicom_series.series_uid,
                    "detected_teeth": segmentation_result.detected_teeth_count,
                    "stl_files_generated": len(stl_files),
                    "input_dimensions": dicom_series.image_dimensions
                }
            )
            
            self.completed_jobs.append(result)
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            peak_memory = memory_monitor.get_peak_usage()
            
            self.logger.error(f"Processing failed for job {job_id}: {e}")
            
            # Log error with context
            get_logging_manager().log_error_with_context(
                error=e,
                component="processing_engine",
                operation="process_dicom_series",
                context={
                    "job_id": job_id,
                    "dicom_series_uid": dicom_series.series_uid if dicom_series else "unknown",
                    "processing_time": total_time,
                    "peak_memory_mb": peak_memory
                }
            )
            
            # Create failed result
            result = ProcessingResult(
                job_id=job_id,
                dicom_series=dicom_series,
                total_processing_time=total_time,
                memory_usage_peak_mb=peak_memory,
                status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
            
            if job_id in self.active_jobs:
                self._update_job_progress(self.active_jobs[job_id], 0.0, ProcessingStatus.FAILED)
            
            self.completed_jobs.append(result)
            raise ProcessingError(f"DICOM series processing failed: {e}")
    
    def _update_job_progress(
        self,
        job: ProcessingJob,
        progress: float,
        status: Optional[ProcessingStatus] = None
    ) -> None:
        """
        Update job progress and status.
        
        Args:
            job: Processing job
            progress: Progress percentage (0-100)
            status: New status (optional)
        """
        job.progress_percentage = progress
        if status:
            job.status = status
            if status == ProcessingStatus.COMPLETED:
                job.completed_at = datetime.now()
        
        self.active_jobs[job.job_id] = job
        
        # Log progress
        get_logging_manager().log_processing_status(
            job_id=job.job_id,
            dicom_series_uid=job.dicom_series_uid,
            status=job.status.value,
            progress_percentage=job.progress_percentage
        )
    
    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring thread."""
        if self.monitoring_enabled and self.monitor_thread is None:
            self.monitor_thread = threading.Thread(
                target=self._resource_monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()
            self.logger.debug("Resource monitoring started")
    
    def _stop_resource_monitoring(self) -> None:
        """Stop resource monitoring thread."""
        if self.monitor_thread:
            self.monitoring_enabled = False
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None
            self.logger.debug("Resource monitoring stopped")
    
    def _resource_monitor_loop(self) -> None:
        """Resource monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Get system resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Check for resource limits
                if memory.percent > 90:
                    self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
                
                if disk.percent > 90:
                    self.logger.warning(f"High disk usage: {disk.percent:.1f}%")
                
                # Log resource usage
                get_logging_manager().log_resource_usage(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_usage_percent=disk.percent
                )
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.warning(f"Resource monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _generate_processing_report(
        self,
        results: List[ProcessingResult],
        input_path: str,
        output_path: str
    ) -> None:
        """
        Generate processing report.
        
        Args:
            results: List of processing results
            input_path: Input directory
            output_path: Output directory
        """
        try:
            successful_results = [r for r in results if r.status == ProcessingStatus.COMPLETED]
            failed_results = [r for r in results if r.status == ProcessingStatus.FAILED]
            
            total_time = sum(r.total_processing_time for r in results)
            total_stl_files = sum(len(r.stl_files) for r in successful_results)
            avg_memory_usage = sum(r.memory_usage_peak_mb for r in results) / len(results) if results else 0
            
            report = {
                "processing_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "input_directory": input_path,
                    "output_directory": output_path,
                    "total_series_processed": len(results),
                    "successful_series": len(successful_results),
                    "failed_series": len(failed_results),
                    "total_stl_files_generated": total_stl_files,
                    "total_processing_time_seconds": total_time,
                    "average_memory_usage_mb": avg_memory_usage
                },
                "successful_series": [],
                "failed_series": []
            }
            
            # Add successful series details
            for result in successful_results:
                series_info = {
                    "series_uid": result.dicom_series.series_uid if result.dicom_series else "unknown",
                    "processing_time": result.total_processing_time,
                    "detected_teeth": result.segmentation_result.detected_teeth_count if result.segmentation_result else 0,
                    "stl_files_generated": len(result.stl_files),
                    "memory_usage_mb": result.memory_usage_peak_mb
                }
                report["successful_series"].append(series_info)
            
            # Add failed series details
            for result in failed_results:
                series_info = {
                    "series_uid": result.dicom_series.series_uid if result.dicom_series else "unknown",
                    "error_message": result.error_message,
                    "processing_time": result.total_processing_time
                }
                report["failed_series"].append(series_info)
            
            # Write report file
            report_path = Path(output_path) / f"processing_report_{int(time.time())}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Processing report saved to: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate processing report: {e}")
    
    def get_processing_status(self, job_id: str) -> Optional[ProcessingJob]:
        """
        Get processing status for a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            ProcessingJob or None if not found
        """
        return self.active_jobs.get(job_id)
    
    def get_all_active_jobs(self) -> List[ProcessingJob]:
        """
        Get all active processing jobs.
        
        Returns:
            List of active ProcessingJob objects
        """
        return list(self.active_jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = ProcessingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            self.logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False


class MemoryMonitor:
    """Simple memory usage monitor for a processing session."""
    
    def __init__(self):
        self.start_memory = psutil.virtual_memory().used
        self.peak_memory = self.start_memory
    
    def update(self):
        """Update peak memory usage."""
        current_memory = psutil.virtual_memory().used
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB."""
        self.update()
        peak_mb = (self.peak_memory - self.start_memory) / (1024 * 1024)
        return max(0.0, peak_mb)  # Return 0 if negative