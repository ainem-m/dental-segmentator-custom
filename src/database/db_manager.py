"""
Database Manager for Dental Segmentator Application.

This module provides comprehensive database functionality including:
- SQLAlchemy ORM models for all data entities
- CRUD operations for DICOM series, segmentation results, and STL outputs
- Processing history tracking and statistics
- Database migrations and schema management
- Connection pooling and transaction management
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean,
    Text, JSON, ForeignKey, Index, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ..exceptions import DatabaseError

# SQLAlchemy Base
Base = declarative_base()

# Configure logging
logger = logging.getLogger(__name__)


class DICOMSeriesModel(Base):
    """Model for DICOM series information."""
    
    __tablename__ = 'dicom_series'
    
    id = Column(String, primary_key=True)
    series_uid = Column(String, unique=True, nullable=False, index=True)
    study_uid = Column(String, nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    modality = Column(String, nullable=False)
    file_paths = Column(JSON, nullable=False)  # List of file paths
    image_dimensions = Column(JSON, nullable=False)  # [width, height, depth]
    spacing = Column(JSON, nullable=False)  # [x_spacing, y_spacing, z_spacing]
    file_size_bytes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    segmentation_results = relationship("SegmentationResultModel", back_populates="dicom_series")
    
    # Indexes
    __table_args__ = (
        Index('idx_dicom_series_patient_modality', 'patient_id', 'modality'),
        Index('idx_dicom_series_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<DICOMSeries(id='{self.id}', series_uid='{self.series_uid}', modality='{self.modality}')>"


class SegmentationResultModel(Base):
    """Model for segmentation results."""
    
    __tablename__ = 'segmentation_results'
    
    id = Column(String, primary_key=True)
    dicom_series_id = Column(String, ForeignKey('dicom_series.id'), nullable=False)
    dicom_series_uid = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    confidence_scores = Column(JSON)  # List of confidence scores per tooth
    detected_teeth_count = Column(Integer)
    processing_time_seconds = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_used = Column(Boolean, default=False)
    status = Column(String, nullable=False, default='PENDING')  # PENDING, PROCESSING, SUCCESS, FAILED
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    dicom_series = relationship("DICOMSeriesModel", back_populates="segmentation_results")
    stl_outputs = relationship("STLOutputModel", back_populates="segmentation_result")
    
    # Indexes
    __table_args__ = (
        Index('idx_segmentation_status', 'status'),
        Index('idx_segmentation_created', 'created_at'),
        Index('idx_segmentation_model', 'model_version'),
    )
    
    def __repr__(self):
        return f"<SegmentationResult(id='{self.id}', status='{self.status}', teeth_count={self.detected_teeth_count})>"


class STLOutputModel(Base):
    """Model for STL output files."""
    
    __tablename__ = 'stl_outputs'
    
    id = Column(String, primary_key=True)
    segmentation_result_id = Column(String, ForeignKey('segmentation_results.id'), nullable=False)
    file_path = Column(String, nullable=False)
    file_size_bytes = Column(Integer)
    mesh_vertices = Column(Integer)
    mesh_faces = Column(Integer)
    anatomical_region = Column(String)  # e.g., "Upper Right Molar"
    tooth_label = Column(Integer)  # Numeric tooth label from segmentation
    is_watertight = Column(Boolean)
    quality_score = Column(Float)  # Mesh quality metric (0-1)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    segmentation_result = relationship("SegmentationResultModel", back_populates="stl_outputs")
    
    # Indexes
    __table_args__ = (
        Index('idx_stl_tooth_label', 'tooth_label'),
        Index('idx_stl_quality', 'quality_score'),
        Index('idx_stl_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<STLOutput(id='{self.id}', tooth_label={self.tooth_label}, quality={self.quality_score})>"


class ProcessingLogModel(Base):
    """Model for processing logs and events."""
    
    __tablename__ = 'processing_logs'
    
    id = Column(String, primary_key=True)
    level = Column(String, nullable=False)  # DEBUG, INFO, WARNING, ERROR
    component = Column(String, nullable=False)  # e.g., 'dicom_processor', 'segmentator'
    operation = Column(String, nullable=False)  # e.g., 'load_dicom', 'run_inference'
    message = Column(Text, nullable=False)
    context = Column(JSON)  # Additional context information
    dicom_series_uid = Column(String, index=True)
    segmentation_result_id = Column(String)
    duration_seconds = Column(Float)
    memory_usage_mb = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_log_level_component', 'level', 'component'),
        Index('idx_log_created', 'created_at'),
        Index('idx_log_series', 'dicom_series_uid'),
    )
    
    def __repr__(self):
        return f"<ProcessingLog(level='{self.level}', component='{self.component}', operation='{self.operation}')>"


class DatabaseManager:
    """
    Comprehensive database manager for the dental segmentator application.
    
    Provides high-level database operations including:
    - Connection management and pooling
    - CRUD operations for all models
    - Transaction management
    - Migration support
    - Statistics and reporting
    """
    
    def __init__(
        self,
        database_path: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False
    ):
        """
        Initialize database manager.
        
        Args:
            database_path: Path to SQLite database file
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            echo: Enable SQL query logging
        """
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database engine
        self.engine = create_engine(
            f"sqlite:///{self.database_path}",
            poolclass=StaticPool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo,
            connect_args={
                'check_same_thread': False,  # Allow multi-threading
                'timeout': 20  # Connection timeout
            }
        )
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"Database manager initialized with database: {self.database_path}")
    
    def _initialize_database(self):
        """Initialize database tables and schema."""
        try:
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Set up initial schema version
            with self.get_session() as session:
                # Check if we need to run initial setup
                result = session.execute("PRAGMA user_version").fetchone()
                if result[0] == 0:
                    # Set initial schema version
                    session.execute("PRAGMA user_version = 1")
                    session.commit()
                    logger.info("Database initialized with schema version 1")
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session with automatic cleanup.
        
        Yields:
            SQLAlchemy session
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database operation failed: {str(e)}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Database connections closed")
    
    # DICOM Series Operations
    
    def create_dicom_series(
        self,
        series_uid: str,
        study_uid: str,
        patient_id: str,
        modality: str,
        file_paths: List[str],
        image_dimensions: List[int],
        spacing: List[float],
        file_size_bytes: Optional[int] = None
    ) -> str:
        """
        Create new DICOM series record.
        
        Args:
            series_uid: Unique series identifier
            study_uid: Study identifier
            patient_id: Patient identifier
            modality: DICOM modality (CT, CBCT, etc.)
            file_paths: List of DICOM file paths
            image_dimensions: Image dimensions [width, height, depth]
            spacing: Voxel spacing [x, y, z]
            file_size_bytes: Total file size in bytes
            
        Returns:
            Created series record ID
            
        Raises:
            DatabaseError: If series creation fails
        """
        try:
            with self.get_session() as session:
                # Check for duplicate series_uid
                existing = session.query(DICOMSeriesModel).filter_by(series_uid=series_uid).first()
                if existing:
                    raise DatabaseError(f"DICOM series with UID {series_uid} already exists")
                
                # Generate unique ID
                import uuid
                series_id = str(uuid.uuid4())
                
                # Create new series record
                series = DICOMSeriesModel(
                    id=series_id,
                    series_uid=series_uid,
                    study_uid=study_uid,
                    patient_id=patient_id,
                    modality=modality,
                    file_paths=file_paths,
                    image_dimensions=image_dimensions,
                    spacing=spacing,
                    file_size_bytes=file_size_bytes
                )
                
                session.add(series)
                session.commit()
                
                logger.info(f"Created DICOM series record: {series_id}")
                return series_id
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to create DICOM series: {str(e)}")
    
    def get_dicom_series_by_uid(self, series_uid: str) -> Optional[DICOMSeriesModel]:
        """Get DICOM series by series UID."""
        try:
            with self.get_session() as session:
                return session.query(DICOMSeriesModel).filter_by(series_uid=series_uid).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve DICOM series: {str(e)}")
    
    def get_dicom_series(self, series_id: str) -> Optional[DICOMSeriesModel]:
        """Get DICOM series by ID."""
        try:
            with self.get_session() as session:
                return session.query(DICOMSeriesModel).filter_by(id=series_id).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve DICOM series: {str(e)}")
    
    # Segmentation Result Operations
    
    def create_segmentation_result(
        self,
        dicom_series_uid: str,
        model_version: str,
        model_path: str,
        confidence_scores: Optional[List[float]] = None,
        detected_teeth_count: Optional[int] = None,
        processing_time_seconds: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
        gpu_used: bool = False,
        status: str = 'SUCCESS'
    ) -> str:
        """
        Create segmentation result record.
        
        Args:
            dicom_series_uid: Associated DICOM series UID
            model_version: Segmentation model version
            model_path: Path to model files
            confidence_scores: List of confidence scores per tooth
            detected_teeth_count: Number of teeth detected
            processing_time_seconds: Processing time
            memory_usage_mb: Peak memory usage
            gpu_used: Whether GPU was used
            status: Processing status
            
        Returns:
            Created segmentation result ID
        """
        try:
            with self.get_session() as session:
                # Verify DICOM series exists
                series = session.query(DICOMSeriesModel).filter_by(series_uid=dicom_series_uid).first()
                if not series:
                    raise DatabaseError(f"DICOM series with UID {dicom_series_uid} not found")
                
                # Generate unique ID
                import uuid
                result_id = str(uuid.uuid4())
                
                # Create segmentation result
                result = SegmentationResultModel(
                    id=result_id,
                    dicom_series_id=series.id,
                    dicom_series_uid=dicom_series_uid,
                    model_version=model_version,
                    model_path=model_path,
                    confidence_scores=confidence_scores,
                    detected_teeth_count=detected_teeth_count,
                    processing_time_seconds=processing_time_seconds,
                    memory_usage_mb=memory_usage_mb,
                    gpu_used=gpu_used,
                    status=status,
                    completed_at=datetime.utcnow() if status in ['SUCCESS', 'FAILED'] else None
                )
                
                session.add(result)
                session.commit()
                
                logger.info(f"Created segmentation result: {result_id}")
                return result_id
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to create segmentation result: {str(e)}")
    
    def get_segmentation_result(self, result_id: str) -> Optional[SegmentationResultModel]:
        """Get segmentation result by ID."""
        try:
            with self.get_session() as session:
                return session.query(SegmentationResultModel).filter_by(id=result_id).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve segmentation result: {str(e)}")
    
    # STL Output Operations
    
    def create_stl_output(
        self,
        segmentation_result_id: str,
        file_path: str,
        file_size_bytes: Optional[int] = None,
        mesh_vertices: Optional[int] = None,
        mesh_faces: Optional[int] = None,
        anatomical_region: Optional[str] = None,
        tooth_label: Optional[int] = None,
        is_watertight: Optional[bool] = None,
        quality_score: Optional[float] = None
    ) -> str:
        """
        Create STL output record.
        
        Args:
            segmentation_result_id: Associated segmentation result ID
            file_path: Path to STL file
            file_size_bytes: File size in bytes
            mesh_vertices: Number of mesh vertices
            mesh_faces: Number of mesh faces
            anatomical_region: Anatomical region description
            tooth_label: Numeric tooth label
            is_watertight: Whether mesh is watertight
            quality_score: Mesh quality score (0-1)
            
        Returns:
            Created STL output ID
        """
        try:
            with self.get_session() as session:
                # Verify segmentation result exists
                seg_result = session.query(SegmentationResultModel).filter_by(id=segmentation_result_id).first()
                if not seg_result:
                    raise DatabaseError(f"Segmentation result {segmentation_result_id} not found")
                
                # Generate unique ID
                import uuid
                stl_id = str(uuid.uuid4())
                
                # Create STL output record
                stl_output = STLOutputModel(
                    id=stl_id,
                    segmentation_result_id=segmentation_result_id,
                    file_path=file_path,
                    file_size_bytes=file_size_bytes,
                    mesh_vertices=mesh_vertices,
                    mesh_faces=mesh_faces,
                    anatomical_region=anatomical_region,
                    tooth_label=tooth_label,
                    is_watertight=is_watertight,
                    quality_score=quality_score
                )
                
                session.add(stl_output)
                session.commit()
                
                logger.info(f"Created STL output record: {stl_id}")
                return stl_id
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to create STL output: {str(e)}")
    
    def get_stl_output(self, stl_id: str) -> Optional[STLOutputModel]:
        """Get STL output by ID."""
        try:
            with self.get_session() as session:
                return session.query(STLOutputModel).filter_by(id=stl_id).first()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve STL output: {str(e)}")
    
    # Processing History and Statistics
    
    def get_processing_history(
        self,
        patient_id: Optional[str] = None,
        modality: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DICOMSeriesModel]:
        """
        Get processing history.
        
        Args:
            patient_id: Filter by patient ID
            modality: Filter by modality
            limit: Maximum number of records
            offset: Record offset for pagination
            
        Returns:
            List of DICOM series records
        """
        try:
            with self.get_session() as session:
                query = session.query(DICOMSeriesModel).order_by(DICOMSeriesModel.created_at.desc())
                
                if patient_id:
                    query = query.filter(DICOMSeriesModel.patient_id == patient_id)
                if modality:
                    query = query.filter(DICOMSeriesModel.modality == modality)
                
                return query.offset(offset).limit(limit).all()
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve processing history: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with various statistics
        """
        try:
            with self.get_session() as session:
                stats = {}
                
                # Basic counts
                stats['total_series_processed'] = session.query(DICOMSeriesModel).count()
                stats['total_segmentation_results'] = session.query(SegmentationResultModel).count()
                stats['total_successful_segmentations'] = session.query(SegmentationResultModel).filter_by(status='SUCCESS').count()
                stats['total_stl_files_generated'] = session.query(STLOutputModel).count()
                
                # Processing time statistics
                avg_processing_time = session.query(func.avg(SegmentationResultModel.processing_time_seconds)).filter(
                    SegmentationResultModel.status == 'SUCCESS'
                ).scalar()
                stats['average_processing_time'] = float(avg_processing_time) if avg_processing_time else 0.0
                
                # Memory usage statistics
                avg_memory = session.query(func.avg(SegmentationResultModel.memory_usage_mb)).filter(
                    SegmentationResultModel.status == 'SUCCESS'
                ).scalar()
                stats['average_memory_usage_mb'] = float(avg_memory) if avg_memory else 0.0
                
                # Teeth detection statistics
                total_teeth = session.query(func.sum(SegmentationResultModel.detected_teeth_count)).filter(
                    SegmentationResultModel.status == 'SUCCESS'
                ).scalar()
                stats['total_teeth_detected'] = int(total_teeth) if total_teeth else 0
                
                # Modality breakdown
                modality_stats = session.query(
                    DICOMSeriesModel.modality,
                    func.count(DICOMSeriesModel.id)
                ).group_by(DICOMSeriesModel.modality).all()
                stats['modality_breakdown'] = {modality: count for modality, count in modality_stats}
                
                # Recent activity (last 30 days)
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                recent_series = session.query(DICOMSeriesModel).filter(
                    DICOMSeriesModel.created_at >= thirty_days_ago
                ).count()
                stats['series_processed_last_30_days'] = recent_series
                
                return stats
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve statistics: {str(e)}")
    
    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old processing records.
        
        Args:
            days_to_keep: Number of days to keep records
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            with self.get_session() as session:
                # Delete old processing logs
                old_logs = session.query(ProcessingLogModel).filter(
                    ProcessingLogModel.created_at < cutoff_date
                )
                deleted_count += old_logs.count()
                old_logs.delete()
                
                # Note: We typically don't delete DICOM series and segmentation results
                # as they represent valuable processing history
                
                session.commit()
                logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to cleanup old records: {str(e)}")
    
    def get_schema_version(self) -> int:
        """Get current database schema version."""
        try:
            with self.get_session() as session:
                result = session.execute("PRAGMA user_version").fetchone()
                return result[0] if result else 0
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get schema version: {str(e)}")
    
    def migrate_schema(self, target_version: int):
        """
        Migrate database schema to target version.
        
        Args:
            target_version: Target schema version
            
        Note: This is a placeholder for future migration logic
        """
        current_version = self.get_schema_version()
        
        if current_version == target_version:
            logger.info(f"Database already at version {target_version}")
            return
        
        logger.info(f"Migrating database from version {current_version} to {target_version}")
        
        # Migration logic would be implemented here
        # For now, just update the version number
        try:
            with self.get_session() as session:
                session.execute(f"PRAGMA user_version = {target_version}")
                session.commit()
                logger.info(f"Database migrated to version {target_version}")
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to migrate database: {str(e)}")


# Custom Exception for Database Operations
class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass