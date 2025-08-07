"""
Configuration Manager for Dental Segmentator Application.

This module provides comprehensive configuration management including:
- YAML configuration file loading and validation
- Environment variable integration
- Default value management
- Configuration schema validation
- Runtime configuration updates
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging


@dataclass
class SegmentationParams:
    """Configuration parameters for dental segmentation."""
    model_name: str = "dental_segmentator_v1"
    confidence_threshold: float = 0.5
    post_processing: bool = True
    mesh_simplification: bool = True
    smoothing_iterations: int = 5
    simplification_ratio: float = 0.1


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations."""
    input_directory: str = "./data/input"
    output_directory: str = "./data/output"
    temp_directory: str = "./data/temp"
    parallel_jobs: int = 2
    batch_size: int = 1
    cleanup_temp_files: bool = True


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    base_path: str = "./models"
    dental_segmentator_url: str = "https://zenodo.org/records/10829675/files/model.zip"
    dental_segmentator_checksum: str = ""
    auto_download: bool = True


@dataclass
class HardwareConfig:
    """Configuration for hardware utilization."""
    gpu_enabled: bool = True
    gpu_memory_limit: int = 8192  # MB
    cpu_threads: int = -1  # Auto-detect
    memory_limit: int = 16384  # MB


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    max_file_size: str = "100MB"
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    type: str = "sqlite"
    path: str = "./database/dental_segmentator.db"
    connection_timeout: int = 30


@dataclass
class SecurityConfig:
    """Configuration for security and privacy."""
    anonymize_dicom_tags: bool = True
    secure_temp_cleanup: bool = True
    log_sensitive_info: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    application_name: str = "dental-segmentator"
    version: str = "1.0.0"
    
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigManager:
    """
    Configuration manager for the dental segmentator application.
    
    Features:
    - Load configuration from YAML files
    - Environment variable override support
    - Configuration validation
    - Default value management
    - Runtime configuration updates
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: AppConfig = AppConfig()
        self._env_prefix = "DENTAL_SEGMENTATOR_"
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        
        # Apply environment variable overrides
        self._apply_environment_overrides()
    
    def load_config(self, config_path: str) -> AppConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded configuration
            
        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ConfigurationError(f"Empty configuration file: {config_path}")
            
            # Validate and create configuration
            self.config = self._create_config_from_dict(config_data)
            self.config_path = config_path
            
            # Validate the loaded configuration
            self._validate_config()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> AppConfig:
        """
        Create AppConfig from dictionary data.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            AppConfig instance
        """
        # Create sub-configurations
        processing_config = ProcessingConfig()
        if 'processing' in config_data:
            processing_data = config_data['processing']
            for key, value in processing_data.items():
                if hasattr(processing_config, key):
                    setattr(processing_config, key, value)
        
        models_config = ModelConfig()
        if 'models' in config_data:
            models_data = config_data['models']
            for key, value in models_data.items():
                if hasattr(models_config, key):
                    setattr(models_config, key, value)
        
        segmentation_config = SegmentationParams()
        if 'segmentation' in config_data:
            seg_data = config_data['segmentation']
            for key, value in seg_data.items():
                if hasattr(segmentation_config, key):
                    setattr(segmentation_config, key, value)
            
            # Handle nested mesh optimization settings
            if 'mesh_optimization' in seg_data:
                mesh_opts = seg_data['mesh_optimization']
                if 'enable_smoothing' in mesh_opts:
                    # This would be handled differently in actual implementation
                    pass
                if 'smoothing_iterations' in mesh_opts:
                    segmentation_config.smoothing_iterations = mesh_opts['smoothing_iterations']
                if 'enable_simplification' in mesh_opts:
                    segmentation_config.mesh_simplification = mesh_opts['enable_simplification']
                if 'simplification_ratio' in mesh_opts:
                    segmentation_config.simplification_ratio = mesh_opts['simplification_ratio']
        
        hardware_config = HardwareConfig()
        if 'hardware' in config_data:
            hw_data = config_data['hardware']
            for key, value in hw_data.items():
                if hasattr(hardware_config, key):
                    setattr(hardware_config, key, value)
        
        logging_config = LoggingConfig()
        if 'logging' in config_data:
            log_data = config_data['logging']
            for key, value in log_data.items():
                if hasattr(logging_config, key):
                    setattr(logging_config, key, value)
        
        database_config = DatabaseConfig()
        if 'database' in config_data:
            db_data = config_data['database']
            for key, value in db_data.items():
                if hasattr(database_config, key):
                    setattr(database_config, key, value)
        
        security_config = SecurityConfig()
        if 'security' in config_data:
            sec_data = config_data['security']
            for key, value in sec_data.items():
                if hasattr(security_config, key):
                    setattr(security_config, key, value)
        
        # Create main config
        app_config = AppConfig(
            processing=processing_config,
            models=models_config,
            segmentation=segmentation_config,
            hardware=hardware_config,
            logging=logging_config,
            database=database_config,
            security=security_config
        )
        
        # Set application-level settings
        if 'application' in config_data:
            app_data = config_data['application']
            if 'name' in app_data:
                app_config.application_name = app_data['name']
            if 'version' in app_data:
                app_config.version = app_data['version']
        
        return app_config
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            f"{self._env_prefix}INPUT_DIR": ("processing", "input_directory"),
            f"{self._env_prefix}OUTPUT_DIR": ("processing", "output_directory"),
            f"{self._env_prefix}LOG_LEVEL": ("logging", "level"),
            f"{self._env_prefix}GPU_ENABLED": ("hardware", "gpu_enabled"),
            f"{self._env_prefix}GPU_MEMORY_LIMIT": ("hardware", "gpu_memory_limit"),
            f"{self._env_prefix}PARALLEL_JOBS": ("processing", "parallel_jobs"),
            f"{self._env_prefix}MODEL_PATH": ("models", "base_path"),
            f"{self._env_prefix}DB_PATH": ("database", "path"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(self.config, section)
                current_value = getattr(section_obj, key)
                
                # Type conversion based on current value type
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(section_obj, key, value)
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.
        
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate paths
        self._validate_paths()
        
        # Validate numeric values
        self._validate_numeric_values()
        
        # Validate enum-like values
        self._validate_enum_values()
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_paths(self) -> None:
        """Validate file and directory paths."""
        # Check if parent directories exist for required paths
        paths_to_check = [
            (self.config.processing.input_directory, "Input directory parent"),
            (self.config.processing.output_directory, "Output directory parent"),
            (self.config.processing.temp_directory, "Temp directory parent"),
            (self.config.models.base_path, "Models directory parent"),
            (self.config.database.path, "Database file parent directory"),
        ]
        
        for path_str, description in paths_to_check:
            path = Path(path_str)
            parent_dir = path.parent if path.suffix else path
            
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ConfigurationError(
                    f"{description} is not writable: {parent_dir}"
                )
    
    def _validate_numeric_values(self) -> None:
        """Validate numeric configuration values."""
        if self.config.processing.parallel_jobs < 1:
            raise ConfigurationError("parallel_jobs must be >= 1")
        
        if not (0.0 <= self.config.segmentation.confidence_threshold <= 1.0):
            raise ConfigurationError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.config.segmentation.smoothing_iterations < 0:
            raise ConfigurationError("smoothing_iterations must be >= 0")
        
        if not (0.0 < self.config.segmentation.simplification_ratio <= 1.0):
            raise ConfigurationError("simplification_ratio must be between 0.0 and 1.0")
        
        if self.config.hardware.gpu_memory_limit < 1024:
            raise ConfigurationError("gpu_memory_limit must be >= 1024 MB")
        
        if self.config.hardware.memory_limit < 2048:
            raise ConfigurationError("memory_limit must be >= 2048 MB")
    
    def _validate_enum_values(self) -> None:
        """Validate enumeration-like values."""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if self.config.logging.level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.config.logging.level}. "
                f"Must be one of: {valid_log_levels}"
            )
        
        valid_db_types = ['sqlite']
        if self.config.database.type.lower() not in valid_db_types:
            raise ConfigurationError(
                f"Invalid database type: {self.config.database.type}. "
                f"Must be one of: {valid_db_types}"
            )
    
    def _validate_dependencies(self) -> None:
        """Validate configuration dependencies."""
        # If GPU is enabled, check if CUDA is available (in actual implementation)
        if self.config.hardware.gpu_enabled:
            # This would check torch.cuda.is_available() in actual implementation
            pass
    
    def get_model_paths(self) -> List[str]:
        """
        Get list of model file paths.
        
        Returns:
            List of model file paths
        """
        model_base = Path(self.config.models.base_path)
        model_paths = []
        
        # Look for dental segmentator model
        dental_model_path = model_base / "dental_segmentator"
        if dental_model_path.exists():
            # Find model files (typically .pth or .pkl files)
            for model_file in dental_model_path.rglob("*.pth"):
                model_paths.append(str(model_file))
            for model_file in dental_model_path.rglob("*.pkl"):
                model_paths.append(str(model_file))
        
        return model_paths
    
    def update_config_value(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value at runtime.
        
        Args:
            section: Configuration section name
            key: Configuration key
            value: New value
            
        Raises:
            ConfigurationError: If section or key is invalid
        """
        if not hasattr(self.config, section):
            raise ConfigurationError(f"Invalid configuration section: {section}")
        
        section_obj = getattr(self.config, section)
        if not hasattr(section_obj, key):
            raise ConfigurationError(f"Invalid configuration key: {section}.{key}")
        
        setattr(section_obj, key, value)
        
        # Re-validate after update
        self._validate_config()
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'application': {
                'name': self.config.application_name,
                'version': self.config.version
            },
            'processing': {
                'input_directory': self.config.processing.input_directory,
                'output_directory': self.config.processing.output_directory,
                'temp_directory': self.config.processing.temp_directory,
                'parallel_jobs': self.config.processing.parallel_jobs,
                'batch_size': self.config.processing.batch_size,
                'cleanup_temp_files': self.config.processing.cleanup_temp_files
            },
            'models': {
                'base_path': self.config.models.base_path,
                'dental_segmentator_url': self.config.models.dental_segmentator_url,
                'dental_segmentator_checksum': self.config.models.dental_segmentator_checksum,
                'auto_download': self.config.models.auto_download
            },
            'segmentation': {
                'model_name': self.config.segmentation.model_name,
                'confidence_threshold': self.config.segmentation.confidence_threshold,
                'post_processing': self.config.segmentation.post_processing,
                'mesh_optimization': {
                    'enable_smoothing': True,  # Derived from smoothing_iterations > 0
                    'smoothing_iterations': self.config.segmentation.smoothing_iterations,
                    'enable_simplification': self.config.segmentation.mesh_simplification,
                    'simplification_ratio': self.config.segmentation.simplification_ratio
                }
            },
            'hardware': {
                'gpu_enabled': self.config.hardware.gpu_enabled,
                'gpu_memory_limit': self.config.hardware.gpu_memory_limit,
                'cpu_threads': self.config.hardware.cpu_threads,
                'memory_limit': self.config.hardware.memory_limit
            },
            'logging': {
                'level': self.config.logging.level,
                'max_file_size': self.config.logging.max_file_size,
                'backup_count': self.config.logging.backup_count,
                'format': self.config.logging.log_format
            },
            'database': {
                'type': self.config.database.type,
                'path': self.config.database.path,
                'connection_timeout': self.config.database.connection_timeout
            },
            'security': {
                'anonymize_dicom_tags': self.config.security.anonymize_dicom_tags,
                'secure_temp_cleanup': self.config.security.secure_temp_cleanup,
                'log_sensitive_info': self.config.security.log_sensitive_info
            }
        }


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Initialize the global configuration manager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager