"""
Unit tests for ConfigManager functionality.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import os

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager, ConfigurationError, AppConfig


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_config(self, config_data):
        """Create a test configuration file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def test_default_config_initialization(self):
        """Test initialization with default configuration."""
        config_manager = ConfigManager()
        
        assert config_manager.config is not None
        assert isinstance(config_manager.config, AppConfig)
        assert config_manager.config.application_name == "dental-segmentator"
        assert config_manager.config.processing.parallel_jobs == 2
        assert config_manager.config.hardware.gpu_enabled is True
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            'application': {
                'name': 'test-app',
                'version': '2.0.0'
            },
            'processing': {
                'parallel_jobs': 4,
                'batch_size': 2
            },
            'segmentation': {
                'confidence_threshold': 0.7
            }
        }
        
        self.create_test_config(config_data)
        config_manager = ConfigManager(str(self.config_path))
        
        assert config_manager.config.application_name == 'test-app'
        assert config_manager.config.version == '2.0.0'
        assert config_manager.config.processing.parallel_jobs == 4
        assert config_manager.config.processing.batch_size == 2
        assert config_manager.config.segmentation.confidence_threshold == 0.7
    
    def test_load_nonexistent_config(self):
        """Test loading a nonexistent configuration file."""
        with pytest.raises(ConfigurationError):
            ConfigManager("nonexistent.yaml")
    
    def test_invalid_yaml_config(self):
        """Test loading invalid YAML configuration."""
        with open(self.config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError):
            ConfigManager(str(self.config_path))
    
    def test_config_validation_parallel_jobs(self):
        """Test validation of parallel_jobs parameter."""
        config_data = {
            'processing': {
                'parallel_jobs': 0  # Invalid: must be >= 1
            }
        }
        
        self.create_test_config(config_data)
        
        with pytest.raises(ConfigurationError):
            ConfigManager(str(self.config_path))
    
    def test_config_validation_confidence_threshold(self):
        """Test validation of confidence_threshold parameter."""
        config_data = {
            'segmentation': {
                'confidence_threshold': 1.5  # Invalid: must be <= 1.0
            }
        }
        
        self.create_test_config(config_data)
        
        with pytest.raises(ConfigurationError):
            ConfigManager(str(self.config_path))
    
    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ["DENTAL_SEGMENTATOR_PARALLEL_JOBS"] = "8"
        os.environ["DENTAL_SEGMENTATOR_GPU_ENABLED"] = "false"
        
        try:
            config_manager = ConfigManager()
            
            assert config_manager.config.processing.parallel_jobs == 8
            assert config_manager.config.hardware.gpu_enabled is False
            
        finally:
            # Clean up environment variables
            del os.environ["DENTAL_SEGMENTATOR_PARALLEL_JOBS"]
            del os.environ["DENTAL_SEGMENTATOR_GPU_ENABLED"]
    
    def test_get_model_paths_empty(self):
        """Test getting model paths when no models exist."""
        config_manager = ConfigManager()
        model_paths = config_manager.get_model_paths()
        
        assert isinstance(model_paths, list)
        assert len(model_paths) == 0
    
    def test_update_config_value(self):
        """Test updating configuration values at runtime."""
        config_manager = ConfigManager()
        
        # Update valid configuration
        config_manager.update_config_value('processing', 'parallel_jobs', 6)
        assert config_manager.config.processing.parallel_jobs == 6
        
        # Test invalid section
        with pytest.raises(ConfigurationError):
            config_manager.update_config_value('invalid_section', 'key', 'value')
        
        # Test invalid key
        with pytest.raises(ConfigurationError):
            config_manager.update_config_value('processing', 'invalid_key', 'value')
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config_manager = ConfigManager()
        config_dict = config_manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'application' in config_dict
        assert 'processing' in config_dict
        assert 'segmentation' in config_dict
        assert config_dict['application']['name'] == 'dental-segmentator'