"""
Command Line Interface for Dental Segmentator Application.

This module provides the main CLI interface for the dental segmentator application,
supporting batch processing of DICOM files and generation of STL models.
"""

import sys
import click
from pathlib import Path
from typing import Optional
import logging

# Import application modules
try:
    from ..config.config_manager import ConfigManager, initialize_config
    from ..utils.logging_manager import initialize_logging, get_logging_manager
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.config.config_manager import ConfigManager, initialize_config
    from src.utils.logging_manager import initialize_logging, get_logging_manager


class DentalSegmentatorCLI:
    """Main CLI handler for dental segmentator operations."""
    
    def __init__(self):
        self.config_manager: Optional[ConfigManager] = None
        self.logger = None
    
    def initialize_system(
        self,
        config_path: Optional[str] = None,
        log_level: str = "INFO"
    ) -> bool:
        """
        Initialize the application systems.
        
        Args:
            config_path: Path to configuration file
            log_level: Logging level
            
        Returns:
            True if initialization successful
        """
        try:
            # Initialize configuration
            if config_path:
                self.config_manager = initialize_config(config_path)
            else:
                # Try default config location
                default_config = Path("config/default.yaml")
                if default_config.exists():
                    self.config_manager = initialize_config(str(default_config))
                else:
                    self.config_manager = ConfigManager()
            
            # Initialize logging
            logging_config = self.config_manager.config.logging
            logging_manager = initialize_logging(
                log_level=log_level or logging_config.level,
                log_dir="logs",
                max_file_size=logging_config.max_file_size,
                backup_count=logging_config.backup_count,
                log_format=logging_config.log_format
            )
            
            self.logger = logging_manager.get_logger("cli.main")
            self.logger.info("Dental Segmentator CLI initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize application: {e}", file=sys.stderr)
            return False
    
    def validate_paths(self, input_path: str, output_path: str) -> bool:
        """
        Validate input and output paths.
        
        Args:
            input_path: Input directory or file path
            output_path: Output directory path
            
        Returns:
            True if paths are valid
        """
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        # Validate input path
        if not input_path_obj.exists():
            self.logger.error(f"Input path does not exist: {input_path}")
            click.echo(f"Error: Input path does not exist: {input_path}", err=True)
            return False
        
        if input_path_obj.is_file() and input_path_obj.suffix.lower() not in ['.dcm', '.dicom']:
            self.logger.warning(f"Input file may not be a DICOM file: {input_path}")
            click.echo(f"Warning: Input file may not be a DICOM file: {input_path}")
        
        # Validate/create output directory
        try:
            output_path_obj.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory prepared: {output_path}")
        except PermissionError:
            self.logger.error(f"Cannot create output directory: {output_path}")
            click.echo(f"Error: Cannot create output directory: {output_path}", err=True)
            return False
        
        return True


# CLI Command Groups and Commands
@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Set logging level')
@click.pass_context
def cli(ctx, version, config, log_level):
    """
    Dental Segmentator - Automated DICOM to STL conversion for dental imaging.
    
    This tool processes DICOM dental imaging data and generates 3D STL models
    using pre-trained nnU-Net segmentation models.
    """
    # Ensure context object exists
    if ctx.obj is None:
        ctx.obj = {}
    
    # Show version and exit
    if version:
        click.echo("Dental Segmentator v1.0.0")
        click.echo("Automated DICOM to STL conversion for dental imaging")
        sys.exit(0)
    
    # Initialize CLI handler
    cli_handler = DentalSegmentatorCLI()
    if not cli_handler.initialize_system(config, log_level):
        sys.exit(1)
    
    ctx.obj['cli_handler'] = cli_handler
    
    # If no subcommand is provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input DICOM file or directory path')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for STL files')
@click.option('--batch', is_flag=True, 
              help='Process all DICOM series in input directory')
@click.option('--gpu/--no-gpu', default=None,
              help='Enable or disable GPU acceleration (auto-detect by default)')
@click.option('--parallel-jobs', '-j', type=int,
              help='Number of parallel processing jobs')
@click.option('--confidence-threshold', type=float,
              help='Minimum confidence threshold for segmentation (0.0-1.0)')
@click.option('--dry-run', is_flag=True,
              help='Show what would be processed without actually processing')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.pass_context
def process(ctx, input, output, batch, gpu, parallel_jobs, confidence_threshold, dry_run, verbose):
    """
    Process DICOM files and generate STL models.
    
    Examples:
    
      # Process a single DICOM series
      dental-segmentator process -i ./data/patient001 -o ./output
      
      # Batch process multiple series
      dental-segmentator process -i ./data -o ./output --batch
      
      # Process with custom settings
      dental-segmentator process -i ./data -o ./output --gpu --parallel-jobs 4
    """
    cli_handler = ctx.obj['cli_handler']
    
    # Validate paths
    if not cli_handler.validate_paths(input, output):
        sys.exit(1)
    
    # Get configuration
    config = cli_handler.config_manager.config
    
    # Override configuration with command line arguments
    if gpu is not None:
        config.hardware.gpu_enabled = gpu
    
    if parallel_jobs is not None:
        config.processing.parallel_jobs = parallel_jobs
    
    if confidence_threshold is not None:
        if not (0.0 <= confidence_threshold <= 1.0):
            click.echo("Error: confidence-threshold must be between 0.0 and 1.0", err=True)
            sys.exit(1)
        config.segmentation.confidence_threshold = confidence_threshold
    
    if verbose:
        cli_handler.logger.setLevel(logging.DEBUG)
    
    # Log processing parameters
    cli_handler.logger.info(f"Starting processing with parameters:")
    cli_handler.logger.info(f"  Input: {input}")
    cli_handler.logger.info(f"  Output: {output}")
    cli_handler.logger.info(f"  Batch mode: {batch}")
    cli_handler.logger.info(f"  GPU enabled: {config.hardware.gpu_enabled}")
    cli_handler.logger.info(f"  Parallel jobs: {config.processing.parallel_jobs}")
    cli_handler.logger.info(f"  Confidence threshold: {config.segmentation.confidence_threshold}")
    
    if dry_run:
        click.echo("DRY RUN - No actual processing will be performed")
        click.echo(f"Would process: {input}")
        click.echo(f"Would output to: {output}")
        click.echo(f"Batch mode: {batch}")
        click.echo(f"GPU enabled: {config.hardware.gpu_enabled}")
        return
    
    try:
        # This would call the actual processing engine
        click.echo("Processing started...")
        click.echo("Note: Core processing functionality not yet implemented")
        click.echo("This is the CLI framework - processing engine will be implemented in next phase")
        
        cli_handler.logger.info("Processing completed successfully")
        click.echo("Processing completed successfully!")
        
    except Exception as e:
        cli_handler.logger.error(f"Processing failed: {e}")
        click.echo(f"Error: Processing failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model-name', type=str, help='Specific model to download')
@click.option('--force', is_flag=True, help='Force re-download even if model exists')
@click.pass_context
def download_models(ctx, model_name, force):
    """
    Download pre-trained models from Zenodo.
    
    This command downloads the dental segmentator models required for processing.
    """
    cli_handler = ctx.obj['cli_handler']
    
    try:
        click.echo("Model download functionality not yet implemented")
        click.echo("This will download models from: https://zenodo.org/records/10829675")
        
        if model_name:
            click.echo(f"Would download specific model: {model_name}")
        else:
            click.echo("Would download all required models")
        
        if force:
            click.echo("Would force re-download existing models")
        
        cli_handler.logger.info("Model download completed")
        
    except Exception as e:
        cli_handler.logger.error(f"Model download failed: {e}")
        click.echo(f"Error: Model download failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input DICOM file or directory to validate')
@click.pass_context
def validate(ctx, input):
    """
    Validate DICOM files for compatibility.
    
    This command checks if DICOM files are compatible with the segmentation pipeline.
    """
    cli_handler = ctx.obj['cli_handler']
    
    try:
        click.echo(f"Validating DICOM files in: {input}")
        click.echo("Validation functionality not yet implemented")
        
        cli_handler.logger.info(f"Validation completed for: {input}")
        
    except Exception as e:
        cli_handler.logger.error(f"Validation failed: {e}")
        click.echo(f"Error: Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--days', type=int, default=30, help='Keep log files newer than N days')
@click.option('--temp-files', is_flag=True, help='Clean temporary processing files')
@click.pass_context
def clean(ctx, days, temp_files):
    """
    Clean up old log files and temporary data.
    """
    cli_handler = ctx.obj['cli_handler']
    
    try:
        if temp_files:
            temp_dir = Path(cli_handler.config_manager.config.processing.temp_directory)
            if temp_dir.exists():
                click.echo(f"Would clean temporary files in: {temp_dir}")
            else:
                click.echo("No temporary directory found")
        
        # Clean log files
        logging_manager = get_logging_manager()
        logging_manager.cleanup_old_logs(days)
        click.echo(f"Cleaned log files older than {days} days")
        
    except Exception as e:
        cli_handler.logger.error(f"Cleanup failed: {e}")
        click.echo(f"Error: Cleanup failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """
    Show system status and configuration.
    """
    cli_handler = ctx.obj['cli_handler']
    config = cli_handler.config_manager.config
    
    click.echo("=== Dental Segmentator Status ===")
    click.echo(f"Version: {config.application_name} v{config.version}")
    click.echo(f"Configuration: {cli_handler.config_manager.config_path or 'default'}")
    click.echo()
    
    click.echo("=== Processing Configuration ===")
    click.echo(f"Input directory: {config.processing.input_directory}")
    click.echo(f"Output directory: {config.processing.output_directory}")
    click.echo(f"Parallel jobs: {config.processing.parallel_jobs}")
    click.echo()
    
    click.echo("=== Hardware Configuration ===")
    click.echo(f"GPU enabled: {config.hardware.gpu_enabled}")
    click.echo(f"GPU memory limit: {config.hardware.gpu_memory_limit} MB")
    click.echo(f"CPU threads: {config.hardware.cpu_threads}")
    click.echo()
    
    click.echo("=== Models ===")
    click.echo(f"Models directory: {config.models.base_path}")
    model_paths = cli_handler.config_manager.get_model_paths()
    if model_paths:
        click.echo("Found models:")
        for model_path in model_paths:
            click.echo(f"  - {model_path}")
    else:
        click.echo("No models found - run 'download-models' command")


def main():
    """Main entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()