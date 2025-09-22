"""
Command-line interface for QBES.
"""

import click
import os
import sys
from typing import Optional

from .config_manager import ConfigurationManager
from .simulation_engine import SimulationEngine
from .utils.file_io import FileIOUtils


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Quantum Biological Environment Simulator (QBES) command-line interface."""
    pass


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default=None, help='Output directory (overrides config)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running simulation')
def run(config_file: str, output_dir: Optional[str], verbose: bool, dry_run: bool):
    """Run a QBES simulation with the specified configuration file."""
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_file)
        
        # Override output directory if specified
        if output_dir:
            config.output_directory = output_dir
        
        if verbose:
            click.echo(f"Loaded configuration from: {config_file}")
            click.echo(f"Output directory: {config.output_directory}")
        
        # Validate configuration
        validation_result = config_manager.validate_parameters(config)
        if not validation_result.is_valid:
            click.echo("Configuration validation failed:", err=True)
            for error in validation_result.errors:
                click.echo(f"  Error: {error}", err=True)
            sys.exit(1)
        
        if validation_result.warnings and verbose:
            for warning in validation_result.warnings:
                click.echo(f"  Warning: {warning}")
        
        if dry_run:
            click.echo("Configuration is valid. Dry run complete.")
            return
        
        # Initialize and run simulation
        engine = SimulationEngine()
        init_result = engine.initialize_simulation(config)
        
        if not init_result.is_valid:
            click.echo("Simulation initialization failed:", err=True)
            for error in init_result.errors:
                click.echo(f"  Error: {error}", err=True)
            sys.exit(1)
        
        click.echo("Starting simulation...")
        results = engine.run_simulation()
        
        click.echo(f"Simulation completed successfully!")
        click.echo(f"Results saved to: {config.output_directory}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('output_file', type=click.Path())
@click.option('--template', '-t', default='default', 
              help='Configuration template (default, photosystem, enzyme)')
def generate_config(output_file: str, template: str):
    """Generate a configuration file template."""
    try:
        config_manager = ConfigurationManager()
        success = config_manager.generate_default_config(output_file)
        
        if success:
            click.echo(f"Configuration template generated: {output_file}")
        else:
            click.echo("Failed to generate configuration template", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate(config_file: str):
    """Validate a configuration file."""
    try:
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_file)
        
        validation_result = config_manager.validate_parameters(config)
        
        if validation_result.is_valid:
            click.echo("✓ Configuration is valid")
            if validation_result.warnings:
                click.echo("\nWarnings:")
                for warning in validation_result.warnings:
                    click.echo(f"  • {warning}")
        else:
            click.echo("✗ Configuration validation failed")
            click.echo("\nErrors:")
            for error in validation_result.errors:
                click.echo(f"  • {error}")
            
            if validation_result.warnings:
                click.echo("\nWarnings:")
                for warning in validation_result.warnings:
                    click.echo(f"  • {warning}")
            
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.group()
def config_main():
    """Configuration management utilities."""
    pass


@config_main.command()
@click.argument('output_file', type=click.Path())
def generate(output_file: str):
    """Generate default configuration file."""
    generate_config.callback(output_file, 'default')


@config_main.command()
@click.argument('config_file', type=click.Path(exists=True))
def check(config_file: str):
    """Validate configuration file."""
    validate.callback(config_file)


@click.group()
def benchmark_main():
    """Benchmarking and validation utilities."""
    pass


@benchmark_main.command()
@click.option('--output-dir', '-o', default='./benchmark_results', 
              help='Output directory for benchmark results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def run_benchmarks(output_dir: str, verbose: bool):
    """Run the complete benchmark suite."""
    click.echo("Running QBES benchmark suite...")
    click.echo("This feature is not yet implemented.")
    # Placeholder for benchmark implementation


@benchmark_main.command()
@click.argument('results_dir', type=click.Path(exists=True))
def analyze(results_dir: str):
    """Analyze benchmark results."""
    click.echo(f"Analyzing benchmark results from: {results_dir}")
    click.echo("This feature is not yet implemented.")
    # Placeholder for benchmark analysis


if __name__ == '__main__':
    main()