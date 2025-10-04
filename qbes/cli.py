"""
Command-line interface for QBES.
"""

import click
import os
import sys
import time
import json
import threading
from typing import Optional
from pathlib import Path

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
@click.option('--monitor', '-m', is_flag=True, help='Enable real-time progress monitoring')
@click.option('--checkpoint-interval', default=1000, type=int, help='Checkpoint save interval (steps)')
def run(config_file: str, output_dir: Optional[str], verbose: bool, dry_run: bool, 
        monitor: bool, checkpoint_interval: int):
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
        
        # Set checkpoint interval
        if hasattr(engine, 'checkpoint_interval'):
            engine.checkpoint_interval = checkpoint_interval
        
        click.echo("Starting simulation...")
        
        if monitor:
            # Run simulation with real-time monitoring
            results = _run_with_monitoring(engine, verbose)
        else:
            # Run simulation normally
            results = engine.run_simulation()
        
        click.echo(f"Simulation completed successfully!")
        click.echo(f"Results saved to: {config.output_directory}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _run_with_monitoring(engine: SimulationEngine, verbose: bool):
    """Run simulation with real-time progress monitoring."""
    
    def monitor_progress():
        """Monitor simulation progress in a separate thread."""
        last_progress = 0.0
        start_time = time.time()
        
        while engine.is_running:
            current_progress = engine.get_progress()
            
            if current_progress > last_progress + 5.0 or verbose:  # Update every 5%
                elapsed_time = time.time() - start_time
                if current_progress > 0:
                    estimated_total = elapsed_time * 100 / current_progress
                    remaining_time = estimated_total - elapsed_time
                    
                    click.echo(f"\rProgress: {current_progress:.1f}% | "
                             f"Elapsed: {elapsed_time:.1f}s | "
                             f"ETA: {remaining_time:.1f}s", nl=False)
                else:
                    click.echo(f"\rProgress: {current_progress:.1f}% | "
                             f"Elapsed: {elapsed_time:.1f}s", nl=False)
                
                last_progress = current_progress
            
            time.sleep(1.0)  # Check every second
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    try:
        # Run the simulation
        results = engine.run_simulation()
        click.echo()  # New line after progress updates
        return results
    except KeyboardInterrupt:
        click.echo("\n\nSimulation interrupted by user.")
        click.echo("Attempting to save current state...")
        
        # Try to save checkpoint
        try:
            checkpoint_file = os.path.join(engine.config.output_directory, "interrupt_checkpoint.pkl")
            engine.save_checkpoint(checkpoint_file)
            click.echo(f"Checkpoint saved to: {checkpoint_file}")
        except Exception as e:
            click.echo(f"Failed to save checkpoint: {e}")
        
        sys.exit(1)


@main.command()
@click.argument('checkpoint_file', type=click.Path(exists=True))
@click.option('--monitor', '-m', is_flag=True, help='Enable real-time progress monitoring')
def resume(checkpoint_file: str, monitor: bool):
    """Resume a simulation from a checkpoint file."""
    try:
        click.echo(f"Resuming simulation from: {checkpoint_file}")
        
        # Initialize engine and load checkpoint
        engine = SimulationEngine()
        success = engine.load_checkpoint(checkpoint_file)
        
        if not success:
            click.echo("Failed to load checkpoint file", err=True)
            sys.exit(1)
        
        click.echo(f"Checkpoint loaded. Resuming from {engine.get_progress():.1f}% completion...")
        
        if monitor:
            results = _run_with_monitoring(engine, False)
        else:
            results = engine.run_simulation()
        
        click.echo("Simulation completed successfully!")
        click.echo(f"Results saved to: {engine.config.output_directory}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('output_dir', type=click.Path())
@click.option('--interval', '-i', default=5.0, type=float, help='Update interval in seconds')
def monitor_sim(output_dir: str, interval: float):
    """Monitor a running simulation by watching output directory."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        click.echo(f"Output directory does not exist: {output_dir}", err=True)
        sys.exit(1)
    
    click.echo(f"Monitoring simulation in: {output_dir}")
    click.echo("Press Ctrl+C to stop monitoring")
    
    try:
        last_checkpoint_time = None
        
        while True:
            # Look for checkpoint files
            checkpoint_files = list(output_path.glob("checkpoint_*.pkl"))
            intermediate_files = list(output_path.glob("intermediate_*.json"))
            
            if intermediate_files:
                # Read latest intermediate results
                latest_file = max(intermediate_files, key=lambda f: f.stat().st_mtime)
                
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    progress = data.get('progress', 0.0)
                    step = data.get('step', 0)
                    current_time = data.get('time', 0.0)
                    
                    click.echo(f"\rStep: {step} | Time: {current_time:.2e}s | "
                             f"Progress: {progress:.1f}%", nl=False)
                    
                except Exception:
                    pass
            
            elif checkpoint_files:
                # Check checkpoint modification times
                latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
                checkpoint_time = latest_checkpoint.stat().st_mtime
                
                if last_checkpoint_time != checkpoint_time:
                    click.echo(f"\rCheckpoint updated: {latest_checkpoint.name}", nl=False)
                    last_checkpoint_time = checkpoint_time
            
            else:
                click.echo(f"\rWaiting for simulation data...", nl=False)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        click.echo("\nMonitoring stopped.")


def _generate_specialized_config(output_file: str, template: str) -> bool:
    """Generate specialized configuration templates."""
    import yaml
    
    templates = {
        'photosystem': {
            'system': {
                'pdb_file': 'photosystem.pdb',
                'force_field': 'amber14',
                'solvent_model': 'tip3p',
                'ionic_strength': 0.15
            },
            'simulation': {
                'temperature': 300.0,
                'simulation_time': 1.0e-12,  # 1 ps
                'time_step': 1.0e-15  # 1 fs
            },
            'quantum_subsystem': {
                'selection_method': 'chromophores',
                'custom_selection': 'resname CHL BCL',
                'max_quantum_atoms': 200
            },
            'noise_model': {
                'type': 'protein_ohmic',
                'coupling_strength': 2.0,
                'cutoff_frequency': 5.0e13,
                'reorganization_energy': 35.0  # cm^-1
            },
            'output': {
                'directory': './photosystem_output',
                'save_trajectory': True,
                'save_checkpoints': True,
                'checkpoint_interval': 500,
                'plot_format': 'png'
            }
        },
        'enzyme': {
            'system': {
                'pdb_file': 'enzyme.pdb',
                'force_field': 'amber14',
                'solvent_model': 'tip3p',
                'ionic_strength': 0.15
            },
            'simulation': {
                'temperature': 310.0,  # Body temperature
                'simulation_time': 5.0e-13,  # 500 fs
                'time_step': 5.0e-16  # 0.5 fs
            },
            'quantum_subsystem': {
                'selection_method': 'active_site',
                'custom_selection': 'resname HIS CYS ASP GLU SER THR',
                'max_quantum_atoms': 50
            },
            'noise_model': {
                'type': 'protein_ohmic',
                'coupling_strength': 1.5,
                'cutoff_frequency': 2.0e13,
                'reorganization_energy': 50.0  # cm^-1
            },
            'output': {
                'directory': './enzyme_output',
                'save_trajectory': True,
                'save_checkpoints': True,
                'checkpoint_interval': 1000,
                'plot_format': 'png'
            }
        },
        'membrane': {
            'system': {
                'pdb_file': 'membrane_protein.pdb',
                'force_field': 'charmm36',
                'solvent_model': 'tip3p',
                'ionic_strength': 0.15
            },
            'simulation': {
                'temperature': 300.0,
                'simulation_time': 2.0e-12,  # 2 ps
                'time_step': 2.0e-15  # 2 fs
            },
            'quantum_subsystem': {
                'selection_method': 'chromophores',
                'custom_selection': 'resname HEM CHL',
                'max_quantum_atoms': 100
            },
            'noise_model': {
                'type': 'membrane_lipid',
                'coupling_strength': 0.8,
                'cutoff_frequency': 1.0e13,
                'reorganization_energy': 25.0  # cm^-1
            },
            'output': {
                'directory': './membrane_output',
                'save_trajectory': True,
                'save_checkpoints': True,
                'checkpoint_interval': 800,
                'plot_format': 'png'
            }
        }
    }
    
    try:
        config = templates.get(template)
        if not config:
            return False
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return True
    except Exception:
        return False


@main.command()
@click.argument('output_dir', type=click.Path())
def status(output_dir: str):
    """Check the status of a simulation."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        click.echo(f"Output directory does not exist: {output_dir}", err=True)
        sys.exit(1)
    
    click.echo(f"Simulation Status for: {output_dir}")
    click.echo("=" * 50)
    
    # Check for various status indicators
    checkpoint_files = list(output_path.glob("checkpoint_*.pkl"))
    intermediate_files = list(output_path.glob("intermediate_*.json"))
    result_files = list(output_path.glob("simulation_results.json"))
    error_files = list(output_path.glob("error_*.log"))
    
    if result_files:
        click.echo("Status: COMPLETED ✓")
        latest_result = max(result_files, key=lambda f: f.stat().st_mtime)
        mod_time = time.ctime(latest_result.stat().st_mtime)
        click.echo(f"Completed: {mod_time}")
        
        # Try to read completion info
        try:
            with open(latest_result, 'r') as f:
                data = json.load(f)
            
            if 'simulation_time' in data:
                click.echo(f"Total simulation time: {data['simulation_time']:.2e} s")
            if 'final_step' in data:
                click.echo(f"Final step: {data['final_step']}")
                
        except Exception:
            pass
            
    elif error_files:
        click.echo("Status: FAILED ✗")
        latest_error = max(error_files, key=lambda f: f.stat().st_mtime)
        mod_time = time.ctime(latest_error.stat().st_mtime)
        click.echo(f"Failed: {mod_time}")
        
        # Show last few lines of error log
        try:
            with open(latest_error, 'r') as f:
                lines = f.readlines()
            
            click.echo("\nLast error messages:")
            for line in lines[-5:]:
                click.echo(f"  {line.strip()}")
                
        except Exception:
            pass
            
    elif intermediate_files:
        click.echo("Status: RUNNING ⏳")
        
        # Get latest progress
        latest_file = max(intermediate_files, key=lambda f: f.stat().st_mtime)
        mod_time = time.ctime(latest_file.stat().st_mtime)
        click.echo(f"Last update: {mod_time}")
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            progress = data.get('progress', 0.0)
            step = data.get('step', 0)
            current_time = data.get('time', 0.0)
            
            click.echo(f"Progress: {progress:.1f}%")
            click.echo(f"Current step: {step}")
            click.echo(f"Simulation time: {current_time:.2e} s")
            
        except Exception:
            pass
            
    elif checkpoint_files:
        click.echo("Status: PAUSED ⏸")
        latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        mod_time = time.ctime(latest_checkpoint.stat().st_mtime)
        click.echo(f"Last checkpoint: {mod_time}")
        click.echo(f"Checkpoint file: {latest_checkpoint.name}")
        
    else:
        click.echo("Status: NOT STARTED ⚪")
        click.echo("No simulation files found in output directory")
    
    # Show file counts
    click.echo(f"\nFiles found:")
    click.echo(f"  Checkpoints: {len(checkpoint_files)}")
    click.echo(f"  Intermediate results: {len(intermediate_files)}")
    click.echo(f"  Result files: {len(result_files)}")
    click.echo(f"  Error logs: {len(error_files)}")


@main.command()
@click.argument('output_file', type=click.Path())
@click.option('--template', '-t', 
              type=click.Choice(['default', 'photosystem', 'enzyme', 'membrane']),
              default='default', help='Configuration template type')
def generate_config(output_file: str, template: str):
    """Generate a configuration file template."""
    try:
        config_manager = ConfigurationManager()
        
        if template == 'default':
            success = config_manager.generate_default_config(output_file)
        else:
            success = _generate_specialized_config(output_file, template)
        
        if success:
            click.echo(f"Configuration template ({template}) generated: {output_file}")
            click.echo(f"Edit the file to customize parameters for your system.")
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
@click.option('--final-time', '-t', default=1.0, type=float,
              help='Final simulation time for benchmarks')
@click.option('--time-step', '-dt', default=0.01, type=float,
              help='Time step for benchmark simulations')
@click.option('--scaling-test', is_flag=True, 
              help='Run performance scaling analysis')
def run_benchmarks(output_dir: str, verbose: bool, final_time: float, 
                  time_step: float, scaling_test: bool):
    """Run the complete benchmark suite."""
    from .benchmarks.benchmark_systems import BenchmarkRunner
    import os
    
    click.echo("Running QBES benchmark suite...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize and run benchmarks
        runner = BenchmarkRunner()
        runner.add_standard_benchmarks()
        
        if verbose:
            click.echo(f"Running {len(runner.benchmarks)} benchmark tests...")
            click.echo(f"Final time: {final_time}")
            click.echo(f"Time step: {time_step}")
        
        # Run benchmarks
        results = runner.run_all_benchmarks(final_time=final_time, time_step=time_step)
        
        # Generate and save report
        report_file = os.path.join(output_dir, "benchmark_report.txt")
        runner.save_report(report_file)
        
        # Summary
        passed_tests = sum(1 for r in results if r.test_passed)
        total_tests = len(results)
        
        click.echo(f"\nBenchmark Results:")
        click.echo(f"  Total Tests: {total_tests}")
        click.echo(f"  Passed: {passed_tests}")
        click.echo(f"  Failed: {total_tests - passed_tests}")
        click.echo(f"  Success Rate: {100 * passed_tests / total_tests:.1f}%")
        click.echo(f"\nDetailed report saved to: {report_file}")
        
        # Run scaling test if requested
        if scaling_test:
            click.echo("\nRunning performance scaling analysis...")
            scaling_results = runner.performance_scaling_test()
            
            scaling_file = os.path.join(output_dir, "scaling_results.txt")
            with open(scaling_file, 'w') as f:
                f.write("Performance Scaling Results\n")
                f.write("=" * 30 + "\n\n")
                for size, time in scaling_results.items():
                    f.write(f"System Size: {size:3d} | Computation Time: {time:.3f}s\n")
            
            click.echo(f"Scaling results saved to: {scaling_file}")
        
        # Exit with error code if any tests failed
        if passed_tests < total_tests:
            click.echo("\nSome benchmark tests failed. Check the report for details.", err=True)
            sys.exit(1)
        else:
            click.echo("\nAll benchmark tests passed!")
            
    except Exception as e:
        click.echo(f"Error running benchmarks: {str(e)}", err=True)
        sys.exit(1)


@benchmark_main.command()
@click.argument('results_dir', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(results_dir: str, verbose: bool):
    """Analyze benchmark results from a previous run."""
    import glob
    import os
    
    click.echo(f"Analyzing benchmark results from: {results_dir}")
    
    # Look for benchmark report files
    report_files = glob.glob(os.path.join(results_dir, "*benchmark_report*.txt"))
    scaling_files = glob.glob(os.path.join(results_dir, "*scaling_results*.txt"))
    
    if not report_files:
        click.echo("No benchmark report files found in the specified directory.", err=True)
        sys.exit(1)
    
    # Display latest report
    latest_report = max(report_files, key=os.path.getctime)
    
    if verbose:
        click.echo(f"Reading report: {latest_report}")
    
    try:
        with open(latest_report, 'r') as f:
            report_content = f.read()
        
        click.echo("\n" + "=" * 60)
        click.echo(report_content)
        click.echo("=" * 60)
        
        # Display scaling results if available
        if scaling_files:
            latest_scaling = max(scaling_files, key=os.path.getctime)
            
            if verbose:
                click.echo(f"\nReading scaling results: {latest_scaling}")
            
            with open(latest_scaling, 'r') as f:
                scaling_content = f.read()
            
            click.echo("\n" + scaling_content)
        
    except Exception as e:
        click.echo(f"Error reading benchmark results: {str(e)}", err=True)
        sys.exit(1)


# Add command groups to main CLI
main.add_command(benchmark_main, name='benchmark')


if __name__ == '__main__':
    main()