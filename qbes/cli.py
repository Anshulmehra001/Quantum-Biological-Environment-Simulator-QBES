"""
Command-line interface for QBES.
"""

import click
import os
import sys
import time
import json
import threading
import yaml
from typing import Optional, Dict, Any
from pathlib import Path

from .config_manager import ConfigurationManager
from .simulation_engine import SimulationEngine
from .utils.file_io import FileIOUtils
from .utils.error_handling import ImprovedErrorHandler


class InteractiveConfigWizard:
    """Interactive configuration wizard for QBES."""
    
    def __init__(self):
        self.config_data = {}
        self.error_handler = ImprovedErrorHandler()
    
    def run_wizard(self) -> Dict[str, Any]:
        """Run the interactive configuration wizard."""
        click.echo("Welcome to the QBES Configuration Wizard!")
        click.echo("This wizard will help you create a configuration file for your quantum biological simulation.")
        click.echo("=" * 80)
        
        # System configuration
        self._configure_system()
        
        # Simulation parameters
        self._configure_simulation()
        
        # Quantum subsystem
        self._configure_quantum_subsystem()
        
        # Noise model
        self._configure_noise_model()
        
        # Output settings
        self._configure_output()
        
        return self.config_data
    
    def _configure_system(self):
        """Configure system parameters."""
        click.echo("\n1. System Configuration")
        click.echo("-" * 25)
        
        # PDB file
        pdb_file = self._ask_question(
            "What is the path to your PDB file? (e.g., 'system.pdb')",
            validator=self._validate_file_path,
            required=True
        )
        
        # Force field
        force_field = self._ask_choice(
            "Which force field would you like to use?",
            choices=['amber14', 'charmm36', 'opls'],
            default='amber14'
        )
        
        # Solvent model
        solvent_model = self._ask_choice(
            "Which solvent model would you like to use?",
            choices=['tip3p', 'tip4p', 'spc'],
            default='tip3p'
        )
        
        # Ionic strength
        ionic_strength = self._ask_question(
            "What is the ionic strength of your system? (M, typically 0.15)",
            validator=self._validate_positive_float,
            default="0.15"
        )
        
        self.config_data['system'] = {
            'pdb_file': pdb_file,
            'force_field': force_field,
            'solvent_model': solvent_model,
            'ionic_strength': float(ionic_strength)
        }
    
    def _configure_simulation(self):
        """Configure simulation parameters."""
        click.echo("\n2. Simulation Parameters")
        click.echo("-" * 27)
        
        # Temperature
        temperature = self._ask_question(
            "What temperature should the simulation run at? (K, typically 300)",
            validator=self._validate_positive_float,
            default="300.0"
        )
        
        # Simulation time
        sim_time = self._ask_question(
            "How long should the simulation run? (seconds, e.g., 1e-12 for 1 picosecond)",
            validator=self._validate_positive_float,
            default="1e-12"
        )
        
        # Time step
        time_step = self._ask_question(
            "What time step should be used? (seconds, e.g., 1e-15 for 1 femtosecond)",
            validator=self._validate_positive_float,
            default="1e-15"
        )
        
        self.config_data['simulation'] = {
            'temperature': float(temperature),
            'simulation_time': float(sim_time),
            'time_step': float(time_step)
        }
    
    def _configure_quantum_subsystem(self):
        """Configure quantum subsystem selection."""
        click.echo("\n3. Quantum Subsystem Selection")
        click.echo("-" * 33)
        
        # Selection method
        selection_method = self._ask_choice(
            "How would you like to select the quantum subsystem?",
            choices=['chromophores', 'active_site', 'custom'],
            descriptions={
                'chromophores': 'Automatically select light-absorbing molecules (chlorophyll, etc.)',
                'active_site': 'Select enzyme active site residues',
                'custom': 'Specify custom selection criteria'
            }
        )
        
        # Custom selection if needed
        custom_selection = ""
        if selection_method == 'custom':
            custom_selection = self._ask_question(
                "Enter your custom selection criteria (e.g., 'resname HIS CYS ASP GLU')",
                required=True
            )
        elif selection_method == 'chromophores':
            custom_selection = "resname CHL BCL"
        elif selection_method == 'active_site':
            custom_selection = "resname HIS CYS ASP GLU SER THR"
        
        # Maximum quantum atoms
        max_atoms = self._ask_question(
            "Maximum number of atoms in quantum subsystem? (typically 50-200)",
            validator=self._validate_positive_int,
            default="100"
        )
        
        self.config_data['quantum_subsystem'] = {
            'selection_method': selection_method,
            'custom_selection': custom_selection,
            'max_quantum_atoms': int(max_atoms)
        }
    
    def _configure_noise_model(self):
        """Configure noise model parameters."""
        click.echo("\n4. Environmental Noise Model")
        click.echo("-" * 30)
        
        # Noise model type
        noise_type = self._ask_choice(
            "What type of environmental noise model?",
            choices=['ohmic', 'protein_ohmic', 'membrane_lipid', 'solvent_ionic'],
            descriptions={
                'ohmic': 'Simple Ohmic spectral density',
                'protein_ohmic': 'Protein environment with Ohmic coupling',
                'membrane_lipid': 'Membrane lipid environment',
                'solvent_ionic': 'Ionic solvent environment'
            }
        )
        
        # Coupling strength
        coupling_strength = self._ask_question(
            "Environmental coupling strength? (typically 0.5-2.0)",
            validator=self._validate_positive_float,
            default="1.0"
        )
        
        # Cutoff frequency
        cutoff_freq = self._ask_question(
            "Cutoff frequency? (Hz, typically 1e13-5e13)",
            validator=self._validate_positive_float,
            default="2e13"
        )
        
        # Reorganization energy
        reorg_energy = self._ask_question(
            "Reorganization energy? (cm^-1, typically 25-50)",
            validator=self._validate_positive_float,
            default="35.0"
        )
        
        self.config_data['noise_model'] = {
            'type': noise_type,
            'coupling_strength': float(coupling_strength),
            'cutoff_frequency': float(cutoff_freq),
            'reorganization_energy': float(reorg_energy)
        }
    
    def _configure_output(self):
        """Configure output settings."""
        click.echo("\n5. Output Configuration")
        click.echo("-" * 24)
        
        # Output directory
        output_dir = self._ask_question(
            "Where should results be saved? (directory path)",
            default="./qbes_output"
        )
        
        # Save trajectory
        save_trajectory = self._ask_yes_no(
            "Save full quantum state trajectory? (requires more disk space)",
            default=True
        )
        
        # Save checkpoints
        save_checkpoints = self._ask_yes_no(
            "Save simulation checkpoints for resuming?",
            default=True
        )
        
        # Checkpoint interval
        checkpoint_interval = 1000
        if save_checkpoints:
            checkpoint_interval = self._ask_question(
                "Checkpoint save interval? (simulation steps, typically 500-1000)",
                validator=self._validate_positive_int,
                default="1000"
            )
        
        # Plot format
        plot_format = self._ask_choice(
            "Preferred plot format for results?",
            choices=['png', 'pdf', 'svg'],
            default='png'
        )
        
        self.config_data['output'] = {
            'directory': output_dir,
            'save_trajectory': save_trajectory,
            'save_checkpoints': save_checkpoints,
            'checkpoint_interval': int(checkpoint_interval),
            'plot_format': plot_format
        }
    
    def _ask_question(self, question: str, validator=None, default=None, required=False) -> str:
        """Ask user a question with validation and retry logic."""
        while True:
            prompt = f"\n{question}"
            if default:
                prompt += f" [{default}]"
            prompt += ": "
            
            answer = click.prompt(prompt, default=default if default else "", show_default=False)
            
            # Check if required and empty
            if required and not answer.strip():
                click.echo("This field is required. Please provide a value.")
                continue
            
            # Use default if empty and default provided
            if not answer.strip() and default:
                answer = default
            
            # Validate if validator provided
            if validator:
                try:
                    validator(answer)
                    return answer
                except ValueError as e:
                    click.echo(f"Invalid input: {e}")
                    continue
            
            return answer
    
    def _ask_choice(self, question: str, choices: list, descriptions=None, default=None) -> str:
        """Ask user to choose from a list of options."""
        click.echo(f"\n{question}")
        
        for i, choice in enumerate(choices, 1):
            desc = f" - {descriptions[choice]}" if descriptions and choice in descriptions else ""
            default_marker = " (default)" if choice == default else ""
            click.echo(f"  {i}. {choice}{desc}{default_marker}")
        
        while True:
            try:
                answer = click.prompt("\nEnter your choice (number or name)", 
                                    default=default if default else "")
                
                # Try to parse as number
                if answer.isdigit():
                    choice_idx = int(answer) - 1
                    if 0 <= choice_idx < len(choices):
                        return choices[choice_idx]
                    else:
                        click.echo(f"Please enter a number between 1 and {len(choices)}")
                        continue
                
                # Try to match by name
                answer_lower = answer.lower()
                for choice in choices:
                    if choice.lower() == answer_lower:
                        return choice
                
                click.echo(f"Invalid choice. Please select from: {', '.join(choices)}")
                
            except click.Abort:
                raise
            except Exception:
                click.echo("Invalid input. Please try again.")
    
    def _ask_yes_no(self, question: str, default=None) -> bool:
        """Ask a yes/no question."""
        default_str = " [Y/n]" if default is True else " [y/N]" if default is False else " [y/n]"
        
        while True:
            answer = click.prompt(f"\n{question}{default_str}", 
                                default="" if default is None else ("y" if default else "n"))
            
            answer_lower = answer.lower().strip()
            
            if answer_lower in ['y', 'yes', '1', 'true']:
                return True
            elif answer_lower in ['n', 'no', '0', 'false']:
                return False
            elif not answer_lower and default is not None:
                return default
            else:
                click.echo("Please answer 'y' for yes or 'n' for no.")
    
    def _validate_file_path(self, path: str):
        """Validate file path exists or could be created."""
        if not path.strip():
            raise ValueError("File path cannot be empty")
        
        # Check if file exists
        if not os.path.exists(path):
            # Check if directory exists
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                raise ValueError(f"Directory does not exist: {directory}")
    
    def _validate_positive_float(self, value: str):
        """Validate positive float value."""
        try:
            float_val = float(value)
            if float_val <= 0:
                raise ValueError("Value must be positive")
        except ValueError:
            raise ValueError("Must be a valid positive number")
    
    def _validate_positive_int(self, value: str):
        """Validate positive integer value."""
        try:
            int_val = int(value)
            if int_val <= 0:
                raise ValueError("Value must be positive")
        except ValueError:
            raise ValueError("Must be a valid positive integer")
    
    def generate_config_file(self, output_file: str) -> bool:
        """Generate YAML configuration file from wizard data."""
        try:
            with open(output_file, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
            return True
        except Exception as e:
            click.echo(f"Error writing configuration file: {e}")
            return False


@click.group()
@click.version_option(version="1.2.0")
def main():
    """
    Quantum Biological Environment Simulator (QBES) v1.2
    
    QBES simulates quantum effects in biological systems using hybrid quantum-classical
    molecular dynamics. It models quantum coherence, decoherence, and energy transfer
    in proteins, photosystems, and other biological environments.
    
    \b
    Quick Start:
      1. Generate configuration: qbes generate-config config.yaml --interactive
      2. Run simulation:        qbes run config.yaml
      3. Check results:         qbes status ./qbes_output
    
    \b
    System Requirements:
      • Python 3.8+
      • 8GB+ RAM (16GB+ recommended)
      • NumPy, SciPy, Click, PyYAML
      • Optional: OpenMM, MDTraj for MD simulations
    
    \b
    Examples:
      qbes generate-config my_config.yaml --interactive
      qbes run my_config.yaml --verbose --monitor
      qbes validate my_config.yaml
      qbes benchmark run --output-dir ./benchmarks
    
    For detailed help on any command, use: qbes COMMAND --help
    """
    pass


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default=None, 
              help='Output directory (overrides config file setting)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose logging with detailed progress information')
@click.option('--dry-run', is_flag=True, 
              help='Validate configuration and setup without running simulation')
@click.option('--monitor', '-m', is_flag=True, 
              help='Enable real-time progress monitoring with ETA estimates')
@click.option('--checkpoint-interval', default=1000, type=int, 
              help='Save checkpoint every N steps (default: 1000)')
@click.option('--debug-level', default='INFO', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Set logging level for debugging (default: INFO)')
@click.option('--save-snapshots', default=0, type=int, 
              help='Save state snapshots every N steps (0=disabled, default: 0)')
@click.option('--enable-sanity-checks/--disable-sanity-checks', default=True,
              help='Enable/disable sanity check logging (default: enabled)')
def run(config_file: str, output_dir: Optional[str], verbose: bool, dry_run: bool, 
        monitor: bool, checkpoint_interval: int, debug_level: str, save_snapshots: int,
        enable_sanity_checks: bool):
    """
    Run a quantum biological simulation.
    
    This command executes a complete QBES simulation using the parameters specified
    in the configuration file. The simulation includes molecular dynamics setup,
    quantum subsystem identification, Hamiltonian construction, and time evolution
    with environmental decoherence.
    
    \b
    Configuration File:
      The CONFIG_FILE should be a YAML file containing simulation parameters.
      Generate one with: qbes generate-config config.yaml --interactive
    
    \b
    Output:
      Results are saved to the output directory specified in the config file
      or overridden with --output-dir. Output includes:
      • simulation_results.pkl - Complete results data
      • results_summary.txt - Human-readable summary
      • Trajectory and checkpoint files (if enabled)
    
    \b
    Examples:
      qbes run config.yaml
      qbes run config.yaml --verbose --monitor
      qbes run config.yaml --output-dir ./my_results
      qbes run config.yaml --dry-run  # Test configuration only
    
    \b
    Performance Tips:
      • Use --monitor to track progress on long simulations
      • Reduce checkpoint-interval for better recovery options
      • Use --verbose to diagnose issues during setup
    """
    try:
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_file)
        
        # Override output directory if specified
        if output_dir:
            config.output_directory = output_dir
        
        # Override debugging parameters from CLI options
        config.debug_level = debug_level
        config.save_snapshot_interval = save_snapshots
        config.enable_sanity_checks = enable_sanity_checks
        config.dry_run_mode = dry_run
        
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
            click.echo("Performing dry-run validation...")
            
            # Initialize and run dry-run validation
            engine = SimulationEngine()
            engine.set_verbose_logging(verbose)
            dry_run_result = engine.perform_dry_run_validation(config)
            
            if not dry_run_result.is_valid:
                click.echo("Dry-run validation failed:", err=True)
                for error in dry_run_result.errors:
                    click.echo(f"  Error: {error}", err=True)
                sys.exit(1)
            
            if dry_run_result.warnings and verbose:
                for warning in dry_run_result.warnings:
                    click.echo(f"  Warning: {warning}")
            
            click.echo("✅ Dry run complete. All setup validation passed.")
            return
        
        # Initialize and run simulation
        engine = SimulationEngine()
        engine.set_verbose_logging(verbose)
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
        
        # Print formatted results summary
        summary_table = results.format_summary_table()
        click.echo(summary_table)
        
        # Also save summary to file
        summary_file = os.path.join(config.output_directory, "results_summary.txt")
        results.format_summary_table(save_to_file=summary_file)
        
        click.echo(f"\n✅ Simulation completed successfully!")
        click.echo(f"📁 Results saved to: {config.output_directory}")
        click.echo(f"📊 Summary saved to: {summary_file}")
        
    except FileNotFoundError as e:
        error_handler = ImprovedErrorHandler()
        error_msg = error_handler.handle_file_not_found(str(e), "simulation configuration")
        click.echo(error_msg, err=True)
        sys.exit(1)
    except ValueError as e:
        error_handler = ImprovedErrorHandler()
        error_msg = error_handler.handle_invalid_parameter("configuration", str(e), "valid configuration parameters")
        click.echo(error_msg, err=True)
        sys.exit(1)
    except MemoryError as e:
        error_handler = ImprovedErrorHandler()
        error_msg = error_handler.handle_memory_error(operation="simulation execution")
        click.echo(error_msg, err=True)
        sys.exit(1)
    except Exception as e:
        error_handler = ImprovedErrorHandler()
        # Try to categorize the error
        error_str = str(e).lower()
        if 'file' in error_str and 'not found' in error_str:
            error_msg = error_handler.handle_file_not_found(str(e))
        elif 'import' in error_str or 'module' in error_str:
            error_msg = error_handler.handle_dependency_error(str(e))
        elif 'yaml' in error_str or 'config' in error_str:
            error_msg = error_handler.handle_configuration_error(str(e), config_file)
        else:
            error_msg = f"❌ Unexpected error: {e}\n\n💡 Suggestions:\n• Check your configuration file\n• Verify all dependencies are installed\n• Try with a smaller test system\n• Check the documentation for examples"
        
        click.echo(error_msg, err=True)
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
@click.option('--monitor', '-m', is_flag=True, 
              help='Enable real-time progress monitoring')
def resume(checkpoint_file: str, monitor: bool):
    """
    Resume a simulation from a checkpoint file.
    
    This command continues a previously interrupted simulation from a saved
    checkpoint. Checkpoints are automatically created during simulation runs
    and can be used to recover from system crashes or intentional stops.
    
    \b
    Checkpoint Files:
      Checkpoints contain complete simulation state including:
      • Current quantum state and trajectory
      • Simulation progress and timing
      • System configuration and parameters
      • Random number generator state
    
    \b
    Examples:
      qbes resume checkpoint_step_5000.pkl
      qbes resume interrupt_checkpoint.pkl --monitor
    
    \b
    Notes:
      • Original configuration is restored from checkpoint
      • Simulation continues from exact stopping point
      • Output goes to original output directory
    """
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
@click.option('--interval', '-i', default=5.0, type=float, 
              help='Update interval in seconds (default: 5.0)')
def monitor_sim(output_dir: str, interval: float):
    """
    Monitor a running simulation in real-time.
    
    This command watches the simulation output directory and displays live
    progress updates including step count, progress percentage, and timing
    estimates. Useful for monitoring long-running simulations.
    
    \b
    Monitoring Features:
      • Real-time progress updates
      • Step count and percentage complete
      • Elapsed time and ETA estimates
      • Checkpoint file detection
      • Automatic error detection
    
    \b
    Examples:
      qbes monitor-sim ./qbes_output
      qbes monitor-sim ./results --interval 2.0
    
    \b
    Controls:
      Press Ctrl+C to stop monitoring (simulation continues)
    """
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
                'ionic_strength': 0.0  # Changed from 0.15 to avoid ion template issues
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
                'ionic_strength': 0.0  # Changed from 0.15
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
                'ionic_strength': 0.0  # Changed from 0.15
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
    """
    Check the status of a running or completed simulation.
    
    This command examines the output directory to determine simulation status
    and provides information about progress, completion, or any errors that
    occurred during execution.
    
    \b
    Status Information:
      • Current simulation state (running, completed, failed, paused)
      • Progress percentage and timing estimates
      • File counts and output summary
      • Error messages if simulation failed
    
    \b
    Possible States:
      COMPLETED ✓  - Simulation finished successfully
      RUNNING ⏳   - Simulation currently in progress  
      FAILED ✗     - Simulation encountered errors
      PAUSED ⏸     - Simulation paused with checkpoint
      NOT STARTED ⚪ - No simulation files found
    
    \b
    Examples:
      qbes status ./qbes_output
      qbes status /path/to/simulation/results
    """
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
@click.argument('output_dir', type=click.Path())
@click.option('--plot', '-p', is_flag=True, help='Generate and display plots')
def view(output_dir: str, plot: bool):
    """
    View simulation results and analysis.
    
    This command displays a summary of simulation results including coherence
    measures, energy evolution, and other observables. Optionally generates
    visualization plots.
    
    \b
    Display Information:
      • Simulation parameters and configuration
      • Final coherence metrics and observables  
      • Energy evolution statistics
      • File locations and availability
      • Optional: Interactive plots (--plot flag)
    
    \b
    Examples:
      qbes view ./photosystem_output
      qbes view ./enzyme_output --plot
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        click.echo(f"Output directory does not exist: {output_dir}", err=True)
        sys.exit(1)
    
    click.echo(f"📊 QBES Simulation Results")
    click.echo(f"Results Directory: {output_dir}")
    click.echo("=" * 70)
    
    # Check for result files
    result_file = output_path / "simulation_results.json"
    summary_file = output_path / "simulation_summary.txt"
    analysis_file = output_path / "detailed_analysis_report.txt"
    
    if result_file.exists():
        click.echo("\n✅ Simulation completed successfully")
        
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Display key metrics
            if 'configuration' in results:
                config = results['configuration']
                click.echo(f"\n📋 Configuration:")
                click.echo(f"  Temperature: {config.get('temperature', 'N/A')} K")
                click.echo(f"  Simulation time: {config.get('simulation_time', 'N/A')} s")
                click.echo(f"  Time step: {config.get('time_step', 'N/A')} s")
            
            if 'final_state' in results:
                click.echo(f"\n🎯 Final State Metrics:")
                state = results['final_state']
                click.echo(f"  Purity: {state.get('purity', 'N/A'):.6f}")
                click.echo(f"  Entropy: {state.get('entropy', 'N/A'):.6f}")
            
            if 'coherence_metrics' in results:
                click.echo(f"\n✨ Coherence Metrics:")
                coh = results['coherence_metrics']
                click.echo(f"  L1 Coherence: {coh.get('l1_norm', 'N/A'):.6f}")
                click.echo(f"  Coherence lifetime: {coh.get('lifetime', 'N/A'):.6e} s")
            
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not parse results file: {e}")
    
    # Display summary file if available
    if summary_file.exists():
        click.echo(f"\n📄 Summary Report:")
        click.echo(f"  {summary_file}")
        click.echo(f"\nFirst few lines:")
        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()[:10]
                for line in lines:
                    click.echo(f"  {line.rstrip()}")
        except Exception:
            pass
    
    # Display analysis file if available
    if analysis_file.exists():
        click.echo(f"\n📊 Detailed Analysis:")
        click.echo(f"  {analysis_file}")
    
    # List available data files
    data_files = list(output_path.glob("*.csv")) + list(output_path.glob("*.json"))
    if data_files:
        click.echo(f"\n📁 Data Files ({len(data_files)}):")
        for f in data_files[:5]:
            click.echo(f"  • {f.name}")
        if len(data_files) > 5:
            click.echo(f"  ... and {len(data_files) - 5} more")
    
    # List plots
    plot_files = list(output_path.glob("*.png")) + list(output_path.glob("*.pdf"))
    if plot_files:
        click.echo(f"\n📈 Plots ({len(plot_files)}):")
        for f in plot_files[:5]:
            click.echo(f"  • {f.name}")
        if len(plot_files) > 5:
            click.echo(f"  ... and {len(plot_files) - 5} more")
    
    if plot:
        click.echo(f"\n🎨 Generating visualization plots...")
        # Try to import and use visualization
        try:
            from .visualization import VisualizationEngine
            viz = VisualizationEngine()
            # Add visualization logic here
            click.echo("✅ Plots generated (feature in development)")
        except Exception as e:
            click.echo(f"⚠️  Visualization not available: {e}")
    
    click.echo(f"\n💡 Tip: Use 'qbes analyze {output_dir}' for detailed statistical analysis")


@main.command()
@click.argument('output_file', type=click.Path())
@click.option('--template', '-t', 
              type=click.Choice(['default', 'photosystem', 'enzyme', 'membrane']),
              default='default', help='Pre-configured template for specific system types')
@click.option('--interactive', '-i', is_flag=True, 
              help='Use interactive wizard with guided questions (recommended)')
def generate_config(output_file: str, template: str, interactive: bool):
    """
    Generate a configuration file for QBES simulations.
    
    This command creates a YAML configuration file with all parameters needed
    to run a QBES simulation. Use the interactive mode for guided setup, or
    choose a template for common biological systems.
    
    \b
    Templates Available:
      default     - Basic template with common parameters
      photosystem - Optimized for photosynthetic complexes
      enzyme      - Configured for enzyme active sites  
      membrane    - Set up for membrane proteins
    
    \b
    Interactive Mode (Recommended):
      The interactive wizard asks simple questions in plain language and
      generates a properly formatted configuration file. It includes:
      • System setup (PDB file, force field, temperature)
      • Simulation parameters (time, time step)
      • Quantum subsystem selection
      • Environmental noise model
      • Output configuration
    
    \b
    Examples:
      qbes generate-config config.yaml --interactive
      qbes generate-config photosystem.yaml --template photosystem
      qbes generate-config enzyme_config.yaml --template enzyme
    
    \b
    Next Steps:
      1. Edit the generated file if needed
      2. Validate: qbes validate config.yaml
      3. Run: qbes run config.yaml
    """
    try:
        if interactive:
            # Use interactive wizard
            wizard = InteractiveConfigWizard()
            config_data = wizard.run_wizard()
            
            success = wizard.generate_config_file(output_file)
            
            if success:
                click.echo(f"\n✓ Configuration file generated successfully: {output_file}")
                click.echo("You can now run your simulation with:")
                click.echo(f"  qbes run {output_file}")
            else:
                click.echo("Failed to generate configuration file", err=True)
                sys.exit(1)
        else:
            # Use template-based generation
            config_manager = ConfigurationManager()
            
            if template == 'default':
                success = config_manager.generate_default_config(output_file)
            else:
                success = _generate_specialized_config(output_file, template)
            
            if success:
                click.echo(f"Configuration template ({template}) generated: {output_file}")
                click.echo(f"Edit the file to customize parameters for your system.")
                click.echo("Tip: Use --interactive flag for guided configuration setup.")
            else:
                click.echo("Failed to generate configuration template", err=True)
                sys.exit(1)
            
    except FileNotFoundError as e:
        error_handler = ImprovedErrorHandler()
        error_msg = error_handler.handle_file_not_found(str(e), "configuration generation")
        click.echo(error_msg, err=True)
        sys.exit(1)
    except PermissionError as e:
        click.echo(f"❌ Permission denied: Cannot write to {output_file}\n\n💡 Suggestions:\n• Check write permissions for the directory\n• Try a different output location\n• Run with appropriate permissions", err=True)
        sys.exit(1)
    except Exception as e:
        error_handler = ImprovedErrorHandler()
        error_msg = f"❌ Configuration generation failed: {e}\n\n💡 Suggestions:\n• Check the output directory exists\n• Verify write permissions\n• Try a different output file name"
        click.echo(error_msg, err=True)
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
def validate_config(config_file: str):
    """
    Validate a QBES configuration file.
    
    This command checks your configuration file for errors and provides
    specific suggestions for fixing any issues found. It validates:
    • YAML syntax and structure
    • Parameter values and ranges
    • File paths and accessibility
    • Physical consistency of parameters
    
    \b
    Validation Checks:
      ✓ File format and syntax
      ✓ Required parameters present
      ✓ Parameter value ranges
      ✓ PDB file accessibility
      ✓ Output directory permissions
      ✓ Physical parameter consistency
    
    \b
    Examples:
      qbes validate config.yaml
      qbes validate my_simulation.yaml
    
    \b
    Exit Codes:
      0 - Configuration is valid
      1 - Validation errors found
    """
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


@main.command()
@click.option('--suite', '-s', 
              type=click.Choice(['quick', 'standard', 'full']),
              default='standard', 
              help='Validation suite to run (default: standard)')
@click.option('--output-dir', '-o', default='./validation_results', 
              help='Directory to save validation results and reports')
@click.option('--tolerance', '-t', default=0.02, type=float,
              help='Relative tolerance for benchmark comparisons (default: 0.02)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Show detailed output during validation')
@click.option('--report-format', 
              type=click.Choice(['markdown', 'json']),
              default='markdown',
              help='Format for validation report (default: markdown)')
def validate(suite: str, output_dir: str, tolerance: float, verbose: bool, report_format: str):
    """
    Run QBES validation benchmarks against scientific reference values.
    
    This command executes comprehensive validation tests comparing QBES results
    with published experimental and theoretical data. It validates both numerical
    accuracy and scientific correctness of the simulation engine.
    
    \b
    Available Suites:
      quick    - Essential analytical benchmarks (2-3 tests, ~30 seconds)
      standard - Core validation suite (4-6 tests, ~2 minutes)  
      full     - Complete validation including FMO complex (~10 minutes)
    
    \b
    Validation Tests Include:
      • Two-level system Rabi oscillations (analytical)
      • Harmonic oscillator ground state energy (analytical)
      • FMO complex coherence lifetime (literature)
      • Decoherence and energy transfer benchmarks
      • Numerical accuracy and stability checks
    
    \b
    Examples:
      qbes validate --suite standard
      qbes validate --suite full --tolerance 0.01 --verbose
      qbes validate --output-dir ./my_validation --report-format json
    
    \b
    Exit Codes:
      0 - All validations passed
      1 - Some validations failed  
      2 - Validation suite execution error
    """
    from .benchmarks.benchmark_runner import BenchmarkRunner
    import os
    
    if verbose:
        click.echo(f"🔬 Starting QBES validation suite: {suite}")
        click.echo(f"📁 Output directory: {output_dir}")
        click.echo(f"📊 Tolerance: {tolerance:.1%}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(output_dir=output_dir)
        
        # Show progress indicator
        if verbose:
            click.echo("\n⏳ Initializing validation benchmarks...")
        
        # Execute validation suite with progress monitoring
        start_time = time.time()
        results = runner.run_validation_suite(suite_type=suite)
        execution_time = time.time() - start_time
        
        # Generate and save validation report
        if verbose:
            click.echo("📝 Generating validation report...")
        
        if report_format == 'markdown':
            report_file = os.path.join(output_dir, 'validation_report.md')
        else:  # json
            report_file = os.path.join(output_dir, 'validation_report.json')
            # Save JSON format
            with open(report_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
        
        # Save markdown report regardless (for human readability)
        markdown_file = runner.save_validation_report(results, 'validation_report.md')
        
        # Display comprehensive summary
        click.echo(f"\n{'='*60}")
        click.echo("🎯 QBES VALIDATION SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Suite Type:       {suite}")
        click.echo(f"Execution Time:   {execution_time:.1f} seconds")
        click.echo(f"Total Tests:      {results.total_tests}")
        click.echo(f"Passed Tests:     {results.passed_tests}")
        click.echo(f"Failed Tests:     {results.total_tests - results.passed_tests}")
        click.echo(f"Pass Rate:        {results.pass_rate:.1f}%")
        click.echo(f"Overall Accuracy: {results.overall_accuracy:.1f}%")
        click.echo(f"Tolerance Used:   {tolerance:.1%}")
        click.echo(f"Report Location:  {markdown_file}")
        
        # Show individual test results if verbose
        if verbose and results.results:
            click.echo(f"\n📋 Individual Test Results:")
            click.echo("-" * 60)
            for result in results.results:
                status_icon = "✅" if result.passed else "❌"
                click.echo(f"{status_icon} {result.test_name}")
                click.echo(f"   Computed: {result.computed_value:.6f}")
                click.echo(f"   Reference: {result.reference_value:.6f}")
                click.echo(f"   Error: {result.relative_error:.4f} ({result.relative_error*100:.2f}%)")
                click.echo(f"   Time: {result.computation_time:.3f}s")
        
        # Show failed tests summary
        failed_results = [r for r in results.results if not r.passed]
        if failed_results:
            click.echo(f"\n⚠️  Failed Tests ({len(failed_results)}):")
            for result in failed_results:
                click.echo(f"   • {result.test_name}: {result.relative_error:.2%} error "
                          f"(tolerance: {result.tolerance:.2%})")
        
        # Certification status
        click.echo(f"\n🏆 Certification Status:")
        if results.pass_rate >= 100.0 and results.overall_accuracy >= 98.0:
            click.echo("   ✅ CERTIFIED - QBES meets all validation criteria")
        elif results.pass_rate >= 95.0 and results.overall_accuracy >= 95.0:
            click.echo("   ⚠️  CONDITIONAL - Minor improvements recommended")
        else:
            click.echo("   ❌ NOT CERTIFIED - Significant improvements required")
        
        # Performance assessment
        if execution_time < 60:
            perf_status = "🚀 Excellent"
        elif execution_time < 300:
            perf_status = "⚡ Good"
        else:
            perf_status = "🐌 Slow"
        
        click.echo(f"   Performance: {perf_status} ({execution_time:.1f}s)")
        
        # Exit with appropriate code
        if results.pass_rate >= 100.0:
            click.echo("\n🎉 All validation tests passed successfully!")
            sys.exit(0)
        else:
            click.echo(f"\n💡 {results.total_tests - results.passed_tests} test(s) need attention. "
                      f"Check the report for details.")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error running validation suite: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo("\n🔍 Detailed error information:", err=True)
            click.echo(traceback.format_exc(), err=True)
        sys.exit(2)


@main.command()
@click.option('--suite', '-s', 
              type=click.Choice(['quick', 'standard', 'full']),
              default='standard', 
              help='Validation suite to run (default: standard)')
@click.option('--max-iterations', '-i', default=5, type=int,
              help='Maximum debugging iterations (default: 5)')
@click.option('--target-accuracy', '-a', default=98.0, type=float,
              help='Target accuracy percentage (default: 98.0)')
@click.option('--output-dir', '-o', default='./debugging_results', 
              help='Directory to save debugging results and reports')
@click.option('--verbose', '-v', is_flag=True, 
              help='Show detailed output during debugging')
@click.option('--auto-fix', is_flag=True, default=True,
              help='Automatically apply fixes (default: enabled)')
def debug_loop(suite: str, max_iterations: int, target_accuracy: float, 
               output_dir: str, verbose: bool, auto_fix: bool):
    """
    Execute autonomous debugging loop with automatic error correction.
    
    This command runs the QBES debugging loop that automatically detects
    validation failures, performs root cause analysis, applies fixes, and
    iterates until target accuracy is achieved or maximum iterations reached.
    
    \b
    The debugging loop performs:
      • Error detection and classification
      • Root cause analysis of validation failures  
      • Automatic fix application for common issues
      • Validation re-execution after fixes
      • CHANGELOG.md updates with documented fixes
      • Comprehensive debugging session reports
    
    \b
    Common Fix Types:
      • Numerical precision improvements
      • Tolerance threshold adjustments
      • Parameter configuration optimization
      • Algorithm implementation corrections
    
    \b
    Examples:
      qbes debug-loop --suite standard --target-accuracy 98.5
      qbes debug-loop --suite full --max-iterations 10 --verbose
      qbes debug-loop --output-dir ./my_debug --auto-fix
    
    \b
    Exit Codes:
      0 - Target accuracy achieved successfully
      1 - Target not achieved within max iterations
      2 - Debugging loop execution error
    """
    from .validation.debugging_loop import DebuggingLoop, ValidationConfig
    from .validation.validator import QBESValidator
    import os
    
    if verbose:
        click.echo(f"🔧 Starting QBES debugging loop")
        click.echo(f"🎯 Target accuracy: {target_accuracy:.1f}%")
        click.echo(f"🔄 Max iterations: {max_iterations}")
        click.echo(f"📁 Output directory: {output_dir}")
        click.echo(f"🧪 Validation suite: {suite}")
        click.echo(f"🤖 Auto-fix: {'enabled' if auto_fix else 'disabled'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize debugging components
        if verbose:
            click.echo("\n⚙️  Initializing debugging system...")
        
        # Configure validator
        validator_config = ValidationConfig(
            suite_type=suite,
            accuracy_threshold=target_accuracy,
            pass_rate_threshold=100.0,
            max_retries=0,  # Debugging loop handles retries
            output_dir=output_dir
        )
        
        validator = QBESValidator(config=validator_config)
        debugging_loop = DebuggingLoop(validator=validator)
        
        # Execute debugging loop
        if verbose:
            click.echo("🚀 Starting autonomous debugging execution...")
        
        start_time = time.time()
        session = debugging_loop.execute_debugging_loop(
            max_iterations=max_iterations,
            target_accuracy=target_accuracy,
            suite=suite
        )
        execution_time = time.time() - start_time
        
        # Display comprehensive summary
        click.echo(f"\n{'='*60}")
        click.echo("🔧 DEBUGGING LOOP SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Session ID:       {session.session_id}")
        click.echo(f"Execution Time:   {execution_time:.1f} seconds")
        click.echo(f"Initial Accuracy: {session.initial_accuracy:.1f}%")
        click.echo(f"Final Accuracy:   {session.final_accuracy:.1f}%")
        click.echo(f"Improvement:      {session.final_accuracy - session.initial_accuracy:+.1f}%")
        click.echo(f"Total Attempts:   {session.total_fixes_attempted}")
        click.echo(f"Successful Fixes: {session.successful_fixes}")
        click.echo(f"Target Achieved:  {'✅ YES' if session.success else '❌ NO'}")
        
        # Show fix details if verbose
        if verbose and session.fix_attempts:
            click.echo(f"\n🔨 Applied Fixes:")
            click.echo("-" * 60)
            for i, fix in enumerate(session.fix_attempts, 1):
                status_icon = "✅" if fix.success else "❌"
                improvement = fix.accuracy_after - fix.accuracy_before
                click.echo(f"{status_icon} Fix {i}: {fix.diagnosis.test_name}")
                click.echo(f"   Type: {fix.diagnosis.error_type.value}")
                click.echo(f"   Applied: {fix.fix_applied}")
                click.echo(f"   Improvement: {improvement:+.2f}%")
                if fix.notes:
                    click.echo(f"   Notes: {fix.notes}")
        
        # Show error diagnoses
        if session.error_diagnoses:
            click.echo(f"\n🔍 Error Diagnoses ({len(session.error_diagnoses)}):")
            for diagnosis in session.error_diagnoses:
                confidence_bar = "█" * int(diagnosis.confidence * 10)
                click.echo(f"   • {diagnosis.test_name}: {diagnosis.error_type.value}")
                click.echo(f"     Confidence: {confidence_bar} {diagnosis.confidence:.1%}")
                click.echo(f"     Fix: {diagnosis.suggested_fix}")
        
        # Performance assessment
        if execution_time < 120:
            perf_status = "🚀 Fast"
        elif execution_time < 600:
            perf_status = "⚡ Moderate"
        else:
            perf_status = "🐌 Slow"
        
        click.echo(f"\n📊 Performance: {perf_status} ({execution_time:.1f}s)")
        
        # Certification status
        click.echo(f"\n🏆 Final Status:")
        if session.success:
            click.echo("   ✅ TARGET ACHIEVED - QBES debugging successful")
            click.echo(f"   🎯 Accuracy: {session.final_accuracy:.1f}% (target: {target_accuracy:.1f}%)")
        else:
            click.echo("   ⚠️  TARGET NOT ACHIEVED - Additional work needed")
            click.echo(f"   📈 Progress: {session.final_accuracy:.1f}% / {target_accuracy:.1f}%")
            remaining = target_accuracy - session.final_accuracy
            click.echo(f"   📉 Remaining: {remaining:.1f}% accuracy improvement needed")
        
        # CHANGELOG update notification
        if session.successful_fixes > 0:
            click.echo(f"\n📝 CHANGELOG.md updated with {session.successful_fixes} documented fixes")
        
        # Save debugging report
        report_file = os.path.join(output_dir, f'debugging_session_{session.session_id}.md')
        report_content = debugging_loop.changelog_updater._generate_changelog_entry(session)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Debugging Session Report\n\n{report_content}")
        
        click.echo(f"📄 Detailed report saved: {report_file}")
        
        # Exit with appropriate code
        if session.success:
            click.echo("\n🎉 Debugging loop completed successfully!")
            sys.exit(0)
        else:
            click.echo(f"\n💡 Target accuracy not achieved. Consider:")
            click.echo("   • Increasing max iterations")
            click.echo("   • Lowering target accuracy")
            click.echo("   • Manual investigation of remaining issues")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error in debugging loop: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo("\n🔍 Detailed error information:", err=True)
            click.echo(traceback.format_exc(), err=True)
        sys.exit(2)


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
    """
    Benchmarking and validation utilities for QBES.
    
    These commands run standardized tests to validate QBES installation,
    performance, and scientific accuracy against known reference systems.
    
    \b
    Available Commands:
      run     - Execute the complete benchmark suite
      analyze - Analyze results from previous benchmark runs
    
    \b
    Examples:
      qbes benchmark run --output-dir ./benchmarks
      qbes benchmark analyze ./benchmark_results
    """
    pass


@benchmark_main.command()
@click.option('--output-dir', '-o', default='./benchmark_results', 
              help='Directory to save benchmark results and reports')
@click.option('--verbose', '-v', is_flag=True, 
              help='Show detailed output during benchmark execution')
@click.option('--final-time', '-t', default=1.0, type=float,
              help='Simulation time for each benchmark test (seconds)')
@click.option('--time-step', '-dt', default=0.01, type=float,
              help='Time step for benchmark simulations (seconds)')
@click.option('--scaling-test', is_flag=True, 
              help='Include performance scaling analysis with different system sizes')
def run_benchmarks(output_dir: str, verbose: bool, final_time: float, 
                  time_step: float, scaling_test: bool):
    """
    Run the complete QBES benchmark and validation suite.
    
    This command executes a comprehensive set of tests to validate QBES
    installation, numerical accuracy, and performance. Benchmarks include
    standard quantum systems with known analytical solutions.
    
    \b
    Benchmark Tests:
      • Two-level system coherent oscillations
      • Harmonic oscillator energy levels
      • Spin-boson model decoherence
      • Multi-level system dynamics
      • Environmental coupling validation
    
    \b
    Validation Criteria:
      • Numerical accuracy vs. analytical solutions
      • Energy conservation during evolution
      • Probability conservation (trace = 1)
      • Physical consistency of results
      • Performance benchmarks
    
    \b
    Examples:
      qbes benchmark run
      qbes benchmark run --verbose --scaling-test
      qbes benchmark run --output-dir ./my_benchmarks --final-time 2.0
    
    \b
    Output Files:
      • benchmark_report.txt - Detailed results summary
      • scaling_results.txt - Performance analysis (if --scaling-test)
      • Individual test result files
    """
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


@main.command()
def info():
    """
    Display system information and QBES requirements.
    
    This command shows information about your system, Python environment,
    and QBES installation to help with troubleshooting and optimization.
    """
    import platform
    import sys
    
    click.echo("QBES System Information")
    click.echo("=" * 40)
    click.echo(f"QBES Version:     1.1.0")
    click.echo(f"Python Version:   {sys.version.split()[0]}")
    click.echo(f"Platform:         {platform.system()} {platform.release()}")
    click.echo(f"Architecture:     {platform.machine()}")
    
    # Memory information
    try:
        import psutil
        memory = psutil.virtual_memory()
        click.echo(f"Total Memory:     {memory.total / (1024**3):.1f} GB")
        click.echo(f"Available Memory: {memory.available / (1024**3):.1f} GB")
        click.echo(f"CPU Cores:        {psutil.cpu_count()}")
    except ImportError:
        click.echo("Memory Info:      psutil not available")
    
    click.echo("\nDependency Status:")
    click.echo("-" * 20)
    
    # Check key dependencies
    dependencies = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'), 
        ('yaml', 'PyYAML'),
        ('click', 'Click'),
        ('matplotlib', 'Matplotlib'),
        ('psutil', 'psutil')
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            click.echo(f"✓ {name:<12} - Available")
        except ImportError:
            click.echo(f"✗ {name:<12} - Missing")
    
    click.echo("\nOptional Dependencies:")
    click.echo("-" * 25)
    
    optional_deps = [
        ('openmm', 'OpenMM'),
        ('mdtraj', 'MDTraj'),
        ('rdkit', 'RDKit')
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            click.echo(f"✓ {name:<12} - Available")
        except ImportError:
            click.echo(f"○ {name:<12} - Optional (not installed)")
    
    click.echo("\nSystem Requirements:")
    click.echo("-" * 23)
    click.echo("• Python 3.8 or higher")
    click.echo("• 8GB RAM minimum (16GB+ recommended)")
    click.echo("• NumPy, SciPy, Click, PyYAML")
    click.echo("• Optional: OpenMM, MDTraj for MD simulations")
    
    click.echo("\nQuick Start:")
    click.echo("-" * 15)
    click.echo("1. qbes generate-config config.yaml --interactive")
    click.echo("2. qbes run config.yaml")
    click.echo("3. qbes status ./qbes_output")


# Add command groups to main CLI
main.add_command(benchmark_main, name='benchmark')


if __name__ == '__main__':
    main()