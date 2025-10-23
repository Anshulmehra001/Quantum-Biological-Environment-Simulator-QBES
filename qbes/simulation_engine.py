"""
Main simulation orchestration engine.
"""

import os
import time
import pickle
import logging
from typing import Dict, List, Optional, Any
import numpy as np

from .core.interfaces import SimulationEngineInterface
from .core.data_models import (
    SimulationConfig, SimulationResults, ValidationResult, 
    DensityMatrix, CoherenceMetrics, StatisticalSummary,
    QuantumSubsystem, MolecularSystem, Hamiltonian, LindbladOperator
)
from .quantum_engine import QuantumEngine
from .md_engine import MDEngine
from .noise_models import NoiseModelFactory
from .config_manager import ConfigurationManager
from .analysis import ResultsAnalyzer
from .utils.error_handling import ErrorHandler
from .utils.logging import EnhancedLogger, ProgressInfo, log_simulation_phase, update_progress, finish_progress


class SimulationEngine(SimulationEngineInterface):
    """
    Main simulation orchestration engine for QBES.
    
    This class coordinates quantum mechanical (QM) and molecular dynamics (MD) 
    simulations to study quantum effects in biological environments. It manages
    the coupling between quantum subsystems and classical environments, handles
    noise models, and provides comprehensive analysis of quantum coherence and
    decoherence processes.
    
    Key Features:
    - Hybrid QM/MD simulation orchestration
    - Noise model integration for biological environments
    - Real-time progress monitoring and logging
    - Comprehensive error handling and validation
    - Results analysis and visualization coordination
    
    Attributes:
        config_manager: Configuration management system
        quantum_engine: Quantum mechanics simulation engine
        md_engine: Molecular dynamics simulation engine
        noise_factory: Factory for creating noise models
        analyzer: Results analysis and metrics calculation
        logger: Enhanced logging system with progress tracking
    
    This class implements the hybrid QM/MM simulation loop, managing the interaction
    between quantum mechanical calculations and molecular dynamics simulations to
    model quantum effects in biological environments.
    """
    
    def __init__(self):
        """Initialize the simulation engine with all required components."""
        self.config = None
        self.quantum_engine = QuantumEngine()
        self.md_engine = MDEngine()
        self.noise_factory = NoiseModelFactory()
        self.config_manager = ConfigurationManager()
        self.analyzer = ResultsAnalyzer()
        self.error_handler = ErrorHandler()
        
        # Simulation state
        self.progress = 0.0
        self.is_paused = False
        self.is_running = False
        self.current_time = 0.0
        self.time_step_count = 0
        
        # Simulation data
        self.molecular_system = None
        self.quantum_subsystem = None
        self.hamiltonian = None
        self.noise_model = None
        self.current_state = None
        
        # Results storage
        self.state_trajectory = []
        self.energy_trajectory = []
        self.coherence_trajectory = {}
        self.intermediate_results = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.enhanced_logger = EnhancedLogger(verbose=False)
    
    def set_verbose_logging(self, verbose: bool):
        """Enable or disable verbose logging."""
        self.enhanced_logger.set_verbose(verbose)
    
    def perform_dry_run_validation(self, config: SimulationConfig) -> ValidationResult:
        """
        Perform comprehensive dry-run validation without executing simulation.
        
        This method validates all setup steps including configuration parsing,
        PDB loading, quantum subsystem identification, and Hamiltonian construction
        without running the actual time evolution.
        
        Args:
            config: Simulation configuration to validate
            
        Returns:
            ValidationResult: Comprehensive validation results with detailed feedback
        """
        self.enhanced_logger.info("Starting dry-run validation...")
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Step 1: Configuration validation
            self.enhanced_logger.info("1. Validating configuration parameters...")
            config_validation = self.config_manager.validate_parameters(config)
            if not config_validation.is_valid:
                validation_result.errors.extend(config_validation.errors)
                validation_result.warnings.extend(config_validation.warnings)
                validation_result.is_valid = False
                self.enhanced_logger.error("Configuration validation failed")
            else:
                self.enhanced_logger.info("✅ Configuration parameters valid")
            
            # Step 2: PDB file validation and parsing
            self.enhanced_logger.info("2. Validating PDB file and molecular system...")
            try:
                if not os.path.exists(config.system_pdb):
                    validation_result.add_error(f"PDB file not found: {config.system_pdb}")
                    validation_result.is_valid = False
                else:
                    molecular_system = self.config_manager.parse_pdb(config.system_pdb)
                    atom_count = len(molecular_system.atoms) if hasattr(molecular_system, 'atoms') else 0
                    self.enhanced_logger.info(f"✅ PDB file loaded successfully ({atom_count} atoms)")
                    
                    # Store for subsequent steps
                    self.molecular_system = molecular_system
                    
            except Exception as e:
                validation_result.add_error(f"PDB parsing failed: {str(e)}")
                validation_result.is_valid = False
                self.enhanced_logger.error(f"PDB parsing failed: {str(e)}")
            
            # Step 3: Quantum subsystem identification
            if self.molecular_system and validation_result.is_valid:
                self.enhanced_logger.info("3. Validating quantum subsystem selection...")
                try:
                    quantum_subsystem = self.config_manager.identify_quantum_subsystem(
                        self.molecular_system, config.quantum_subsystem_selection
                    )
                    quantum_atom_count = len(quantum_subsystem.atoms) if hasattr(quantum_subsystem, 'atoms') else 0
                    self.enhanced_logger.info(f"✅ Quantum subsystem identified ({quantum_atom_count} atoms)")
                    
                    # Store for subsequent steps
                    self.quantum_subsystem = quantum_subsystem
                    
                except Exception as e:
                    validation_result.add_error(f"Quantum subsystem identification failed: {str(e)}")
                    validation_result.is_valid = False
                    self.enhanced_logger.error(f"Quantum subsystem identification failed: {str(e)}")
            
            # Step 4: Hamiltonian construction validation
            if self.quantum_subsystem and validation_result.is_valid:
                self.enhanced_logger.info("4. Validating Hamiltonian construction...")
                try:
                    hamiltonian = self.quantum_engine.initialize_hamiltonian(self.quantum_subsystem)
                    hamiltonian_size = hamiltonian.matrix.shape[0] if hasattr(hamiltonian, 'matrix') else 0
                    self.enhanced_logger.info(f"✅ Hamiltonian constructed ({hamiltonian_size}x{hamiltonian_size} matrix)")
                    
                    # Validate Hamiltonian properties
                    if hasattr(hamiltonian, 'matrix'):
                        # Check if Hermitian
                        is_hermitian = np.allclose(hamiltonian.matrix, hamiltonian.matrix.conj().T)
                        if not is_hermitian:
                            validation_result.add_warning("Hamiltonian is not Hermitian")
                        else:
                            self.enhanced_logger.info("✅ Hamiltonian is Hermitian")
                    
                except Exception as e:
                    validation_result.add_error(f"Hamiltonian construction failed: {str(e)}")
                    validation_result.is_valid = False
                    self.enhanced_logger.error(f"Hamiltonian construction failed: {str(e)}")
            
            # Step 5: MD system validation
            self.enhanced_logger.info("5. Validating molecular dynamics setup...")
            try:
                # Test MD system initialization without full setup
                md_system = self.md_engine.initialize_system(
                    config.system_pdb, config.force_field
                )
                self.enhanced_logger.info("✅ MD system initialization successful")
                
            except Exception as e:
                validation_result.add_error(f"MD system initialization failed: {str(e)}")
                validation_result.is_valid = False
                self.enhanced_logger.error(f"MD system initialization failed: {str(e)}")
            
            # Step 6: Noise model validation
            self.enhanced_logger.info("6. Validating noise model setup...")
            try:
                noise_model = self.noise_factory.create_noise_model(
                    config.noise_model_type, config.temperature
                )
                self.enhanced_logger.info(f"✅ Noise model '{config.noise_model_type}' created successfully")
                
            except Exception as e:
                validation_result.add_error(f"Noise model creation failed: {str(e)}")
                validation_result.is_valid = False
                self.enhanced_logger.error(f"Noise model creation failed: {str(e)}")
            
            # Step 7: Output directory validation
            self.enhanced_logger.info("7. Validating output directory...")
            try:
                os.makedirs(config.output_directory, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(config.output_directory, ".dry_run_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.enhanced_logger.info(f"✅ Output directory accessible: {config.output_directory}")
                
            except Exception as e:
                validation_result.add_error(f"Output directory validation failed: {str(e)}")
                validation_result.is_valid = False
                self.enhanced_logger.error(f"Output directory validation failed: {str(e)}")
            
            # Step 8: Simulation parameters validation
            self.enhanced_logger.info("8. Validating simulation parameters...")
            total_steps = int(config.simulation_time / config.time_step)
            if total_steps <= 0:
                validation_result.add_error("Invalid simulation parameters: total steps <= 0")
                validation_result.is_valid = False
            elif total_steps > 1e8:
                validation_result.add_warning(f"Very large number of steps ({total_steps:,}). Consider larger time step.")
            else:
                self.enhanced_logger.info(f"✅ Simulation parameters valid ({total_steps:,} steps)")
            
            # Generate dry-run summary
            if validation_result.is_valid:
                self._print_dry_run_summary(config, total_steps)
            
        except Exception as e:
            validation_result.add_error(f"Unexpected error during dry-run validation: {str(e)}")
            validation_result.is_valid = False
            self.enhanced_logger.error(f"Unexpected error during dry-run validation: {str(e)}")
        
        return validation_result
    
    def _print_dry_run_summary(self, config: SimulationConfig, total_steps: int):
        """Print comprehensive dry-run summary."""
        self.enhanced_logger.info("\n" + "="*60)
        self.enhanced_logger.info("DRY-RUN SIMULATION SUMMARY")
        self.enhanced_logger.info("="*60)
        
        # System information
        self.enhanced_logger.info(f"System PDB: {config.system_pdb}")
        if self.molecular_system and hasattr(self.molecular_system, 'atoms'):
            self.enhanced_logger.info(f"Total atoms: {len(self.molecular_system.atoms):,}")
        
        if self.quantum_subsystem and hasattr(self.quantum_subsystem, 'atoms'):
            self.enhanced_logger.info(f"Quantum atoms: {len(self.quantum_subsystem.atoms):,}")
        
        # Simulation parameters
        self.enhanced_logger.info(f"Temperature: {config.temperature} K")
        self.enhanced_logger.info(f"Force field: {config.force_field}")
        self.enhanced_logger.info(f"Noise model: {config.noise_model_type}")
        
        # Time parameters
        self.enhanced_logger.info(f"Simulation time: {config.simulation_time:.2e} s")
        self.enhanced_logger.info(f"Time step: {config.time_step:.2e} s")
        self.enhanced_logger.info(f"Total steps: {total_steps:,}")
        
        # Estimated resources
        estimated_time_minutes = total_steps * 1e-4  # Rough estimate: 0.1ms per step
        estimated_memory_gb = total_steps * 1e-6  # Rough estimate: 1MB per 1000 steps
        
        self.enhanced_logger.info(f"Estimated runtime: {estimated_time_minutes:.1f} minutes")
        self.enhanced_logger.info(f"Estimated memory: {estimated_memory_gb:.2f} GB")
        
        # Output information
        self.enhanced_logger.info(f"Output directory: {config.output_directory}")
        
        self.enhanced_logger.info("="*60)
        self.enhanced_logger.info("All validation checks passed. Ready to run simulation.")
        self.enhanced_logger.info("="*60)
        
    def initialize_simulation(self, config: SimulationConfig) -> ValidationResult:
        """
        Initialize the simulation with given configuration.
        
        This method sets up all components needed for the hybrid QM/MM simulation,
        including molecular system parsing, quantum subsystem identification,
        and noise model initialization.
        """
        self.enhanced_logger.log_simulation_step('md_initialization')
        self.config = config
        
        # Set logging level based on configuration
        if hasattr(config, 'debug_level'):
            log_level = getattr(logging, config.debug_level.upper())
            self.logger.setLevel(log_level)
            # Also set the root logger level if DEBUG is requested
            if config.debug_level.upper() == 'DEBUG':
                logging.getLogger().setLevel(logging.DEBUG)
        
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Validate configuration parameters
            config_validation = self.config_manager.validate_parameters(config)
            if not config_validation.is_valid:
                validation_result.errors.extend(config_validation.errors)
                validation_result.warnings.extend(config_validation.warnings)
                validation_result.is_valid = False
                return validation_result
            
            # Parse molecular system from PDB
            self.enhanced_logger.info(f"Loading molecular system from: {config.system_pdb}")
            self.molecular_system = self.config_manager.parse_pdb(config.system_pdb)
            
            # Identify quantum subsystem
            self.enhanced_logger.log_simulation_step('quantum_setup')
            self.quantum_subsystem = self.config_manager.identify_quantum_subsystem(
                self.molecular_system, config.quantum_subsystem_selection
            )
            
            # Initialize quantum Hamiltonian
            self.enhanced_logger.log_simulation_step('hamiltonian_construction')
            self.hamiltonian = self.quantum_engine.initialize_hamiltonian(self.quantum_subsystem)
            
            # Initialize MD system
            self.enhanced_logger.info("Setting up molecular dynamics environment...")
            md_system = self.md_engine.initialize_system(
                config.system_pdb, config.force_field
            )
            
            # Setup environment (solvation, etc.)
            self.md_engine.setup_environment(
                md_system, config.solvent_model, config.ionic_strength
            )
            
            # Energy minimization
            self.enhanced_logger.info("Performing energy minimization...")
            final_energy = self.md_engine.minimize_energy()
            self.enhanced_logger.info(f"Energy minimization completed. Final energy: {final_energy:.6f}")
            
            # Initialize noise model
            self.enhanced_logger.log_simulation_step('noise_model_setup')
            self.noise_model = self.noise_factory.create_noise_model(
                config.noise_model_type, config.temperature
            )
            
            # Create output directory
            os.makedirs(config.output_directory, exist_ok=True)
            
            # Reset simulation state
            self.progress = 0.0
            self.current_time = 0.0
            self.time_step_count = 0
            self.state_trajectory = []
            self.energy_trajectory = []
            self.coherence_trajectory = {}
            
            # Initialize quantum state (ground state or specified initial state)
            self.enhanced_logger.info("Preparing initial quantum state...")
            self._initialize_quantum_state()
            
            self.enhanced_logger.info("✅ Simulation initialization completed successfully")
            
        except Exception as e:
            error_msg = self.enhanced_logger.format_error_message(e, "simulation initialization")
            self.enhanced_logger.error(error_msg)
            validation_result.add_error(f"Initialization failed: {str(e)}")
            
        return validation_result
    
    def run_simulation(self) -> SimulationResults:
        """
        Execute the complete simulation workflow.
        
        This method implements the main hybrid QM/MM simulation loop, alternating
        between quantum state evolution and molecular dynamics steps while tracking
        progress and saving intermediate results.
        """
        if not self.config:
            raise RuntimeError("Simulation not initialized. Call initialize_simulation first.")
        
        self.enhanced_logger.log_simulation_step('quantum_evolution')
        self.is_running = True
        start_time = time.time()
        
        try:
            # Calculate simulation parameters
            total_steps = int(self.config.simulation_time / self.config.time_step)
            md_steps_per_qm = max(1, int(0.1 / self.config.time_step))  # MD every 0.1 fs
            
            self.enhanced_logger.info(f"Running {total_steps:,} steps with time step {self.config.time_step:.2e} s")
            self.enhanced_logger.info(f"Total simulation time: {self.config.simulation_time:.2e} s")
            
            # Initialize progress tracking
            progress_info = ProgressInfo(
                current_step=0,
                total_steps=total_steps,
                current_time=0.0,
                target_time=self.config.simulation_time,
                start_time=start_time,
                phase="quantum_evolution"
            )
            self.enhanced_logger.current_progress = progress_info
            
            # Main simulation loop
            for step in range(total_steps):
                if self.is_paused:
                    self.enhanced_logger.info("Simulation paused")
                    break
                
                # Update progress
                self.progress = (step / total_steps) * 100.0
                self.current_time = step * self.config.time_step
                self.time_step_count = step
                
                # Update progress display
                update_progress(step, total_steps, self.current_time, self.config.simulation_time)
                
                # Run MD step periodically to update environment
                if step % md_steps_per_qm == 0:
                    self._run_md_step()
                
                # Evolve quantum state
                self._evolve_quantum_state()
                
                # Calculate and store observables
                self._calculate_observables()
                
                # Save intermediate results periodically
                if step % 1000 == 0:
                    self._save_intermediate_results()
            
            # Finish progress display
            finish_progress()
            
            # Final calculations and analysis
            self.enhanced_logger.log_simulation_step('analysis')
            computation_time = time.time() - start_time
            
            # Generate statistical summary
            statistical_summary = self.analyzer.generate_statistical_summary(
                self._create_preliminary_results()
            )
            
            # Create final results object
            results = SimulationResults(
                state_trajectory=self.state_trajectory.copy(),
                coherence_measures=self.coherence_trajectory.copy(),
                energy_trajectory=self.energy_trajectory.copy(),
                decoherence_rates=self._calculate_decoherence_rates(),
                statistical_summary=statistical_summary,
                simulation_config=self.config,
                computation_time=computation_time
            )
            
            # Save final results
            self.enhanced_logger.log_simulation_step('output_generation')
            self._save_final_results(results)
            
            # Print results summary
            self.enhanced_logger.print_results_summary(results)
            
            self.enhanced_logger.log_simulation_step('completion')
            self.is_running = False
            
            return results
            
        except Exception as e:
            finish_progress()
            error_msg = self.enhanced_logger.format_error_message(e, "simulation execution")
            self.enhanced_logger.error(error_msg)
            self.is_running = False
            raise RuntimeError(f"Simulation execution failed: {str(e)}")
    
    def get_progress(self) -> float:
        """Get current simulation progress as percentage."""
        return self.progress
    
    def pause_simulation(self) -> bool:
        """
        Pause the running simulation.
        
        Returns:
            bool: True if successfully paused, False otherwise
        """
        if self.is_running:
            self.is_paused = True
            self.logger.info("Simulation pause requested")
            return True
        return False
    
    def resume_simulation(self) -> bool:
        """
        Resume a paused simulation.
        
        Returns:
            bool: True if successfully resumed, False otherwise
        """
        if self.is_paused:
            self.is_paused = False
            self.logger.info("Simulation resumed")
            return True
        return False
    
    def save_checkpoint(self, filepath: str) -> bool:
        """
        Save simulation state for later resumption.
        
        Args:
            filepath: Path to save checkpoint file
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            checkpoint_data = {
                'config': self.config,
                'progress': self.progress,
                'current_time': self.current_time,
                'time_step_count': self.time_step_count,
                'current_state': self.current_state,
                'state_trajectory': self.state_trajectory,
                'energy_trajectory': self.energy_trajectory,
                'coherence_trajectory': self.coherence_trajectory,
                'hamiltonian': self.hamiltonian,
                'quantum_subsystem': self.quantum_subsystem
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load simulation state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore simulation state
            self.config = checkpoint_data['config']
            self.progress = checkpoint_data['progress']
            self.current_time = checkpoint_data['current_time']
            self.time_step_count = checkpoint_data['time_step_count']
            self.current_state = checkpoint_data['current_state']
            self.state_trajectory = checkpoint_data['state_trajectory']
            self.energy_trajectory = checkpoint_data['energy_trajectory']
            self.coherence_trajectory = checkpoint_data['coherence_trajectory']
            self.hamiltonian = checkpoint_data['hamiltonian']
            self.quantum_subsystem = checkpoint_data['quantum_subsystem']
            
            self.logger.info(f"Checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return False
    
    def _initialize_quantum_state(self):
        """Initialize the quantum state (typically ground state)."""
        n_states = len(self.quantum_subsystem.basis_states)
        
        # Start with ground state (first basis state)
        initial_coeffs = np.zeros(n_states, dtype=complex)
        initial_coeffs[0] = 1.0
        
        # Create density matrix for pure ground state
        rho = np.outer(initial_coeffs, initial_coeffs.conj())
        
        self.current_state = DensityMatrix(
            matrix=rho,
            basis_labels=[f"state_{i}" for i in range(n_states)],
            time=0.0
        )
        
        # Store initial state
        self.state_trajectory.append(self.current_state)
    
    def _run_md_step(self):
        """Run a molecular dynamics step to update environmental parameters."""
        # Run short MD trajectory to get updated coordinates
        md_trajectory = self.md_engine.run_trajectory(
            duration=self.config.time_step * 10,  # 10 time steps worth
            time_step=self.config.time_step / 10,
            temperature=self.config.temperature
        )
        
        # Extract quantum parameters from MD trajectory
        quantum_atoms = [atom.atom_id for atom in self.quantum_subsystem.atoms]
        quantum_params = self.md_engine.extract_quantum_parameters(
            md_trajectory, quantum_atoms
        )
        
        # Update Hamiltonian if needed (for time-dependent systems)
        # This is a simplified implementation
        pass
    
    def _log_sanity_checks(self, state: DensityMatrix, step: int, phase: str = ""):
        """
        Log sanity checks for quantum state validation with DEBUG level logging.
        
        Args:
            state: Quantum state to validate
            step: Current simulation step
            phase: Description of current phase (e.g., "pre-evolution", "post-evolution")
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        
        try:
            # Check density matrix trace
            trace = np.real(np.trace(state.matrix))
            trace_status = "PASS" if np.isclose(trace, 1.0, rtol=1e-10) else "FAIL"
            self.logger.debug(f"Step {step:06d} {phase}: Density Matrix Trace = {trace:.10f} ({trace_status})")
            
            # Check Hermiticity
            hermitian_error = np.max(np.abs(state.matrix - state.matrix.conj().T))
            hermitian_status = "PASS" if hermitian_error < 1e-10 else "FAIL"
            self.logger.debug(f"Step {step:06d} {phase}: Hermiticity Error = {hermitian_error:.2e} ({hermitian_status})")
            
            # Check positive semidefinite (eigenvalues >= 0)
            eigenvals = np.linalg.eigvals(state.matrix)
            min_eigenval = np.min(np.real(eigenvals))
            psd_status = "PASS" if min_eigenval >= -1e-10 else "FAIL"
            self.logger.debug(f"Step {step:06d} {phase}: Min Eigenvalue = {min_eigenval:.2e} ({psd_status})")
            
            # Calculate purity
            purity = np.real(np.trace(state.matrix @ state.matrix))
            self.logger.debug(f"Step {step:06d} {phase}: Purity = {purity:.6f}")
            
            # Calculate von Neumann entropy
            eigenvals_positive = eigenvals[eigenvals > 1e-15]  # Avoid log(0)
            if len(eigenvals_positive) > 0:
                entropy = -np.sum(eigenvals_positive * np.log(eigenvals_positive))
                self.logger.debug(f"Step {step:06d} {phase}: von Neumann Entropy = {entropy:.6f}")
            
            # Energy calculation if Hamiltonian available
            if self.hamiltonian is not None:
                energy = np.real(np.trace(self.hamiltonian.matrix @ state.matrix))
                self.logger.debug(f"Step {step:06d} {phase}: Energy = {energy:.8f} a.u.")
            
            # Check for numerical instabilities
            if not np.all(np.isfinite(state.matrix)):
                self.logger.error(f"Step {step:06d} {phase}: NUMERICAL INSTABILITY - Non-finite values in density matrix!")
            
            # Check matrix norm
            matrix_norm = np.linalg.norm(state.matrix, 'fro')
            self.logger.debug(f"Step {step:06d} {phase}: Matrix Frobenius Norm = {matrix_norm:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error in sanity checks at step {step} {phase}: {str(e)}")

    def _evolve_quantum_state(self):
        """Evolve the quantum state by one time step with enhanced debugging."""
        # Log pre-evolution state information
        self._log_sanity_checks(self.current_state, self.time_step_count, "pre-evolution")
        
        # Generate Lindblad operators from noise model
        lindblad_ops = self.noise_model.generate_lindblad_operators(
            self.quantum_subsystem, self.config.temperature
        )
        
        # Evolve state using Lindblad master equation
        new_state = self.quantum_engine.evolve_state(
            self.current_state, self.config.time_step, 
            self.hamiltonian, lindblad_ops
        )
        
        # Validate the evolved state
        validation = self.quantum_engine.validate_quantum_state(new_state)
        if not validation.is_valid:
            self.logger.warning(f"Quantum state validation failed: {validation.errors}")
        
        # Update current state and store in trajectory
        new_state.time = self.current_time
        self.current_state = new_state
        self.state_trajectory.append(new_state)
        
        # Log post-evolution state information
        self._log_sanity_checks(self.current_state, self.time_step_count, "post-evolution")
        
        # Save state snapshot if configured
        if (hasattr(self.config, 'save_snapshot_interval') and 
            self.config.save_snapshot_interval > 0 and 
            self.time_step_count % self.config.save_snapshot_interval == 0):
            self._save_state_snapshot(self.current_state, self.time_step_count)
    


    def _calculate_observables(self):
        """Calculate and store observables for current state."""
        # Calculate energy
        energy = np.real(np.trace(self.hamiltonian.matrix @ self.current_state.matrix))
        self.energy_trajectory.append(energy)
        
        # Calculate coherence measures
        coherence_metrics = self.quantum_engine.calculate_coherence_measures(self.current_state)
        
        # Store coherence measures
        for metric_name in ['coherence_lifetime', 'quantum_discord', 'purity', 'von_neumann_entropy']:
            if metric_name not in self.coherence_trajectory:
                self.coherence_trajectory[metric_name] = []
            
            metric_value = getattr(coherence_metrics, metric_name, 0.0)
            self.coherence_trajectory[metric_name].append(metric_value)
    
    def _save_intermediate_results(self):
        """Save intermediate results for monitoring and recovery."""
        if not self.config:
            return
        
        intermediate_file = os.path.join(
            self.config.output_directory, 
            f"intermediate_results_step_{self.time_step_count}.pkl"
        )
        
        intermediate_data = {
            'step': self.time_step_count,
            'time': self.current_time,
            'progress': self.progress,
            'energy_trajectory': self.energy_trajectory[-1000:],  # Last 1000 points
            'coherence_measures': {k: v[-1000:] for k, v in self.coherence_trajectory.items()}
        }
        
        try:
            with open(intermediate_file, 'wb') as f:
                pickle.dump(intermediate_data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save intermediate results: {str(e)}")
    
    def _create_preliminary_results(self) -> SimulationResults:
        """Create preliminary results object for analysis."""
        return SimulationResults(
            state_trajectory=self.state_trajectory,
            coherence_measures=self.coherence_trajectory,
            energy_trajectory=self.energy_trajectory,
            decoherence_rates={},  # Will be calculated
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=len(self.state_trajectory)
            ),
            simulation_config=self.config,
            computation_time=0.0
        )
    
    def _calculate_decoherence_rates(self) -> Dict[str, float]:
        """Calculate decoherence rates from simulation data."""
        rates = {}
        
        # Calculate rates from noise model
        if self.noise_model and self.quantum_subsystem:
            rates = self.noise_model.calculate_decoherence_rates(
                self.quantum_subsystem, self.config.temperature
            )
        
        return rates
    
    def _log_sanity_checks(self, state: DensityMatrix, step: int, phase: str):
        """
        Perform comprehensive sanity checks with DEBUG-level logging.
        
        Args:
            state: Current density matrix to check
            step: Current simulation step
            phase: Phase identifier (e.g., "pre-evolution", "post-evolution")
        """
        # Skip if sanity checks are disabled or DEBUG logging is not enabled
        if (not getattr(self.config, 'enable_sanity_checks', True) or 
            not self.logger.isEnabledFor(logging.DEBUG)):
            return
        
        try:
            # 1. Density matrix trace validation
            trace_val = np.real(np.trace(state.matrix))
            trace_status = "PASS" if np.isclose(trace_val, 1.0, rtol=1e-10) else "FAIL"
            self.logger.debug(f"Step {step} ({phase}): Density Matrix Trace = {trace_val:.12f} ({trace_status})")
            
            # 2. Hermiticity check
            is_hermitian = np.allclose(state.matrix, state.matrix.conj().T, rtol=1e-10)
            hermitian_status = "PASS" if is_hermitian else "FAIL"
            hermitian_error = np.max(np.abs(state.matrix - state.matrix.conj().T))
            self.logger.debug(f"Step {step} ({phase}): Hermiticity Check = {hermitian_status} (max error: {hermitian_error:.2e})")
            
            # 3. Positive semidefinite check
            eigenvals = np.linalg.eigvals(state.matrix)
            min_eigenval = np.min(np.real(eigenvals))
            psd_status = "PASS" if min_eigenval >= -1e-10 else "FAIL"
            self.logger.debug(f"Step {step} ({phase}): Positive Semidefinite = {psd_status} (min eigenval: {min_eigenval:.2e})")
            
            # 4. Purity calculation and bounds check
            purity = np.real(np.trace(state.matrix @ state.matrix))
            dimension = state.matrix.shape[0]
            min_purity = 1.0 / dimension
            purity_bounds_ok = min_purity <= purity <= 1.0 + 1e-10
            purity_status = "PASS" if purity_bounds_ok else "FAIL"
            self.logger.debug(f"Step {step} ({phase}): Purity = {purity:.8f} [{min_purity:.3f}, 1.0] ({purity_status})")
            
            # 5. Energy conservation monitoring (if Hamiltonian available)
            if self.hamiltonian is not None:
                energy = np.real(np.trace(self.hamiltonian.matrix @ state.matrix))
                if hasattr(self, '_initial_energy'):
                    energy_change = abs(energy - self._initial_energy)
                    energy_conservation_ok = energy_change < 1e-6  # Tolerance for energy drift
                    energy_status = "PASS" if energy_conservation_ok else "DRIFT"
                    self.logger.debug(f"Step {step} ({phase}): Energy = {energy:.8f} (change: {energy_change:.2e}) ({energy_status})")
                else:
                    self._initial_energy = energy
                    self.logger.debug(f"Step {step} ({phase}): Initial Energy = {energy:.8f}")
            
            # 6. Numerical stability indicators
            # Check for NaN or Inf values
            has_nan = np.any(np.isnan(state.matrix))
            has_inf = np.any(np.isinf(state.matrix))
            numerical_status = "PASS" if not (has_nan or has_inf) else "FAIL"
            self.logger.debug(f"Step {step} ({phase}): Numerical Stability = {numerical_status} (NaN: {has_nan}, Inf: {has_inf})")
            
            # 7. Matrix condition number (numerical conditioning)
            try:
                cond_number = np.linalg.cond(state.matrix)
                cond_status = "GOOD" if cond_number < 1e12 else "POOR"
                self.logger.debug(f"Step {step} ({phase}): Condition Number = {cond_number:.2e} ({cond_status})")
            except np.linalg.LinAlgError:
                self.logger.debug(f"Step {step} ({phase}): Condition Number = UNDEFINED (singular matrix)")
            
            # 8. Population conservation (diagonal elements should sum to 1)
            populations = np.real(np.diag(state.matrix))
            pop_sum = np.sum(populations)
            pop_status = "PASS" if np.isclose(pop_sum, 1.0, rtol=1e-10) else "FAIL"
            self.logger.debug(f"Step {step} ({phase}): Population Sum = {pop_sum:.12f} ({pop_status})")
            
            # 9. Coherence magnitude monitoring
            off_diagonal = state.matrix - np.diag(np.diag(state.matrix))
            max_coherence = np.max(np.abs(off_diagonal))
            total_coherence = np.sum(np.abs(off_diagonal))
            self.logger.debug(f"Step {step} ({phase}): Max Coherence = {max_coherence:.6f}, Total Coherence = {total_coherence:.6f}")
            
            # 10. Log any critical failures
            critical_failures = []
            if trace_status == "FAIL":
                critical_failures.append("trace")
            if hermitian_status == "FAIL":
                critical_failures.append("hermiticity")
            if psd_status == "FAIL":
                critical_failures.append("positive_semidefinite")
            if numerical_status == "FAIL":
                critical_failures.append("numerical_stability")
            
            if critical_failures:
                self.logger.warning(f"Step {step} ({phase}): CRITICAL SANITY CHECK FAILURES: {', '.join(critical_failures)}")
            
        except Exception as e:
            self.logger.error(f"Step {step} ({phase}): Error during sanity checks: {str(e)}")
    
    def _save_state_snapshot(self, state: DensityMatrix, step: int):
        """
        Save quantum state snapshot to file for intermediate analysis.
        
        Args:
            state: Current quantum state to save
            step: Current simulation step number
        """
        if not self.config:
            return
        
        try:
            # Create snapshots directory if it doesn't exist
            snapshots_dir = os.path.join(self.config.output_directory, "snapshots")
            os.makedirs(snapshots_dir, exist_ok=True)
            
            # Create snapshot data with comprehensive metadata
            snapshot_data = {
                'step': step,
                'time': state.time,
                'density_matrix': state.matrix,
                'basis_labels': state.basis_labels,
                'simulation_time': self.current_time,
                'timestamp': time.time(),
                'config': {
                    'temperature': self.config.temperature,
                    'time_step': self.config.time_step,
                    'noise_model_type': self.config.noise_model_type,
                    'system_pdb': self.config.system_pdb
                },
                'hamiltonian': self.hamiltonian.matrix if self.hamiltonian else None,
                'energy': np.real(np.trace(self.hamiltonian.matrix @ state.matrix)) if self.hamiltonian else None,
                'purity': np.real(np.trace(state.matrix @ state.matrix)),
                'trace': np.real(np.trace(state.matrix)),
                'metadata': {
                    'trace': np.real(np.trace(state.matrix)),
                    'purity': np.real(np.trace(state.matrix @ state.matrix)),
                    'populations': np.real(np.diag(state.matrix)).tolist(),
                    'energy': np.real(np.trace(self.hamiltonian.matrix @ state.matrix)) if self.hamiltonian else None
                }
            }
            
            # Save snapshot using pickle for efficient storage
            snapshot_file = os.path.join(snapshots_dir, f"snapshot_step_{step:08d}.pkl")
            with open(snapshot_file, 'wb') as f:
                pickle.dump(snapshot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Log snapshot save with DEBUG level
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"State snapshot saved: snapshot_step_{step:08d}.pkl "
                                f"(step={step}, time={state.time:.2e}s, "
                                f"purity={snapshot_data['purity']:.6f})")
            
        except Exception as e:
            self.logger.error(f"Failed to save state snapshot at step {step}: {str(e)}")
    
    def load_state_snapshot(self, snapshot_path: str) -> Dict[str, Any]:
        """
        Load quantum state snapshot from file for analysis and debugging.
        
        Args:
            snapshot_path: Path to the snapshot file
            
        Returns:
            Dict containing snapshot data with density matrix and metadata
            
        Raises:
            FileNotFoundError: If snapshot file doesn't exist
            ValueError: If snapshot file is corrupted or invalid
        """
        try:
            if not os.path.exists(snapshot_path):
                raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
            
            # Load snapshot data
            with open(snapshot_path, 'rb') as f:
                snapshot_data = pickle.load(f)
            
            # Validate snapshot data structure
            required_keys = ['density_matrix', 'basis_labels', 'time', 'step']
            missing_keys = [key for key in required_keys if key not in snapshot_data]
            if missing_keys:
                raise ValueError(f"Invalid snapshot file: missing keys {missing_keys}")
            
            # Validate density matrix properties
            rho = snapshot_data['density_matrix']
            if not isinstance(rho, np.ndarray):
                raise ValueError("Invalid snapshot: density_matrix is not a numpy array")
            
            if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
                raise ValueError("Invalid snapshot: density_matrix is not square")
            
            # Check basic quantum state properties
            trace = np.trace(rho)
            if not np.isclose(trace, 1.0, rtol=1e-6):
                self.logger.warning(f"Loaded snapshot has non-unit trace: {trace:.6f}")
            
            # Check Hermiticity
            if not np.allclose(rho, rho.conj().T, rtol=1e-10):
                self.logger.warning("Loaded snapshot density matrix is not Hermitian")
            
            # Reconstruct DensityMatrix object
            density_matrix = DensityMatrix(
                matrix=rho,
                basis_labels=snapshot_data['basis_labels'],
                time=snapshot_data['time']
            )
            
            # Add reconstructed object to snapshot data
            snapshot_data['density_matrix_object'] = density_matrix
            
            self.logger.info(f"Successfully loaded snapshot from step {snapshot_data['step']} "
                           f"(time={snapshot_data['time']:.2e}s)")
            
            return snapshot_data
            
        except Exception as e:
            self.logger.error(f"Failed to load snapshot from {snapshot_path}: {str(e)}")
            raise
    
    def list_available_snapshots(self, snapshots_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available state snapshots in the snapshots directory.
        
        Args:
            snapshots_dir: Directory to search for snapshots. If None, uses default.
            
        Returns:
            List of dictionaries containing snapshot metadata
        """
        if snapshots_dir is None:
            if not self.config:
                raise RuntimeError("No configuration available and no snapshots_dir specified")
            snapshots_dir = os.path.join(self.config.output_directory, "snapshots")
        
        if not os.path.exists(snapshots_dir):
            return []
        
        snapshots = []
        
        try:
            # Find all snapshot files
            for filename in os.listdir(snapshots_dir):
                if filename.startswith("snapshot_step_") and filename.endswith(".pkl"):
                    filepath = os.path.join(snapshots_dir, filename)
                    
                    try:
                        # Load minimal metadata without full snapshot
                        with open(filepath, 'rb') as f:
                            snapshot_data = pickle.load(f)
                        
                        # Extract key metadata
                        metadata = {
                            'filename': filename,
                            'filepath': filepath,
                            'step': snapshot_data.get('step', 0),
                            'time': snapshot_data.get('time', 0.0),
                            'simulation_time': snapshot_data.get('simulation_time', 0.0),
                            'purity': snapshot_data.get('purity', None),
                            'energy': snapshot_data.get('energy', None),
                            'trace': snapshot_data.get('trace', None),
                            'timestamp': snapshot_data.get('timestamp', 0),
                            'matrix_size': snapshot_data['density_matrix'].shape[0] if 'density_matrix' in snapshot_data else 0
                        }
                        
                        snapshots.append(metadata)
                        
                    except Exception as e:
                        self.logger.warning(f"Could not read snapshot metadata from {filename}: {str(e)}")
            
            # Sort by step number
            snapshots.sort(key=lambda x: x['step'])
            
        except Exception as e:
            self.logger.error(f"Error listing snapshots in {snapshots_dir}: {str(e)}")
        
        return snapshots
    
    def analyze_snapshot_trajectory(self, snapshots_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trajectory of saved snapshots to extract coherence evolution.
        
        Args:
            snapshots_dir: Directory containing snapshots. If None, uses default.
            
        Returns:
            Dictionary containing trajectory analysis results
        """
        snapshots = self.list_available_snapshots(snapshots_dir)
        
        if not snapshots:
            return {'error': 'No snapshots found for analysis'}
        
        # Initialize trajectory data
        times = []
        purities = []
        energies = []
        traces = []
        coherences = []
        
        try:
            for snapshot_meta in snapshots:
                # Load full snapshot data
                snapshot_data = self.load_state_snapshot(snapshot_meta['filepath'])
                
                times.append(snapshot_data['time'])
                purities.append(snapshot_data.get('purity', np.nan))
                energies.append(snapshot_data.get('energy', np.nan))
                traces.append(snapshot_data.get('trace', np.nan))
                
                # Calculate coherence measure (sum of off-diagonal elements)
                rho = snapshot_data['density_matrix']
                off_diag_sum = np.sum(np.abs(rho - np.diag(np.diag(rho))))
                coherences.append(off_diag_sum)
            
            # Convert to numpy arrays for analysis
            times = np.array(times)
            purities = np.array(purities)
            energies = np.array(energies)
            traces = np.array(traces)
            coherences = np.array(coherences)
            
            # Calculate analysis metrics
            analysis = {
                'n_snapshots': len(snapshots),
                'time_range': (times[0], times[-1]) if len(times) > 0 else (0, 0),
                'purity_evolution': {
                    'initial': purities[0] if len(purities) > 0 else np.nan,
                    'final': purities[-1] if len(purities) > 0 else np.nan,
                    'min': np.nanmin(purities),
                    'max': np.nanmax(purities),
                    'mean': np.nanmean(purities)
                },
                'energy_evolution': {
                    'initial': energies[0] if len(energies) > 0 else np.nan,
                    'final': energies[-1] if len(energies) > 0 else np.nan,
                    'conservation_error': np.abs(energies[-1] - energies[0]) / np.abs(energies[0]) if len(energies) > 1 and energies[0] != 0 else np.nan
                },
                'coherence_evolution': {
                    'initial': coherences[0] if len(coherences) > 0 else np.nan,
                    'final': coherences[-1] if len(coherences) > 0 else np.nan,
                    'decay_rate': self._estimate_decay_rate(times, coherences) if len(times) > 1 else np.nan
                },
                'trace_validation': {
                    'mean_trace': np.nanmean(traces),
                    'trace_deviation': np.nanstd(traces),
                    'max_trace_error': np.nanmax(np.abs(traces - 1.0))
                },
                'raw_data': {
                    'times': times,
                    'purities': purities,
                    'energies': energies,
                    'coherences': coherences,
                    'traces': traces
                }
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _estimate_decay_rate(self, times: np.ndarray, values: np.ndarray) -> float:
        """Estimate exponential decay rate from time series data."""
        try:
            # Filter out invalid values
            valid_mask = np.isfinite(values) & (values > 0)
            if np.sum(valid_mask) < 2:
                return np.nan
            
            t_valid = times[valid_mask]
            v_valid = values[valid_mask]
            
            # Fit exponential decay: v(t) = v0 * exp(-gamma * t)
            # Take log: ln(v) = ln(v0) - gamma * t
            log_values = np.log(v_valid)
            
            # Linear fit to get decay rate
            coeffs = np.polyfit(t_valid, log_values, 1)
            decay_rate = -coeffs[0]  # Negative slope gives positive decay rate
            
            return decay_rate if decay_rate > 0 else np.nan
            
        except Exception:
            return np.nan
    
    def _save_final_results(self, results: SimulationResults):
        """Save final simulation results to files."""
        if not self.config:
            return
        
        # Save results as pickle file
        results_file = os.path.join(self.config.output_directory, "simulation_results.pkl")
        try:
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Results saved to {results_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
        
        # Save summary as text file
        summary_file = os.path.join(self.config.output_directory, "simulation_summary.txt")
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Quantum Biological Simulation Results\n")
                f.write(f"=====================================\n\n")
                f.write(f"Configuration:\n")
                f.write(f"  PDB file: {self.config.system_pdb}\n")
                f.write(f"  Temperature: {self.config.temperature} K\n")
                f.write(f"  Simulation time: {self.config.simulation_time} fs\n")
                f.write(f"  Time step: {self.config.time_step} fs\n")
                f.write(f"  Noise model: {self.config.noise_model_type}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Total steps: {len(results.state_trajectory)}\n")
                f.write(f"  Computation time: {results.computation_time:.2f} seconds\n")
                f.write(f"  Final energy: {results.energy_trajectory[-1]:.6f}\n")
                
            self.logger.info(f"Summary saved to {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save summary: {str(e)}")