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


class SimulationEngine(SimulationEngineInterface):
    """
    Main simulation orchestration engine that coordinates QM and MD components.
    
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
        
    def initialize_simulation(self, config: SimulationConfig) -> ValidationResult:
        """
        Initialize the simulation with given configuration.
        
        This method sets up all components needed for the hybrid QM/MM simulation,
        including molecular system parsing, quantum subsystem identification,
        and noise model initialization.
        """
        self.logger.info("Initializing simulation with configuration")
        self.config = config
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
            self.logger.info(f"Parsing PDB file: {config.system_pdb}")
            self.molecular_system = self.config_manager.parse_pdb(config.system_pdb)
            
            # Identify quantum subsystem
            self.logger.info("Identifying quantum subsystem")
            self.quantum_subsystem = self.config_manager.identify_quantum_subsystem(
                self.molecular_system, config.quantum_subsystem_selection
            )
            
            # Initialize quantum Hamiltonian
            self.logger.info("Initializing quantum Hamiltonian")
            self.hamiltonian = self.quantum_engine.initialize_hamiltonian(self.quantum_subsystem)
            
            # Initialize MD system
            self.logger.info("Initializing MD system")
            md_system = self.md_engine.initialize_system(
                config.system_pdb, config.force_field
            )
            
            # Setup environment (solvation, etc.)
            self.md_engine.setup_environment(
                md_system, config.solvent_model, config.ionic_strength
            )
            
            # Energy minimization
            self.logger.info("Performing energy minimization")
            final_energy = self.md_engine.minimize_energy()
            self.logger.info(f"Energy minimization completed. Final energy: {final_energy}")
            
            # Initialize noise model
            self.logger.info(f"Initializing noise model: {config.noise_model_type}")
            self.noise_model = self.noise_factory.create_noise_model(
                config.noise_model_type, config.temperature
            )
            
            # Initialize quantum state (ground state or specified initial state)
            self.logger.info("Initializing quantum state")
            self._initialize_quantum_state()
            
            # Create output directory
            os.makedirs(config.output_directory, exist_ok=True)
            
            # Reset simulation state
            self.progress = 0.0
            self.current_time = 0.0
            self.time_step_count = 0
            self.state_trajectory = []
            self.energy_trajectory = []
            self.coherence_trajectory = {}
            
            self.logger.info("Simulation initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Simulation initialization failed: {str(e)}")
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
        
        self.logger.info("Starting simulation execution")
        self.is_running = True
        start_time = time.time()
        
        try:
            # Calculate simulation parameters
            total_steps = int(self.config.simulation_time / self.config.time_step)
            md_steps_per_qm = max(1, int(0.1 / self.config.time_step))  # MD every 0.1 fs
            
            self.logger.info(f"Running {total_steps} steps with time step {self.config.time_step}")
            
            # Main simulation loop
            for step in range(total_steps):
                if self.is_paused:
                    self.logger.info("Simulation paused")
                    break
                
                # Update progress
                self.progress = (step / total_steps) * 100.0
                self.current_time = step * self.config.time_step
                self.time_step_count = step
                
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
                
                # Log progress periodically
                if step % (total_steps // 10) == 0:
                    self.logger.info(f"Progress: {self.progress:.1f}% (step {step}/{total_steps})")
            
            # Final calculations and analysis
            self.logger.info("Performing final analysis")
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
            self._save_final_results(results)
            
            self.logger.info(f"Simulation completed successfully in {computation_time:.2f} seconds")
            self.is_running = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
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
    
    def _evolve_quantum_state(self):
        """Evolve the quantum state by one time step."""
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