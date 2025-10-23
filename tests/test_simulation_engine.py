"""
Integration tests for the simulation engine.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from qbes.simulation_engine import SimulationEngine
from qbes.core.data_models import (
    SimulationConfig, ValidationResult, MolecularSystem, QuantumSubsystem,
    Atom, QuantumState, DensityMatrix, Hamiltonian, CoherenceMetrics,
    StatisticalSummary, SimulationResults
)


class TestSimulationEngine:
    """Test cases for the SimulationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SimulationEngine()
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = SimulationConfig(
            system_pdb="test_system.pdb",
            temperature=300.0,
            simulation_time=1.0,  # 1 fs for fast testing
            time_step=0.1,
            quantum_subsystem_selection="residue 1",
            noise_model_type="protein",
            output_directory=self.temp_dir,
            force_field="amber14",
            solvent_model="tip3p",
            ionic_strength=0.15,
            # New debugging features
            debug_level="DEBUG",
            save_snapshot_interval=0,
            enable_sanity_checks=True,
            dry_run_mode=False
        )
        
        # Create mock molecular system
        self.mock_atoms = [
            Atom("C", np.array([0.0, 0.0, 0.0]), 0.0, 12.0, 1),
            Atom("N", np.array([1.0, 0.0, 0.0]), 0.0, 14.0, 2)
        ]
        
        self.mock_molecular_system = MolecularSystem(
            atoms=self.mock_atoms,
            bonds=[(0, 1)],
            residues={1: "TEST"},
            system_name="test_system"
        )
        
        # Create mock quantum subsystem
        basis_states = [
            QuantumState(np.array([1.0, 0.0]), ["ground", "excited"]),
            QuantumState(np.array([0.0, 1.0]), ["ground", "excited"])
        ]
        
        self.mock_quantum_subsystem = QuantumSubsystem(
            atoms=self.mock_atoms,
            hamiltonian_parameters={"coupling": 0.1},
            coupling_matrix=np.array([[0.0, 0.1], [0.1, 1.0]]),
            basis_states=basis_states
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test simulation engine initialization."""
        assert self.engine.config is None
        assert self.engine.progress == 0.0
        assert not self.engine.is_paused
        assert not self.engine.is_running
        assert self.engine.current_time == 0.0
        assert self.engine.time_step_count == 0
    
    @patch('qbes.simulation_engine.ConfigurationManager')
    @patch('qbes.simulation_engine.MDEngine')
    @patch('qbes.simulation_engine.NoiseModelFactory')
    def test_initialize_simulation_success(self, mock_noise_factory, mock_md_engine, mock_config_manager):
        """Test successful simulation initialization."""
        # Setup mocks
        mock_config_manager_instance = Mock()
        mock_config_manager.return_value = mock_config_manager_instance
        
        mock_config_manager_instance.validate_parameters.return_value = ValidationResult(is_valid=True)
        mock_config_manager_instance.parse_pdb.return_value = self.mock_molecular_system
        mock_config_manager_instance.identify_quantum_subsystem.return_value = self.mock_quantum_subsystem
        
        mock_md_engine_instance = Mock()
        mock_md_engine.return_value = mock_md_engine_instance
        mock_md_engine_instance.initialize_system.return_value = self.mock_molecular_system
        mock_md_engine_instance.setup_environment.return_value = True
        mock_md_engine_instance.minimize_energy.return_value = -100.0
        
        mock_noise_factory_instance = Mock()
        mock_noise_factory.return_value = mock_noise_factory_instance
        mock_noise_model = Mock()
        mock_noise_factory_instance.create_noise_model.return_value = mock_noise_model
        
        # Replace engine components with mocks
        self.engine.config_manager = mock_config_manager_instance
        self.engine.md_engine = mock_md_engine_instance
        self.engine.noise_factory = mock_noise_factory_instance
        
        # Mock quantum engine
        mock_hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 0.1], [0.1, 1.0]]),
            basis_labels=["ground", "excited"]
        )
        self.engine.quantum_engine.initialize_hamiltonian = Mock(return_value=mock_hamiltonian)
        
        # Test initialization
        result = self.engine.initialize_simulation(self.test_config)
        
        assert result.is_valid
        assert self.engine.config == self.test_config
        assert self.engine.molecular_system == self.mock_molecular_system
        assert self.engine.quantum_subsystem == self.mock_quantum_subsystem
        assert self.engine.hamiltonian == mock_hamiltonian
        assert self.engine.current_state is not None
        assert len(self.engine.state_trajectory) == 1
    
    @patch('qbes.simulation_engine.ConfigurationManager')
    def test_initialize_simulation_validation_failure(self, mock_config_manager):
        """Test simulation initialization with validation failure."""
        # Setup mock to return validation failure
        mock_config_manager_instance = Mock()
        mock_config_manager.return_value = mock_config_manager_instance
        
        validation_result = ValidationResult(is_valid=False)
        validation_result.add_error("Invalid temperature")
        mock_config_manager_instance.validate_parameters.return_value = validation_result
        
        self.engine.config_manager = mock_config_manager_instance
        
        # Test initialization
        result = self.engine.initialize_simulation(self.test_config)
        
        assert not result.is_valid
        assert "Invalid temperature" in result.errors
    
    def test_run_simulation_not_initialized(self):
        """Test running simulation without initialization."""
        with pytest.raises(RuntimeError, match="Simulation not initialized"):
            self.engine.run_simulation()
    
    @patch('qbes.simulation_engine.time.time')
    def test_run_simulation_success(self, mock_time):
        """Test successful simulation execution."""
        # Mock time for computation time calculation
        mock_time.side_effect = [0.0, 10.0]  # Start and end times
        
        # Setup initialized engine
        self._setup_initialized_engine()
        
        # Mock analyzer
        mock_statistical_summary = StatisticalSummary(
            mean_values={"energy": -50.0},
            std_deviations={"energy": 5.0},
            confidence_intervals={"energy": (-55.0, -45.0)},
            sample_size=10
        )
        self.engine.analyzer.generate_statistical_summary = Mock(return_value=mock_statistical_summary)
        
        # Run simulation
        results = self.engine.run_simulation()
        
        # Verify results
        assert isinstance(results, SimulationResults)
        assert results.computation_time == 10.0
        assert len(results.state_trajectory) > 0
        assert len(results.energy_trajectory) > 0
        assert results.simulation_config == self.test_config
        assert not self.engine.is_running
    
    def test_pause_and_resume_simulation(self):
        """Test pausing and resuming simulation."""
        # Test pause when not running
        assert not self.engine.pause_simulation()
        
        # Test pause when running
        self.engine.is_running = True
        assert self.engine.pause_simulation()
        assert self.engine.is_paused
        
        # Test resume
        assert self.engine.resume_simulation()
        assert not self.engine.is_paused
        
        # Test resume when not paused
        assert not self.engine.resume_simulation()
    
    def test_save_and_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        # Setup some simulation state
        self._setup_initialized_engine()
        self.engine.progress = 50.0
        self.engine.current_time = 0.5
        self.engine.time_step_count = 5
        
        # Test save checkpoint
        checkpoint_file = os.path.join(self.temp_dir, "test_checkpoint.pkl")
        assert self.engine.save_checkpoint(checkpoint_file)
        assert os.path.exists(checkpoint_file)
        
        # Modify state
        self.engine.progress = 0.0
        self.engine.current_time = 0.0
        
        # Test load checkpoint
        assert self.engine.load_checkpoint(checkpoint_file)
        assert self.engine.progress == 50.0
        assert self.engine.current_time == 0.5
        assert self.engine.time_step_count == 5
    
    def test_save_checkpoint_failure(self):
        """Test checkpoint saving failure."""
        # Try to save to invalid path
        invalid_path = "/invalid/path/checkpoint.pkl"
        assert not self.engine.save_checkpoint(invalid_path)
    
    def test_load_checkpoint_failure(self):
        """Test checkpoint loading failure."""
        # Try to load non-existent file
        nonexistent_file = "nonexistent_checkpoint.pkl"
        assert not self.engine.load_checkpoint(nonexistent_file)
    
    def test_get_progress(self):
        """Test progress reporting."""
        assert self.engine.get_progress() == 0.0
        
        self.engine.progress = 75.5
        assert self.engine.get_progress() == 75.5
    
    def test_initialize_quantum_state(self):
        """Test quantum state initialization."""
        # Setup quantum subsystem
        self.engine.quantum_subsystem = self.mock_quantum_subsystem
        
        # Initialize state
        self.engine._initialize_quantum_state()
        
        # Verify state
        assert self.engine.current_state is not None
        assert len(self.engine.state_trajectory) == 1
        
        # Check that it's a ground state (first coefficient = 1)
        state_matrix = self.engine.current_state.matrix
        assert np.isclose(state_matrix[0, 0], 1.0)
        assert np.isclose(np.trace(state_matrix), 1.0)
    
    def test_calculate_observables(self):
        """Test observable calculations."""
        # Setup initialized engine
        self._setup_initialized_engine()
        
        # Mock coherence metrics
        mock_coherence = CoherenceMetrics(
            coherence_lifetime=1.0,
            quantum_discord=0.5,
            entanglement_measure=0.3,
            purity=0.8,
            von_neumann_entropy=0.2
        )
        self.engine.quantum_engine.calculate_coherence_measures = Mock(return_value=mock_coherence)
        
        # Calculate observables
        self.engine._calculate_observables()
        
        # Verify results stored
        assert len(self.engine.energy_trajectory) == 1
        assert 'coherence_lifetime' in self.engine.coherence_trajectory
        assert 'purity' in self.engine.coherence_trajectory
    
    def test_dry_run_validation_success(self):
        """Test successful dry-run validation."""
        # Mock all required components for dry-run
        with patch('qbes.simulation_engine.ConfigurationManager') as mock_config_manager, \
             patch('qbes.simulation_engine.MDEngine') as mock_md_engine, \
             patch('qbes.simulation_engine.NoiseModelFactory') as mock_noise_factory:
            
            # Setup mocks
            mock_config_manager_instance = Mock()
            mock_config_manager.return_value = mock_config_manager_instance
            mock_config_manager_instance.validate_parameters.return_value = ValidationResult(is_valid=True)
            mock_config_manager_instance.parse_pdb.return_value = self.mock_molecular_system
            mock_config_manager_instance.identify_quantum_subsystem.return_value = self.mock_quantum_subsystem
            
            mock_md_engine_instance = Mock()
            mock_md_engine.return_value = mock_md_engine_instance
            mock_md_engine_instance.initialize_system.return_value = self.mock_molecular_system
            mock_md_engine_instance.setup_environment.return_value = True
            
            mock_noise_factory_instance = Mock()
            mock_noise_factory.return_value = mock_noise_factory_instance
            mock_noise_model = Mock()
            mock_noise_factory_instance.create_noise_model.return_value = mock_noise_model
            
            # Set dry-run mode
            dry_run_config = SimulationConfig(
                system_pdb="test_system.pdb",
                temperature=300.0,
                simulation_time=1.0,
                time_step=0.1,
                quantum_subsystem_selection="residue 1",
                noise_model_type="protein",
                output_directory=self.temp_dir,
                dry_run_mode=True
            )
            
            # Test dry-run validation
            result = self.engine.perform_dry_run_validation(dry_run_config)
            
            assert result.is_valid
            assert "Dry-run validation completed successfully" in str(result)
    
    def test_dry_run_validation_failure(self):
        """Test dry-run validation with configuration errors."""
        # Create invalid config (missing required fields)
        invalid_config = SimulationConfig(
            system_pdb="nonexistent.pdb",
            temperature=-100.0,  # Invalid temperature
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="invalid selection",
            noise_model_type="invalid",
            output_directory="/invalid/path",
            dry_run_mode=True
        )
        
        # Test dry-run validation
        result = self.engine.perform_dry_run_validation(invalid_config)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    @patch('qbes.simulation_engine.logging')
    def test_sanity_checks_logging(self, mock_logging):
        """Test sanity check logging functionality."""
        # Setup initialized engine with DEBUG logging
        self._setup_initialized_engine()
        self.engine.config.enable_sanity_checks = True
        
        # Mock logger to capture DEBUG messages
        mock_logger = Mock()
        mock_logger.isEnabledFor.return_value = True
        self.engine.logger = mock_logger
        
        # Create test density matrix
        test_state = DensityMatrix(
            matrix=np.array([[0.9, 0.1], [0.1, 0.1]]),
            basis_labels=["ground", "excited"],
            time=0.1
        )
        
        # Test sanity checks logging
        self.engine._log_sanity_checks(test_state, 10, "test-phase")
        
        # Verify DEBUG logging was called
        mock_logger.debug.assert_called()
        
        # Check that trace validation was logged
        debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
        trace_logged = any("Density Matrix Trace" in call for call in debug_calls)
        assert trace_logged
    
    def test_state_snapshot_saving(self):
        """Test state snapshot saving functionality."""
        # Setup initialized engine
        self._setup_initialized_engine()
        self.engine.config.save_snapshot_interval = 5
        
        # Create test density matrix
        test_state = DensityMatrix(
            matrix=np.array([[0.9, 0.1], [0.1, 0.1]]),
            basis_labels=["ground", "excited"],
            time=0.5
        )
        
        # Test snapshot saving
        self.engine._save_state_snapshot(test_state, 10)
        
        # Verify snapshot file was created
        snapshots_dir = os.path.join(self.temp_dir, "snapshots")
        assert os.path.exists(snapshots_dir)
        
        snapshot_file = os.path.join(snapshots_dir, "snapshot_step_00000010.pkl")
        assert os.path.exists(snapshot_file)
        
        # Test loading the snapshot
        loaded_data = self.engine.load_state_snapshot(snapshot_file)
        
        assert loaded_data['step'] == 10
        assert loaded_data['time'] == 0.5
        assert 'density_matrix' in loaded_data
        assert 'purity' in loaded_data
        assert 'trace' in loaded_data
    
    def test_state_snapshot_loading_nonexistent(self):
        """Test error handling for loading nonexistent snapshot."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_snapshot.pkl")
        
        with pytest.raises(FileNotFoundError):
            self.engine.load_state_snapshot(nonexistent_path)
    
    def test_enhanced_debugging_integration(self):
        """Test integration of enhanced debugging features during simulation."""
        # Setup initialized engine with debugging enabled
        self._setup_initialized_engine()
        self.engine.config.enable_sanity_checks = True
        self.engine.config.save_snapshot_interval = 2
        
        # Mock logger for DEBUG level
        mock_logger = Mock()
        mock_logger.isEnabledFor.return_value = True
        self.engine.logger = mock_logger
        
        # Test quantum state evolution with debugging
        self.engine._evolve_quantum_state()
        
        # Verify sanity checks were called
        debug_calls = [call.args[0] for call in mock_logger.debug.call_args_list]
        pre_evolution_logged = any("pre-evolution" in call for call in debug_calls)
        post_evolution_logged = any("post-evolution" in call for call in debug_calls)
        
        assert pre_evolution_logged
        assert post_evolution_logged
    
    def test_numerical_stability_monitoring(self):
        """Test numerical stability monitoring in sanity checks."""
        # Setup initialized engine
        self._setup_initialized_engine()
        self.engine.config.enable_sanity_checks = True
        
        # Mock logger
        mock_logger = Mock()
        mock_logger.isEnabledFor.return_value = True
        self.engine.logger = mock_logger
        
        # Create numerically unstable state (non-Hermitian)
        unstable_matrix = np.array([[1.0, 0.5j], [0.3j, 0.0]])  # Non-Hermitian
        unstable_state = DensityMatrix(
            matrix=unstable_matrix,
            basis_labels=["ground", "excited"],
            time=0.1
        )
        
        # Test sanity checks with unstable state
        self.engine._log_sanity_checks(unstable_state, 5, "stability-test")
        
        # Verify warning was logged for non-Hermitian matrix
        warning_calls = [call.args[0] for call in mock_logger.warning.call_args_list]
        hermiticity_warning = any("Hermiticity" in call for call in warning_calls)
        assert hermiticity_warning
    
    def _setup_initialized_engine(self):
        """Helper method to setup an initialized engine for testing."""
        self.engine.config = self.test_config
        self.engine.molecular_system = self.mock_molecular_system
        self.engine.quantum_subsystem = self.mock_quantum_subsystem
        
        # Setup hamiltonian
        self.engine.hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 0.1], [0.1, 1.0]]),
            basis_labels=["ground", "excited"]
        )
        
        # Setup noise model
        self.engine.noise_model = Mock()
        self.engine.noise_model.generate_lindblad_operators.return_value = []
        self.engine.noise_model.calculate_decoherence_rates.return_value = {"dephasing": 0.1}
        
        # Initialize quantum state
        self.engine._initialize_quantum_state()
        
        # Mock MD engine methods
        self.engine.md_engine.run_trajectory = Mock(return_value={"coordinates": np.zeros((10, 3))})
        self.engine.md_engine.extract_quantum_parameters = Mock(return_value={"coupling": np.ones(10)})
        
        # Mock quantum engine methods
        mock_evolved_state = DensityMatrix(
            matrix=np.array([[0.9, 0.1], [0.1, 0.1]]),
            basis_labels=["ground", "excited"],
            time=0.1
        )
        self.engine.quantum_engine.evolve_state = Mock(return_value=mock_evolved_state)
        self.engine.quantum_engine.validate_quantum_state = Mock(return_value=ValidationResult(is_valid=True))
        
        mock_coherence = CoherenceMetrics(
            coherence_lifetime=1.0,
            quantum_discord=0.5,
            entanglement_measure=0.3,
            purity=0.8,
            von_neumann_entropy=0.2
        )
        self.engine.quantum_engine.calculate_coherence_measures = Mock(return_value=mock_coherence)


class TestSimulationEngineIntegration:
    """Integration tests for complete simulation pipeline."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('qbes.simulation_engine.ConfigurationManager')
    @patch('qbes.simulation_engine.MDEngine')
    @patch('qbes.simulation_engine.NoiseModelFactory')
    @patch('qbes.simulation_engine.QuantumEngine')
    @patch('qbes.simulation_engine.ResultsAnalyzer')
    def test_complete_simulation_pipeline(self, mock_analyzer, mock_quantum_engine, 
                                        mock_noise_factory, mock_md_engine, mock_config_manager):
        """Test complete simulation pipeline integration."""
        # This test verifies that all components work together correctly
        
        # Setup configuration
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=0.5,  # Very short for testing
            time_step=0.1,
            quantum_subsystem_selection="all",
            noise_model_type="protein",
            output_directory=self.temp_dir
        )
        
        # Setup all mocks to return valid data
        self._setup_integration_mocks(
            mock_config_manager, mock_md_engine, mock_noise_factory,
            mock_quantum_engine, mock_analyzer
        )
        
        # Create and run simulation
        engine = SimulationEngine()
        
        # Initialize
        init_result = engine.initialize_simulation(config)
        assert init_result.is_valid
        
        # Run simulation
        results = engine.run_simulation()
        
        # Verify results
        assert isinstance(results, SimulationResults)
        assert len(results.state_trajectory) > 0
        assert len(results.energy_trajectory) > 0
        assert results.computation_time > 0
        
        # Verify output files were created
        assert os.path.exists(os.path.join(self.temp_dir, "simulation_results.pkl"))
        assert os.path.exists(os.path.join(self.temp_dir, "simulation_summary.txt"))
    
    def _setup_integration_mocks(self, mock_config_manager, mock_md_engine, 
                               mock_noise_factory, mock_quantum_engine, mock_analyzer):
        """Setup mocks for integration testing."""
        # Configuration manager mocks
        mock_config_manager_instance = Mock()
        mock_config_manager.return_value = mock_config_manager_instance
        mock_config_manager_instance.validate_parameters.return_value = ValidationResult(is_valid=True)
        
        # Create mock molecular system
        mock_atoms = [Atom("C", np.array([0.0, 0.0, 0.0]), 0.0, 12.0, 1)]
        mock_molecular_system = MolecularSystem(atoms=mock_atoms, bonds=[], residues={})
        mock_config_manager_instance.parse_pdb.return_value = mock_molecular_system
        
        # Create mock quantum subsystem
        basis_states = [QuantumState(np.array([1.0, 0.0]), ["ground", "excited"])]
        mock_quantum_subsystem = QuantumSubsystem(
            atoms=mock_atoms,
            hamiltonian_parameters={},
            coupling_matrix=np.array([[0.0, 0.1], [0.1, 1.0]]),
            basis_states=basis_states
        )
        mock_config_manager_instance.identify_quantum_subsystem.return_value = mock_quantum_subsystem
        
        # MD engine mocks
        mock_md_engine_instance = Mock()
        mock_md_engine.return_value = mock_md_engine_instance
        mock_md_engine_instance.initialize_system.return_value = mock_molecular_system
        mock_md_engine_instance.setup_environment.return_value = True
        mock_md_engine_instance.minimize_energy.return_value = -100.0
        mock_md_engine_instance.run_trajectory.return_value = {"coordinates": np.zeros((10, 3))}
        mock_md_engine_instance.extract_quantum_parameters.return_value = {"coupling": np.ones(10)}
        
        # Noise factory mocks
        mock_noise_factory_instance = Mock()
        mock_noise_factory.return_value = mock_noise_factory_instance
        mock_noise_model = Mock()
        mock_noise_model.generate_lindblad_operators.return_value = []
        mock_noise_model.calculate_decoherence_rates.return_value = {"dephasing": 0.1}
        mock_noise_factory_instance.create_noise_model.return_value = mock_noise_model
        
        # Quantum engine mocks
        mock_quantum_engine_instance = Mock()
        mock_quantum_engine.return_value = mock_quantum_engine_instance
        
        mock_hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 0.1], [0.1, 1.0]]),
            basis_labels=["ground", "excited"]
        )
        mock_quantum_engine_instance.initialize_hamiltonian.return_value = mock_hamiltonian
        
        mock_evolved_state = DensityMatrix(
            matrix=np.array([[0.9, 0.1], [0.1, 0.1]]),
            basis_labels=["ground", "excited"]
        )
        mock_quantum_engine_instance.evolve_state.return_value = mock_evolved_state
        mock_quantum_engine_instance.validate_quantum_state.return_value = ValidationResult(is_valid=True)
        
        mock_coherence = CoherenceMetrics(
            coherence_lifetime=1.0, quantum_discord=0.5, entanglement_measure=0.3,
            purity=0.8, von_neumann_entropy=0.2
        )
        mock_quantum_engine_instance.calculate_coherence_measures.return_value = mock_coherence
        
        # Analyzer mocks
        mock_analyzer_instance = Mock()
        mock_analyzer.return_value = mock_analyzer_instance
        mock_statistical_summary = StatisticalSummary(
            mean_values={"energy": -50.0}, std_deviations={"energy": 5.0},
            confidence_intervals={"energy": (-55.0, -45.0)}, sample_size=5
        )
        mock_analyzer_instance.generate_statistical_summary.return_value = mock_statistical_summary


if __name__ == "__main__":
    pytest.main([__file__])