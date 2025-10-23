"""
Tests for enhanced debugging with sanity check logging functionality.
"""

import pytest
import numpy as np
import logging
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from qbes.simulation_engine import SimulationEngine
from qbes.core.data_models import (
    SimulationConfig, DensityMatrix, Hamiltonian, 
    QuantumSubsystem, MolecularSystem, ValidationResult
)
from qbes.quantum_engine import QuantumEngine


class TestSanityCheckLogging:
    """Test suite for sanity check logging functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SimulationEngine()
        
        # Create a valid test configuration with debugging enabled
        self.config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophore",
            noise_model_type="dephasing",
            output_directory=tempfile.mkdtemp(),
            debug_level="DEBUG",
            save_snapshot_interval=10,
            enable_sanity_checks=True,
            dry_run_mode=False
        )
        
        # Create a valid test density matrix
        self.test_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Create a test Hamiltonian
        self.test_hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 0.1], [0.1, 1.0]], dtype=complex),
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
        
        # Set up logging capture
        self.log_stream = StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Configure the engine's logger
        self.engine.logger.setLevel(logging.DEBUG)
        self.engine.logger.addHandler(self.log_handler)
        self.engine.config = self.config
        self.engine.hamiltonian = self.test_hamiltonian
    
    def teardown_method(self):
        """Clean up after tests."""
        self.engine.logger.removeHandler(self.log_handler)
        self.log_handler.close()
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.config.output_directory):
            shutil.rmtree(self.config.output_directory)
    
    def test_sanity_checks_with_valid_state(self):
        """Test sanity check logging with a valid quantum state."""
        # Call the sanity check method
        self.engine._log_sanity_checks(self.test_state, step=1, phase="test")
        
        # Get logged output
        log_output = self.log_stream.getvalue()
        
        # Verify that all expected checks are logged
        assert "Density Matrix Trace" in log_output
        assert "PASS" in log_output  # Should pass trace check
        assert "Hermiticity Check" in log_output
        assert "Positive Semidefinite" in log_output
        assert "Purity" in log_output
        assert "Energy" in log_output
        assert "Numerical Stability" in log_output
        assert "Population Sum" in log_output
        assert "Max Coherence" in log_output
        assert "Total Coherence" in log_output
    
    def test_sanity_checks_with_invalid_trace(self):
        """Test sanity check logging with invalid trace."""
        # Create state with invalid trace by directly setting the matrix
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),  # Valid initially
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Modify the matrix after creation to have invalid trace
        invalid_state.matrix = np.array([[0.8, 0.2j], [-0.2j, 0.4]], dtype=complex)  # trace = 1.2
        
        self.engine._log_sanity_checks(invalid_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should detect trace failure
        assert "Density Matrix Trace" in log_output
        assert "FAIL" in log_output
    
    def test_sanity_checks_with_non_hermitian_state(self):
        """Test sanity check logging with non-Hermitian matrix."""
        # Create valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Modify to be non-Hermitian
        invalid_state.matrix = np.array([[0.6, 0.2], [0.3, 0.4]], dtype=complex)
        
        self.engine._log_sanity_checks(invalid_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should detect Hermiticity failure
        assert "Hermiticity Check" in log_output
        assert "FAIL" in log_output
    
    def test_sanity_checks_with_negative_eigenvalues(self):
        """Test sanity check logging with negative eigenvalues."""
        # Create valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Modify to have negative eigenvalues
        invalid_state.matrix = np.array([[0.5, 0.6], [0.6, 0.5]], dtype=complex)
        
        self.engine._log_sanity_checks(invalid_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should detect positive semidefinite failure
        assert "Positive Semidefinite" in log_output
        assert "FAIL" in log_output
    
    def test_sanity_checks_disabled(self):
        """Test that sanity checks are skipped when disabled."""
        # Disable sanity checks
        self.config.enable_sanity_checks = False
        
        self.engine._log_sanity_checks(self.test_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should have no output when disabled
        assert log_output == ""
    
    def test_sanity_checks_with_debug_level_disabled(self):
        """Test that sanity checks are skipped when DEBUG level is not enabled."""
        # Set logger to INFO level (higher than DEBUG)
        self.engine.logger.setLevel(logging.INFO)
        
        self.engine._log_sanity_checks(self.test_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should have no output when DEBUG level is not enabled
        assert log_output == ""
    
    def test_energy_conservation_monitoring(self):
        """Test energy conservation monitoring across multiple steps."""
        # First call should establish initial energy
        self.engine._log_sanity_checks(self.test_state, step=1, phase="test")
        log_output1 = self.log_stream.getvalue()
        assert "Initial Energy" in log_output1
        
        # Clear the log stream
        self.log_stream.truncate(0)
        self.log_stream.seek(0)
        
        # Second call should monitor energy change
        self.engine._log_sanity_checks(self.test_state, step=2, phase="test")
        log_output2 = self.log_stream.getvalue()
        assert "Energy =" in log_output2
        assert "change:" in log_output2
    
    def test_numerical_stability_with_nan_values(self):
        """Test numerical stability detection with NaN values."""
        # Create valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Modify to have NaN values
        invalid_state.matrix = np.array([[0.6, np.nan], [np.nan, 0.4]], dtype=complex)
        
        self.engine._log_sanity_checks(invalid_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # The function should handle the exception and log an error
        # But we should still see some checks before the exception
        assert "Density Matrix Trace" in log_output or "Error during sanity checks" in log_output
    
    def test_numerical_stability_with_inf_values(self):
        """Test numerical stability detection with infinite values."""
        # Create valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Modify to have infinite values
        invalid_state.matrix = np.array([[0.6, np.inf], [-np.inf, 0.4]], dtype=complex)
        
        self.engine._log_sanity_checks(invalid_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # The function should handle the exception and log an error
        # But we should still see some checks before the exception
        assert "Density Matrix Trace" in log_output or "Error during sanity checks" in log_output
    
    def test_condition_number_monitoring(self):
        """Test matrix condition number monitoring."""
        self.engine._log_sanity_checks(self.test_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should include condition number
        assert "Condition Number" in log_output
        assert ("GOOD" in log_output or "POOR" in log_output)
    
    def test_coherence_monitoring(self):
        """Test coherence magnitude monitoring."""
        self.engine._log_sanity_checks(self.test_state, step=1, phase="test")
        log_output = self.log_stream.getvalue()
        
        # Should include coherence measures
        assert "Max Coherence" in log_output
        assert "Total Coherence" in log_output
    
    def test_critical_failure_warning(self):
        """Test that critical failures trigger warnings."""
        # Create valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Modify to have trace failure (which should trigger critical failure warning)
        invalid_state.matrix = np.array([[0.8, 0.2j], [-0.2j, 0.4]], dtype=complex)  # trace = 1.2
        
        # Capture warnings as well
        with patch.object(self.engine.logger, 'warning') as mock_warning:
            self.engine._log_sanity_checks(invalid_state, step=1, phase="test")
            
            # Should have called warning for critical failures
            mock_warning.assert_called()
            warning_call = mock_warning.call_args[0][0]
            assert "CRITICAL SANITY CHECK FAILURES" in warning_call
    
    def test_sanity_check_exception_handling(self):
        """Test that exceptions during sanity checks are handled gracefully."""
        # Create a mock state that will cause an exception during trace calculation
        mock_state = Mock()
        mock_state.matrix = Mock()
        mock_state.matrix.shape = (2, 2)
        
        # Make np.trace raise an exception
        with patch('numpy.trace', side_effect=Exception("Test exception")):
            # Should not raise exception, but log error
            with patch.object(self.engine.logger, 'error') as mock_error:
                self.engine._log_sanity_checks(mock_state, step=1, phase="test")
                mock_error.assert_called()
                error_call = mock_error.call_args[0][0]
                assert "Error during sanity checks" in error_call


class TestStateSnapshotFunctionality:
    """Test suite for state snapshot functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SimulationEngine()
        self.temp_dir = tempfile.mkdtemp()
        
        self.config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophore",
            noise_model_type="dephasing",
            output_directory=self.temp_dir,
            debug_level="DEBUG",
            save_snapshot_interval=10,
            enable_sanity_checks=True,
            dry_run_mode=False
        )
        
        self.test_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=1e-15
        )
        
        self.test_hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 0.1], [0.1, 1.0]], dtype=complex),
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
        
        self.engine.config = self.config
        self.engine.hamiltonian = self.test_hamiltonian
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_state_snapshot(self):
        """Test saving state snapshots."""
        step = 100
        self.engine._save_state_snapshot(self.test_state, step)
        
        # Check that snapshot file was created
        snapshots_dir = os.path.join(self.temp_dir, "snapshots")
        assert os.path.exists(snapshots_dir)
        
        snapshot_file = os.path.join(snapshots_dir, f"snapshot_step_{step:08d}.pkl")
        assert os.path.exists(snapshot_file)
        
        # Load and verify snapshot content
        import pickle
        with open(snapshot_file, 'rb') as f:
            snapshot_data = pickle.load(f)
        
        assert snapshot_data['step'] == step
        assert snapshot_data['time'] == self.test_state.time
        assert np.allclose(snapshot_data['density_matrix'], self.test_state.matrix)
        assert snapshot_data['basis_labels'] == self.test_state.basis_labels
        assert 'metadata' in snapshot_data
        assert 'trace' in snapshot_data['metadata']
        assert 'purity' in snapshot_data['metadata']
        assert 'populations' in snapshot_data['metadata']
        assert 'energy' in snapshot_data['metadata']
    
    def test_save_snapshot_without_config(self):
        """Test that snapshot saving handles missing config gracefully."""
        self.engine.config = None
        
        # Should not raise exception
        self.engine._save_state_snapshot(self.test_state, 100)
        
        # No snapshot should be created
        snapshots_dir = os.path.join(self.temp_dir, "snapshots")
        assert not os.path.exists(snapshots_dir)
    
    def test_save_snapshot_with_io_error(self):
        """Test snapshot saving with I/O errors."""
        # Mock the pickle.dump to raise an exception
        with patch('pickle.dump', side_effect=IOError("Permission denied")):
            # Should handle I/O error gracefully
            with patch.object(self.engine.logger, 'warning') as mock_warning:
                self.engine._save_state_snapshot(self.test_state, 100)
                mock_warning.assert_called()
                warning_call = mock_warning.call_args[0][0]
                assert "Failed to save state snapshot" in warning_call


class TestEnhancedQuantumStateEvolution:
    """Test suite for enhanced quantum state evolution with debugging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SimulationEngine()
        
        # Mock the required components
        self.engine.quantum_engine = Mock(spec=QuantumEngine)
        self.engine.noise_model = Mock()
        self.engine.quantum_subsystem = Mock()
        
        # Create test configuration
        self.config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophore",
            noise_model_type="dephasing",
            output_directory=tempfile.mkdtemp(),
            debug_level="DEBUG",
            save_snapshot_interval=5,
            enable_sanity_checks=True,
            dry_run_mode=False
        )
        
        self.engine.config = self.config
        self.engine.time_step_count = 10
        self.engine.current_time = 1e-14
        self.engine.state_trajectory = []
        
        # Create test states
        self.initial_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        self.evolved_state = DensityMatrix(
            matrix=np.array([[0.65, 0.15j], [-0.15j, 0.35]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=1e-15
        )
        
        self.engine.current_state = self.initial_state
        
        # Mock the quantum engine methods
        self.engine.quantum_engine.evolve_state.return_value = self.evolved_state
        self.engine.quantum_engine.validate_quantum_state.return_value = ValidationResult(is_valid=True)
        self.engine.noise_model.generate_lindblad_operators.return_value = []
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists(self.config.output_directory):
            shutil.rmtree(self.config.output_directory)
    
    def test_evolve_quantum_state_with_debugging(self):
        """Test quantum state evolution with debugging enabled."""
        with patch.object(self.engine, '_log_sanity_checks') as mock_log_checks:
            with patch.object(self.engine, '_save_state_snapshot') as mock_save_snapshot:
                self.engine._evolve_quantum_state()
                
                # Should call sanity checks twice (pre and post evolution)
                assert mock_log_checks.call_count == 2
                
                # Check the calls
                calls = mock_log_checks.call_args_list
                assert calls[0][0][2] == "pre-evolution"  # phase argument
                assert calls[1][0][2] == "post-evolution"  # phase argument
                
                # Should call snapshot saving (step 10 is divisible by interval 5)
                mock_save_snapshot.assert_called_once()
    
    def test_evolve_quantum_state_without_snapshot_interval(self):
        """Test quantum state evolution when snapshot interval doesn't match."""
        self.engine.time_step_count = 7  # Not divisible by 5
        
        with patch.object(self.engine, '_log_sanity_checks') as mock_log_checks:
            with patch.object(self.engine, '_save_state_snapshot') as mock_save_snapshot:
                self.engine._evolve_quantum_state()
                
                # Should still call sanity checks
                assert mock_log_checks.call_count == 2
                
                # Should not call snapshot saving
                mock_save_snapshot.assert_not_called()
    
    def test_evolve_quantum_state_with_disabled_snapshots(self):
        """Test quantum state evolution with snapshots disabled."""
        self.config.save_snapshot_interval = 0  # Disabled
        
        with patch.object(self.engine, '_log_sanity_checks') as mock_log_checks:
            with patch.object(self.engine, '_save_state_snapshot') as mock_save_snapshot:
                self.engine._evolve_quantum_state()
                
                # Should still call sanity checks
                assert mock_log_checks.call_count == 2
                
                # Should not call snapshot saving
                mock_save_snapshot.assert_not_called()
    
    def test_evolve_quantum_state_updates_trajectory(self):
        """Test that quantum state evolution properly updates the trajectory."""
        initial_trajectory_length = len(self.engine.state_trajectory)
        
        self.engine._evolve_quantum_state()
        
        # Should add the evolved state to trajectory
        assert len(self.engine.state_trajectory) == initial_trajectory_length + 1
        assert self.engine.state_trajectory[-1] == self.evolved_state
        assert self.engine.current_state == self.evolved_state


if __name__ == "__main__":
    pytest.main([__file__])