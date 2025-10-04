"""
Tests for error handling and validation functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from qbes.utils.error_handling import ErrorHandler
from qbes.core.data_models import (
    SimulationResults, ValidationResult, DensityMatrix, 
    CoherenceMetrics, StatisticalSummary, SimulationConfig
)


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        
        # Create test density matrices
        self.valid_density_matrix = DensityMatrix(
            matrix=np.array([[0.6, 0.2], [0.2, 0.4]]),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Create test simulation results
        self.test_config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="all",
            noise_model_type="protein",
            output_directory="/tmp"
        )
        
        self.valid_results = SimulationResults(
            state_trajectory=[self.valid_density_matrix],
            coherence_measures={
                "purity": [0.8],
                "von_neumann_entropy": [0.2],
                "coherence_lifetime": [1.0]
            },
            energy_trajectory=[-100.0],
            decoherence_rates={"dephasing": 0.1},
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=1
            ),
            simulation_config=self.test_config,
            computation_time=10.0
        )
    
    def test_initialization(self):
        """Test error handler initialization."""
        assert len(self.error_handler.error_log) == 0
        assert isinstance(self.error_handler.recovery_strategies, dict)
        assert 'convergence_failure' in self.error_handler.recovery_strategies
        assert 'resource_exhaustion' in self.error_handler.recovery_strategies
    
    def test_handle_convergence_failure(self):
        """Test convergence failure handling."""
        simulation_state = {
            'time_step': 1.0,  # Large time step
            'system_size': 100,
            'current_energy': 50.0
        }
        
        action = self.error_handler.handle_convergence_failure(simulation_state)
        
        assert action == "reduce_time_step"
        assert len(self.error_handler.error_log) == 1
        assert self.error_handler.error_log[0]['error_type'] == 'convergence_failure'
    
    def test_handle_convergence_failure_large_system(self):
        """Test convergence failure handling for large systems."""
        simulation_state = {
            'time_step': 0.1,
            'system_size': 15000,  # Very large system
            'current_energy': 50.0
        }
        
        action = self.error_handler.handle_convergence_failure(simulation_state)
        
        assert action == "reduce_system_size"
    
    def test_handle_convergence_failure_unstable_energy(self):
        """Test convergence failure handling for unstable energy."""
        simulation_state = {
            'time_step': 0.1,
            'system_size': 100,
            'current_energy': 1e8  # Very high energy
        }
        
        action = self.error_handler.handle_convergence_failure(simulation_state)
        
        assert action == "reset_initial_conditions"
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_handle_resource_exhaustion_critical_memory(self, mock_cpu, mock_memory):
        """Test resource exhaustion handling for critical memory usage."""
        # Mock critical memory usage
        mock_memory.return_value = Mock(percent=98.0, available=1024**3)  # 98% usage
        mock_cpu.return_value = 50.0
        
        recommendations = self.error_handler.handle_resource_exhaustion(1000)
        
        assert recommendations['action'] == 'aggressive_reduction'
        assert recommendations['new_system_size'] == 250  # 1000 // 4
        assert recommendations['memory_usage'] == 98.0
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_handle_resource_exhaustion_warning_memory(self, mock_cpu, mock_memory):
        """Test resource exhaustion handling for warning level memory usage."""
        # Mock warning level memory usage
        mock_memory.return_value = Mock(percent=85.0, available=2*1024**3)  # 85% usage
        mock_cpu.return_value = 50.0
        
        recommendations = self.error_handler.handle_resource_exhaustion(1000)
        
        assert recommendations['action'] == 'moderate_reduction'
        assert recommendations['new_system_size'] == 500  # 1000 // 2
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_handle_resource_exhaustion_large_system(self, mock_cpu, mock_memory):
        """Test resource exhaustion handling for oversized systems."""
        # Mock normal memory usage but large system
        mock_memory.return_value = Mock(percent=50.0, available=4*1024**3)
        mock_cpu.return_value = 30.0
        
        recommendations = self.error_handler.handle_resource_exhaustion(15000)
        
        assert recommendations['action'] == 'size_optimization'
        assert recommendations['new_system_size'] == 10000  # max_system_size
    
    def test_handle_unphysical_results_valid(self):
        """Test handling of valid physical results."""
        validation_result = self.error_handler.handle_unphysical_results(self.valid_results)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    def test_handle_unphysical_results_invalid_states(self):
        """Test handling of results with invalid quantum states."""
        # Create a valid density matrix first, then modify it to be invalid
        valid_matrix = DensityMatrix(
            matrix=np.array([[0.6, 0.2], [0.2, 0.4]]),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Create results with the valid matrix
        invalid_results = SimulationResults(
            state_trajectory=[valid_matrix],
            coherence_measures={"purity": [0.8]},
            energy_trajectory=[-100.0],
            decoherence_rates={},
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=1
            ),
            simulation_config=self.test_config,
            computation_time=10.0
        )
        
        # Now modify the matrix to be invalid (bypass validation)
        invalid_results.state_trajectory[0].matrix = np.array([[0.6, 0.2], [0.3, 0.4]])
        
        validation_result = self.error_handler.handle_unphysical_results(invalid_results)
        
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
    
    def test_handle_unphysical_results_invalid_coherence(self):
        """Test handling of results with invalid coherence measures."""
        invalid_coherence_results = SimulationResults(
            state_trajectory=[self.valid_density_matrix],
            coherence_measures={
                "purity": [1.5],  # Invalid purity > 1
                "von_neumann_entropy": [-0.1],  # Invalid negative entropy
                "coherence_lifetime": [-1.0]  # Invalid negative lifetime
            },
            energy_trajectory=[-100.0],
            decoherence_rates={},
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=1
            ),
            simulation_config=self.test_config,
            computation_time=10.0
        )
        
        validation_result = self.error_handler.handle_unphysical_results(invalid_coherence_results)
        
        assert not validation_result.is_valid
        assert len(validation_result.errors) >= 3  # Three invalid measures
    
    def test_generate_diagnostic_report(self):
        """Test diagnostic report generation."""
        error_type = "convergence_failure"
        context = {
            "time_step": 0.1,
            "system_size": 1000,
            "current_energy": 100.0
        }
        
        report = self.error_handler.generate_diagnostic_report(error_type, context)
        
        assert "QBES Diagnostic Report" in report
        assert error_type in report
        assert "time_step: 0.1" in report
        assert "System Information:" in report
        assert "Suggested Recovery Actions:" in report
    
    def test_suggest_recovery_actions(self):
        """Test recovery action suggestions."""
        # Test known error type
        actions = self.error_handler.suggest_recovery_actions("convergence_failure")
        assert len(actions) > 0
        assert "Reduce time step" in actions[0]
        
        # Test unknown error type
        actions = self.error_handler.suggest_recovery_actions("unknown_error")
        assert len(actions) > 0
        assert "Check system configuration" in actions[0]
    
    def test_monitor_numerical_stability_valid_state(self):
        """Test numerical stability monitoring for valid states."""
        validation_result = self.error_handler.monitor_numerical_stability(self.valid_density_matrix)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    def test_monitor_numerical_stability_invalid_trace(self):
        """Test numerical stability monitoring for invalid trace."""
        # Create a valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2], [0.2, 0.4]]),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Bypass validation by directly setting matrix with invalid trace
        invalid_state.matrix = np.array([[0.7, 0.2], [0.2, 0.4]])  # trace = 1.1
        
        validation_result = self.error_handler.monitor_numerical_stability(invalid_state)
        
        assert not validation_result.is_valid
        assert any("Trace not preserved" in error for error in validation_result.errors)
    
    def test_monitor_numerical_stability_non_hermitian(self):
        """Test numerical stability monitoring for non-Hermitian matrices."""
        # Create a valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2], [0.2, 0.4]]),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Bypass validation by directly setting non-Hermitian matrix
        invalid_state.matrix = np.array([[0.6, 0.2], [0.3, 0.4]])
        
        validation_result = self.error_handler.monitor_numerical_stability(invalid_state)
        
        assert not validation_result.is_valid
        assert any("not Hermitian" in error for error in validation_result.errors)
    
    def test_monitor_numerical_stability_negative_eigenvalue(self):
        """Test numerical stability monitoring for negative eigenvalues."""
        # Create a valid state first
        invalid_state = DensityMatrix(
            matrix=np.array([[0.6, 0.2], [0.2, 0.4]]),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        # Bypass validation by directly setting matrix with negative eigenvalue
        invalid_state.matrix = np.array([[0.5, 0.8], [0.8, 0.5]])  # Has negative eigenvalue
        
        validation_result = self.error_handler.monitor_numerical_stability(invalid_state)
        
        assert not validation_result.is_valid
        assert any("Negative eigenvalue" in error for error in validation_result.errors)
    
    def test_validate_energy_conservation_good(self):
        """Test energy conservation validation for well-conserved energy."""
        energy_trajectory = [-100.0, -100.1, -99.9, -100.05, -100.0]
        
        validation_result = self.error_handler._validate_energy_conservation(energy_trajectory)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    def test_validate_energy_conservation_poor(self):
        """Test energy conservation validation for poorly conserved energy."""
        energy_trajectory = [-100.0, -90.0, -80.0, -70.0, -60.0]  # 40% drift
        
        validation_result = self.error_handler._validate_energy_conservation(energy_trajectory)
        
        assert not validation_result.is_valid
        assert any("Energy not conserved" in error for error in validation_result.errors)
    
    def test_validate_energy_conservation_insufficient_data(self):
        """Test energy conservation validation with insufficient data."""
        energy_trajectory = [-100.0]  # Only one point
        
        validation_result = self.error_handler._validate_energy_conservation(energy_trajectory)
        
        assert validation_result.is_valid  # Should pass but with warning
        assert any("Insufficient data" in warning for warning in validation_result.warnings)
    
    def test_validate_probability_conservation(self):
        """Test probability conservation validation."""
        # Create states with proper traces
        state1 = DensityMatrix(
            matrix=np.array([[0.6, 0.2], [0.2, 0.4]]),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        state2 = DensityMatrix(
            matrix=np.array([[0.7, 0.1], [0.1, 0.3]]),
            basis_labels=["ground", "excited"],
            time=0.1
        )
        
        validation_result = self.error_handler._validate_probability_conservation([state1, state2])
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    def test_validate_quantum_states(self):
        """Test quantum state validation."""
        states = [self.valid_density_matrix]
        
        validation_result = self.error_handler._validate_quantum_states(states)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    def test_validate_coherence_measures_valid(self):
        """Test coherence measures validation for valid values."""
        coherence_measures = {
            "purity": [0.8, 0.7, 0.9],
            "von_neumann_entropy": [0.2, 0.3, 0.1],
            "coherence_lifetime": [1.0, 1.5, 0.8]
        }
        
        validation_result = self.error_handler._validate_coherence_measures(coherence_measures)
        
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
    
    def test_validate_coherence_measures_invalid(self):
        """Test coherence measures validation for invalid values."""
        coherence_measures = {
            "purity": [1.5, 0.7, -0.1],  # Invalid: >1 and <0
            "von_neumann_entropy": [-0.1, 0.3, 0.1],  # Invalid: <0
            "coherence_lifetime": [-1.0, 1.5, 0.8]  # Invalid: <0
        }
        
        validation_result = self.error_handler._validate_coherence_measures(coherence_measures)
        
        assert not validation_result.is_valid
        assert len(validation_result.errors) >= 4  # Multiple invalid values


class TestErrorHandlerIntegration:
    """Integration tests for error handling in simulation context."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.error_handler = ErrorHandler()
    
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_percent')
    def test_resource_monitoring_integration(self, mock_cpu, mock_memory):
        """Test integrated resource monitoring during simulation."""
        # Mock system resources
        mock_memory.return_value = Mock(percent=75.0, available=2*1024**3)
        mock_cpu.return_value = 60.0
        
        # Simulate resource exhaustion scenario
        system_size = 5000
        recommendations = self.error_handler.handle_resource_exhaustion(system_size)
        
        # Verify appropriate recommendations
        assert recommendations['action'] in ['moderate_reduction', 'size_optimization', 'use_approximations']
        assert recommendations['new_system_size'] <= system_size
        assert 'memory_usage' in recommendations
        assert 'cpu_usage' in recommendations
    
    def test_error_logging_integration(self):
        """Test error logging across multiple error types."""
        # Generate multiple errors
        self.error_handler.handle_convergence_failure({'time_step': 1.0})
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            mock_memory.return_value = Mock(percent=90.0, available=1024**3)
            mock_cpu.return_value = 80.0
            self.error_handler.handle_resource_exhaustion(1000)
        
        # Verify error log contains both errors
        assert len(self.error_handler.error_log) == 2
        error_types = [entry['error_type'] for entry in self.error_handler.error_log]
        assert 'convergence_failure' in error_types
        assert 'resource_exhaustion' in error_types
    
    def test_comprehensive_validation_workflow(self):
        """Test complete validation workflow for simulation results."""
        # Create comprehensive test results
        state_trajectory = []
        for i in range(5):
            state = DensityMatrix(
                matrix=np.array([[0.6 + 0.1*i/10, 0.2], [0.2, 0.4 - 0.1*i/10]]),
                basis_labels=["ground", "excited"],
                time=i*0.1
            )
            state_trajectory.append(state)
        
        coherence_measures = {
            "purity": [0.8, 0.75, 0.7, 0.65, 0.6],
            "von_neumann_entropy": [0.2, 0.25, 0.3, 0.35, 0.4],
            "coherence_lifetime": [1.0, 0.9, 0.8, 0.7, 0.6]
        }
        
        energy_trajectory = [-100.0, -100.1, -99.9, -100.05, -100.02]
        
        results = SimulationResults(
            state_trajectory=state_trajectory,
            coherence_measures=coherence_measures,
            energy_trajectory=energy_trajectory,
            decoherence_rates={"dephasing": 0.1},
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=5
            ),
            simulation_config=SimulationConfig(
                system_pdb="test.pdb", temperature=300.0, simulation_time=1.0,
                time_step=0.1, quantum_subsystem_selection="all",
                noise_model_type="protein", output_directory="/tmp"
            ),
            computation_time=10.0
        )
        
        # Run comprehensive validation
        validation_result = self.error_handler.handle_unphysical_results(results)
        
        # Should pass validation for this well-formed data
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0


if __name__ == "__main__":
    pytest.main([__file__])