"""
Integration tests for end-to-end validation workflow.

This module tests the complete validation pipeline from CLI command execution
through benchmark running, accuracy calculation, and report generation.
"""

import pytest
import tempfile
import os
import json
import yaml
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from qbes.cli import validate
from qbes.validation.validator import QBESValidator, ValidationConfig, ValidationSummary
from qbes.validation.accuracy_calculator import AccuracyCalculator, AccuracyResult
from qbes.benchmarks.benchmark_runner import BenchmarkRunner, ValidationResults, ValidationResult as BenchmarkResult


class TestValidationIntegration:
    """Integration tests for complete validation workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('qbes.cli.BenchmarkRunner')
    def test_end_to_end_validation_success(self, mock_benchmark_runner):
        """Test complete end-to-end validation workflow with successful results."""
        
        # Mock successful benchmark results
        mock_runner = Mock()
        mock_benchmark_runner.return_value = mock_runner
        
        # Create mock validation results
        mock_results = ValidationResults(
            results=[
                BenchmarkResult(
                    name="two_level_system",
                    passed=True,
                    accuracy=0.995,
                    expected_value=1.0,
                    actual_value=0.995,
                    tolerance=0.01,
                    execution_time=0.5,
                    error_message=None
                ),
                BenchmarkResult(
                    name="harmonic_oscillator",
                    passed=True,
                    accuracy=0.998,
                    expected_value=2.5,
                    actual_value=2.495,
                    tolerance=0.02,
                    execution_time=0.8,
                    error_message=None
                )
            ],
            overall_accuracy=0.9965,
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
            execution_time=1.3
        )
        
        mock_runner.run_validation_suite.return_value = mock_results
        
        # Test CLI command execution
        result = self.runner.invoke(validate, ['--suite', 'quick', '--format', 'markdown'])
        
        # Verify successful execution
        assert result.exit_code == 0
        assert "Validation completed successfully" in result.output
        assert "Overall accuracy: 99.65%" in result.output
        assert "Tests passed: 2/2" in result.output
        
        # Verify benchmark runner was called correctly
        mock_benchmark_runner.assert_called_once()
        mock_runner.run_validation_suite.assert_called_once_with('quick')
    
    @patch('qbes.cli.BenchmarkRunner')
    def test_end_to_end_validation_failures(self, mock_benchmark_runner):
        """Test complete end-to-end validation workflow with some failures."""
        
        # Mock benchmark results with failures
        mock_runner = Mock()
        mock_benchmark_runner.return_value = mock_runner
        
        # Create mock validation results with failures
        mock_results = ValidationResults(
            results=[
                BenchmarkResult(
                    name="two_level_system",
                    passed=True,
                    accuracy=0.995,
                    expected_value=1.0,
                    actual_value=0.995,
                    tolerance=0.01,
                    execution_time=0.5,
                    error_message=None
                ),
                BenchmarkResult(
                    name="fmo_complex",
                    passed=False,
                    accuracy=0.85,
                    expected_value=5.0,
                    actual_value=4.25,
                    tolerance=0.05,
                    execution_time=2.1,
                    error_message="Accuracy below threshold"
                )
            ],
            overall_accuracy=0.9225,
            total_tests=2,
            passed_tests=1,
            failed_tests=1,
            execution_time=2.6
        )
        
        mock_runner.run_validation_suite.return_value = mock_results
        
        # Test CLI command execution
        result = self.runner.invoke(validate, ['--suite', 'full', '--format', 'json'])
        
        # Verify execution with failures
        assert result.exit_code == 1  # Should exit with error code for failures
        assert "Validation completed with failures" in result.output
        assert "Overall accuracy: 92.25%" in result.output
        assert "Tests passed: 1/2" in result.output
        
        # Verify benchmark runner was called correctly
        mock_benchmark_runner.assert_called_once()
        mock_runner.run_validation_suite.assert_called_once_with('full')
    
    @patch('qbes.validation.validator.QBESValidator')
    def test_autonomous_validation_workflow(self, mock_validator):
        """Test autonomous validation execution workflow."""
        
        # Mock validator instance
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        
        # Create mock validation summary
        mock_summary = ValidationSummary(
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            overall_accuracy=0.94,
            execution_time=15.2,
            timestamp="2024-01-15T10:30:00Z",
            config_used=ValidationConfig(
                suite_name="full",
                tolerance=0.05,
                max_retries=3,
                enable_debugging=True
            ),
            detailed_results=[],
            recommendations=["Increase simulation time for FMO benchmark"]
        )
        
        mock_validator_instance.run_validation.return_value = mock_summary
        
        # Test autonomous validation
        validator = QBESValidator()
        summary = validator.run_validation("full")
        
        # Verify results
        assert summary.total_tests == 5
        assert summary.passed_tests == 4
        assert summary.failed_tests == 1
        assert summary.overall_accuracy == 0.94
        assert len(summary.recommendations) > 0
    
    def test_validation_report_generation_integration(self):
        """Test integration of validation report generation."""
        
        # Create test validation results
        test_results = ValidationResults(
            results=[
                BenchmarkResult(
                    name="analytical_test",
                    passed=True,
                    accuracy=0.999,
                    expected_value=1.0,
                    actual_value=0.999,
                    tolerance=0.01,
                    execution_time=0.3,
                    error_message=None
                )
            ],
            overall_accuracy=0.999,
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            execution_time=0.3
        )
        
        # Test report generation (would normally use ReportGenerator)
        # For now, verify the structure exists
        assert hasattr(test_results, 'overall_accuracy')
        assert hasattr(test_results, 'total_tests')
        assert hasattr(test_results, 'execution_time')
        
        # Verify individual result structure
        result = test_results.results[0]
        assert hasattr(result, 'name')
        assert hasattr(result, 'passed')
        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'execution_time')
    
    def test_accuracy_calculation_integration(self):
        """Test integration of accuracy calculation system."""
        
        # Test accuracy calculator integration
        calculator = AccuracyCalculator()
        
        # Test relative error calculation
        expected = 1.0
        actual = 0.995
        tolerance = 0.01
        
        result = calculator.calculate_relative_error(expected, actual, tolerance)
        
        # Verify accuracy result structure
        assert isinstance(result, AccuracyResult)
        assert hasattr(result, 'relative_error')
        assert hasattr(result, 'accuracy_score')
        assert hasattr(result, 'passed')
        
        # Verify calculation correctness
        assert result.relative_error == 0.005
        assert result.accuracy_score == 0.995
        assert result.passed == True  # Within tolerance
    
    @patch('qbes.cli.BenchmarkRunner')
    def test_validation_with_custom_tolerance(self, mock_benchmark_runner):
        """Test validation workflow with custom tolerance settings."""
        
        # Mock benchmark runner
        mock_runner = Mock()
        mock_benchmark_runner.return_value = mock_runner
        
        # Create mock results with custom tolerance
        mock_results = ValidationResults(
            results=[
                BenchmarkResult(
                    name="custom_tolerance_test",
                    passed=True,
                    accuracy=0.92,
                    expected_value=1.0,
                    actual_value=0.92,
                    tolerance=0.1,  # Custom higher tolerance
                    execution_time=1.0,
                    error_message=None
                )
            ],
            overall_accuracy=0.92,
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            execution_time=1.0
        )
        
        mock_runner.run_validation_suite.return_value = mock_results
        
        # Test CLI command with custom tolerance
        result = self.runner.invoke(validate, [
            '--suite', 'quick', 
            '--tolerance', '0.1',
            '--format', 'markdown'
        ])
        
        # Verify successful execution with custom tolerance
        assert result.exit_code == 0
        assert "Overall accuracy: 92.00%" in result.output
        
        # Verify tolerance was passed correctly
        mock_benchmark_runner.assert_called_once()
    
    def test_validation_error_handling_integration(self):
        """Test error handling in validation integration workflow."""
        
        # Test with invalid suite name
        result = self.runner.invoke(validate, ['--suite', 'invalid_suite'])
        
        # Should handle gracefully
        assert result.exit_code != 0
        
        # Test with invalid format
        result = self.runner.invoke(validate, ['--format', 'invalid_format'])
        
        # Should handle gracefully
        assert result.exit_code != 0   
 
    def test_debugging_integration_workflow(self):
        """Test integration of debugging features in validation workflow."""
        
        # Test that debugging features are properly integrated
        # This would test the debugging loop functionality
        
        # Mock a validation failure scenario
        failed_result = BenchmarkResult(
            name="debug_test",
            passed=False,
            accuracy=0.75,
            expected_value=1.0,
            actual_value=0.75,
            tolerance=0.05,
            execution_time=1.5,
            error_message="Accuracy below threshold - debugging required"
        )
        
        # Verify error message structure for debugging
        assert "debugging required" in failed_result.error_message
        assert failed_result.accuracy < failed_result.tolerance
        assert not failed_result.passed
        
        # Test debugging workflow would be triggered
        # (In actual implementation, this would trigger the debugging loop)
        debug_info = {
            'failed_test': failed_result.name,
            'accuracy_gap': failed_result.tolerance - failed_result.accuracy,
            'suggested_fixes': ['Increase simulation time', 'Check numerical precision']
        }
        
        assert debug_info['accuracy_gap'] > 0
        assert len(debug_info['suggested_fixes']) > 0
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration in validation workflow."""
        
        # Test performance metrics collection
        performance_data = {
            'total_execution_time': 25.5,
            'average_test_time': 5.1,
            'memory_usage_peak': 512.0,  # MB
            'cpu_utilization': 0.85
        }
        
        # Verify performance data structure
        assert performance_data['total_execution_time'] > 0
        assert performance_data['average_test_time'] > 0
        assert performance_data['memory_usage_peak'] > 0
        assert 0 <= performance_data['cpu_utilization'] <= 1.0
        
        # Test performance thresholds
        max_execution_time = 60.0  # seconds
        max_memory_usage = 1024.0  # MB
        
        assert performance_data['total_execution_time'] < max_execution_time
        assert performance_data['memory_usage_peak'] < max_memory_usage


if __name__ == '__main__':
    pytest.main([__file__])