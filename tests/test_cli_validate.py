"""
Tests for QBES CLI validate command functionality.

This module tests the CLI validate command integration, including
argument parsing, benchmark execution orchestration, and report generation.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock

from qbes.cli import main, validate
from qbes.benchmarks.benchmark_runner import ValidationResult, ValidationResults


class TestCLIValidateCommand:
    """Test suite for CLI validate command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_validate_command_exists(self):
        """Test that validate command is properly registered."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'validate' in result.output
    
    def test_validate_help_message(self):
        """Test validate command help message content."""
        result = self.runner.invoke(main, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'Run QBES validation benchmarks' in result.output
        assert '--suite' in result.output
        assert '--output-dir' in result.output
        assert '--tolerance' in result.output
        assert '--verbose' in result.output
        assert '--report-format' in result.output
    
    def test_validate_suite_options(self):
        """Test that all suite options are available."""
        result = self.runner.invoke(main, ['validate', '--help'])
        assert 'quick' in result.output
        assert 'standard' in result.output
        assert 'full' in result.output
    
    def test_validate_report_format_options(self):
        """Test that report format options are available."""
        result = self.runner.invoke(main, ['validate', '--help'])
        assert 'markdown' in result.output
        assert 'json' in result.output
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_default_parameters(self, mock_runner_class):
        """Test validate command with default parameters."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_benchmark",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "test_report.md"
        
        # Run validate command
        result = self.runner.invoke(main, ['validate'])
        
        # Verify command executed successfully
        assert result.exit_code == 0
        
        # Verify benchmark runner was called with correct parameters
        mock_runner_class.assert_called_once_with(output_dir='./validation_results')
        mock_runner.run_validation_suite.assert_called_once_with(suite_type='standard')
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_custom_parameters(self, mock_runner_class):
        """Test validate command with custom parameters."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_benchmark",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.01,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "custom_report.md"
        
        # Run validate command with custom parameters
        result = self.runner.invoke(main, [
            'validate',
            '--suite', 'full',
            '--output-dir', self.temp_dir,
            '--tolerance', '0.01',
            '--verbose',
            '--report-format', 'json'
        ])
        
        # Verify command executed successfully
        assert result.exit_code == 0
        
        # Verify benchmark runner was called with correct parameters
        mock_runner_class.assert_called_once_with(output_dir=self.temp_dir)
        mock_runner.run_validation_suite.assert_called_once_with(suite_type='full')
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_verbose_output(self, mock_runner_class):
        """Test validate command verbose output."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results with multiple tests
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_pass",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_results.add_result(ValidationResult(
            test_name="test_fail",
            computed_value=1.1,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.2
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "verbose_report.md"
        
        # Run validate command with verbose flag
        result = self.runner.invoke(main, ['validate', '--verbose'])
        
        # Verify verbose output elements are present
        assert 'Starting QBES validation suite' in result.output
        assert 'Individual Test Results' in result.output
        assert 'test_pass' in result.output
        assert 'test_fail' in result.output
        assert 'Computed:' in result.output
        assert 'Reference:' in result.output
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_all_tests_pass(self, mock_runner_class):
        """Test validate command when all tests pass."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results with all passing tests
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_1",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_results.add_result(ValidationResult(
            test_name="test_2",
            computed_value=0.5,
            reference_value=0.5,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "pass_report.md"
        
        # Run validate command
        result = self.runner.invoke(main, ['validate'])
        
        # Verify successful exit code
        assert result.exit_code == 0
        assert 'All validation tests passed successfully!' in result.output
        assert 'CERTIFIED' in result.output
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_some_tests_fail(self, mock_runner_class):
        """Test validate command when some tests fail."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results with some failing tests
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_pass",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_results.add_result(ValidationResult(
            test_name="test_fail",
            computed_value=1.1,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "fail_report.md"
        
        # Run validate command
        result = self.runner.invoke(main, ['validate'])
        
        # Verify failure exit code
        assert result.exit_code == 1
        assert 'test(s) need attention' in result.output
        assert 'Failed Tests' in result.output
        assert 'test_fail' in result.output
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_json_report_generation(self, mock_runner_class):
        """Test JSON report generation."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_benchmark",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "json_report.md"
        
        # Run validate command with JSON format
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = self.runner.invoke(main, [
                'validate',
                '--output-dir', self.temp_dir,
                '--report-format', 'json'
            ])
            
            # Verify JSON file was written
            assert result.exit_code == 0
            mock_open.assert_called()
            mock_file.write.assert_called()
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_output_directory_creation(self, mock_runner_class):
        """Test that output directory is created if it doesn't exist."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_benchmark",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "dir_test_report.md"
        
        # Use a non-existent directory
        test_output_dir = os.path.join(self.temp_dir, 'new_validation_dir')
        
        # Run validate command
        with patch('os.makedirs') as mock_makedirs:
            result = self.runner.invoke(main, [
                'validate',
                '--output-dir', test_output_dir
            ])
            
            # Verify directory creation was attempted
            mock_makedirs.assert_called_once_with(test_output_dir, exist_ok=True)
            assert result.exit_code == 0
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_benchmark_runner_exception(self, mock_runner_class):
        """Test validate command when benchmark runner raises exception."""
        # Mock the benchmark runner to raise an exception
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run_validation_suite.side_effect = Exception("Benchmark execution failed")
        
        # Run validate command
        result = self.runner.invoke(main, ['validate'])
        
        # Verify error exit code and message
        assert result.exit_code == 2
        assert 'Error running validation suite' in result.output
        assert 'Benchmark execution failed' in result.output
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_benchmark_runner_exception_verbose(self, mock_runner_class):
        """Test validate command exception handling with verbose output."""
        # Mock the benchmark runner to raise an exception
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run_validation_suite.side_effect = Exception("Detailed benchmark error")
        
        # Run validate command with verbose flag
        result = self.runner.invoke(main, ['validate', '--verbose'])
        
        # Verify error exit code and detailed error information
        assert result.exit_code == 2
        assert 'Error running validation suite' in result.output
        assert 'Detailed error information' in result.output
    
    def test_validate_suite_parameter_validation(self):
        """Test that invalid suite parameters are rejected."""
        result = self.runner.invoke(main, ['validate', '--suite', 'invalid_suite'])
        
        # Verify command fails with invalid suite option
        assert result.exit_code != 0
        assert 'Invalid value' in result.output or 'Usage:' in result.output
    
    def test_validate_tolerance_parameter_validation(self):
        """Test tolerance parameter validation."""
        # Test with valid tolerance
        with patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner
            
            mock_results = ValidationResults()
            mock_results.add_result(ValidationResult(
                test_name="test_benchmark",
                computed_value=1.0,
                reference_value=1.0,
                tolerance=0.05,
                computation_time=0.1
            ))
            mock_runner.run_validation_suite.return_value = mock_results
            mock_runner.save_validation_report.return_value = "tolerance_report.md"
            
            result = self.runner.invoke(main, ['validate', '--tolerance', '0.05'])
            assert result.exit_code == 0
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_performance_assessment(self, mock_runner_class):
        """Test performance assessment in output."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Create mock validation results
        mock_results = ValidationResults()
        mock_results.add_result(ValidationResult(
            test_name="test_benchmark",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "perf_report.md"
        
        # Mock time.time to control execution time measurement
        with patch('time.time', side_effect=[0.0, 30.0]):  # 30 second execution
            result = self.runner.invoke(main, ['validate'])
            
            # Verify performance assessment is included
            assert result.exit_code == 0
            assert 'Performance:' in result.output
            assert '30.0s' in result.output
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner')
    def test_validate_certification_status_display(self, mock_runner_class):
        """Test certification status display logic."""
        # Mock the benchmark runner
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        # Test certified status (100% pass rate, >98% accuracy)
        mock_results = ValidationResults()
        # Add a perfect result
        mock_results.add_result(ValidationResult(
            test_name="perfect_test",
            computed_value=1.0,
            reference_value=1.0,
            tolerance=0.02,
            computation_time=0.1
        ))
        mock_runner.run_validation_suite.return_value = mock_results
        mock_runner.save_validation_report.return_value = "cert_report.md"
        
        result = self.runner.invoke(main, ['validate'])
        
        # Verify certification status
        assert result.exit_code == 0
        assert 'CERTIFIED' in result.output
        assert 'meets all validation criteria' in result.output


class TestCLIValidateIntegration:
    """Integration tests for CLI validate command with real components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_validate_command_integration_with_real_runner(self):
        """Test validate command integration with real BenchmarkRunner."""
        # This test uses the actual BenchmarkRunner but with mocked simulation execution
        with patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner._execute_benchmark_simulation') as mock_execute:
            # Mock simulation results to return known values
            mock_execute.side_effect = lambda test_name: {
                'two_level_rabi_frequency': 1.0001,
                'harmonic_oscillator_ground_energy': 0.500001
            }.get(test_name, 1.0)
            
            result = self.runner.invoke(main, [
                'validate',
                '--suite', 'quick',
                '--output-dir', self.temp_dir,
                '--verbose'
            ])
            
            # Verify command executed successfully
            assert result.exit_code == 0
            assert 'QBES VALIDATION SUMMARY' in result.output
            
            # Verify output files were created
            report_files = list(Path(self.temp_dir).glob('validation_report*.md'))
            assert len(report_files) > 0
    
    def test_validate_command_with_real_reference_data(self):
        """Test validate command using real reference data file."""
        # Create a temporary reference data file
        reference_data = {
            "test_benchmark": {
                "value": 1.0,
                "unit": "normalized",
                "source": "test_source",
                "tolerance": 0.01
            }
        }
        
        ref_file = os.path.join(self.temp_dir, 'reference_data.json')
        with open(ref_file, 'w') as f:
            json.dump(reference_data, f)
        
        # Mock the benchmark runner to use our test reference data
        with patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner._load_reference_data') as mock_load:
            mock_load.return_value = reference_data
            
            with patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner._execute_benchmark_simulation') as mock_execute:
                mock_execute.return_value = 1.005  # Slightly off from reference
                
                result = self.runner.invoke(main, [
                    'validate',
                    '--suite', 'quick',
                    '--output-dir', self.temp_dir
                ])
                
                # Should pass since error is within tolerance
                assert result.exit_code == 0