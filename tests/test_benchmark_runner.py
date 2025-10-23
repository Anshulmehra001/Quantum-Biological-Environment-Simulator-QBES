"""
Unit tests for QBES Benchmark Runner infrastructure.

This module tests the core validation and benchmarking functionality
to ensure reliable operation of the QBES validation suite.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from qbes.benchmarks.benchmark_runner import (
    BenchmarkRunner, ValidationResult, ValidationResults
)


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test creating a ValidationResult object."""
        result = ValidationResult(
            test_name="test_case",
            computed_value=1.001,
            reference_value=1.0,
            tolerance=0.01,
            computation_time=0.5
        )
        
        self.assertEqual(result.test_name, "test_case")
        self.assertEqual(result.computed_value, 1.001)
        self.assertEqual(result.reference_value, 1.0)
        self.assertEqual(result.tolerance, 0.01)
        self.assertEqual(result.computation_time, 0.5)
        self.assertAlmostEqual(result.relative_error, 0.001, places=6)
        self.assertTrue(result.passed)
    
    def test_validation_result_failure(self):
        """Test ValidationResult with failing test."""
        result = ValidationResult(
            test_name="failing_test",
            computed_value=1.1,
            reference_value=1.0,
            tolerance=0.05
        )
        
        self.assertAlmostEqual(result.relative_error, 0.1, places=6)
        self.assertFalse(result.passed)
    
    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult(
            test_name="dict_test",
            computed_value=0.5,
            reference_value=0.5,
            tolerance=0.001
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['test_name'], "dict_test")
        self.assertEqual(result_dict['computed_value'], 0.5)
        self.assertEqual(result_dict['reference_value'], 0.5)
        self.assertTrue(result_dict['passed'])


class TestValidationResults(unittest.TestCase):
    """Test cases for ValidationResults class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.results = ValidationResults()
    
    def test_empty_validation_results(self):
        """Test empty ValidationResults object."""
        self.assertEqual(self.results.total_tests, 0)
        self.assertEqual(self.results.passed_tests, 0)
        self.assertEqual(self.results.pass_rate, 0.0)
        self.assertEqual(self.results.overall_accuracy, 0.0)
    
    def test_add_validation_results(self):
        """Test adding validation results."""
        result1 = ValidationResult("test1", 1.0, 1.0, 0.01)
        result2 = ValidationResult("test2", 1.005, 1.0, 0.01)
        
        self.results.add_result(result1)
        self.results.add_result(result2)
        
        self.assertEqual(self.results.total_tests, 2)
        self.assertEqual(self.results.passed_tests, 2)
        self.assertEqual(self.results.pass_rate, 100.0)
        self.assertGreater(self.results.overall_accuracy, 99.0)
    
    def test_mixed_validation_results(self):
        """Test ValidationResults with mixed pass/fail results."""
        result1 = ValidationResult("pass_test", 1.0, 1.0, 0.01)
        result2 = ValidationResult("fail_test", 1.1, 1.0, 0.05)
        
        self.results.add_result(result1)
        self.results.add_result(result2)
        
        self.assertEqual(self.results.total_tests, 2)
        self.assertEqual(self.results.passed_tests, 1)
        self.assertEqual(self.results.pass_rate, 50.0)
    
    def test_validation_results_to_dict(self):
        """Test converting ValidationResults to dictionary."""
        result = ValidationResult("test", 1.0, 1.0, 0.01)
        self.results.add_result(result)
        
        results_dict = self.results.to_dict()
        
        self.assertIsInstance(results_dict, dict)
        self.assertEqual(results_dict['total_tests'], 1)
        self.assertEqual(results_dict['passed_tests'], 1)
        self.assertEqual(results_dict['pass_rate'], 100.0)
        self.assertIn('individual_results', results_dict)


class TestBenchmarkRunner(unittest.TestCase):
    """Test cases for BenchmarkRunner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.reference_data_path = Path(self.temp_dir) / "test_reference_data.json"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create test reference data
        test_reference_data = {
            "two_level_rabi_frequency": {
                "value": 1.0,
                "unit": "normalized",
                "source": "analytical_solution",
                "tolerance": 0.001
            },
            "harmonic_oscillator_ground_energy": {
                "value": 0.5,
                "unit": "hbar_omega", 
                "source": "analytical_solution",
                "tolerance": 1e-10
            }
        }
        
        with open(self.reference_data_path, 'w') as f:
            json.dump(test_reference_data, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_benchmark_runner_initialization(self):
        """Test BenchmarkRunner initialization."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        self.assertEqual(runner.reference_data_path, self.reference_data_path)
        self.assertEqual(runner.output_dir, self.output_dir)
        self.assertIsInstance(runner.reference_data, dict)
        self.assertIn("two_level_rabi_frequency", runner.reference_data)
    
    def test_load_reference_data(self):
        """Test loading reference data from JSON file."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        self.assertIn("two_level_rabi_frequency", runner.reference_data)
        self.assertEqual(runner.reference_data["two_level_rabi_frequency"]["value"], 1.0)
    
    def test_load_missing_reference_data(self):
        """Test handling of missing reference data file."""
        missing_path = Path(self.temp_dir) / "missing.json"
        runner = BenchmarkRunner(
            reference_data_path=str(missing_path),
            output_dir=str(self.output_dir)
        )
        
        self.assertEqual(runner.reference_data, {})
    
    def test_run_single_benchmark(self):
        """Test running a single benchmark test."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        result = runner._run_single_benchmark("two_level_rabi_frequency")
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.test_name, "two_level_rabi_frequency")
        self.assertAlmostEqual(result.reference_value, 1.0)
        self.assertGreater(result.computation_time, 0)
    
    def test_run_validation_suite_quick(self):
        """Test running quick validation suite."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        results = runner.run_validation_suite("quick")
        
        self.assertIsInstance(results, ValidationResults)
        self.assertEqual(results.total_tests, 1)
        self.assertGreater(results.pass_rate, 0)
    
    def test_run_validation_suite_standard(self):
        """Test running standard validation suite."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        results = runner.run_validation_suite("standard")
        
        self.assertIsInstance(results, ValidationResults)
        self.assertEqual(results.total_tests, 2)
    
    def test_run_validation_suite_invalid(self):
        """Test running validation suite with invalid suite type."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        with self.assertRaises(ValueError):
            runner.run_validation_suite("invalid_suite")
    
    def test_compare_with_reference(self):
        """Test comparing results with reference data."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        results = ValidationResults()
        result = ValidationResult("test", 1.0, 1.0, 0.01)
        results.add_result(result)
        
        analysis = runner.compare_with_reference(results)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('summary', analysis)
        self.assertIn('detailed_results', analysis)
        self.assertIn('recommendations', analysis)
    
    def test_generate_validation_report(self):
        """Test generating validation report."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        results = ValidationResults()
        result = ValidationResult("test", 1.0, 1.0, 0.01)
        results.add_result(result)
        
        report = runner.generate_validation_report(results)
        
        self.assertIsInstance(report, str)
        self.assertIn("# QBES Validation Report", report)
        self.assertIn("Total Tests:", report)
        self.assertIn("Pass Rate:", report)
    
    def test_save_validation_report(self):
        """Test saving validation report to file."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        results = ValidationResults()
        result = ValidationResult("test", 1.0, 1.0, 0.01)
        results.add_result(result)
        
        report_path = runner.save_validation_report(results, "test_report.md")
        
        self.assertTrue(Path(report_path).exists())
        with open(report_path, 'r') as f:
            content = f.read()
        self.assertIn("# QBES Validation Report", content)
    
    @patch('qbes.benchmarks.benchmark_runner.BenchmarkRunner._execute_benchmark_simulation')
    def test_execute_benchmark_simulation_mock(self, mock_execute):
        """Test benchmark simulation execution with mocking."""
        mock_execute.return_value = 1.0001
        
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        result = runner._run_single_benchmark("two_level_rabi_frequency")
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.computed_value, 1.0001)
        mock_execute.assert_called_once_with("two_level_rabi_frequency")
    
    def test_execute_benchmark_simulation_placeholder(self):
        """Test placeholder benchmark simulation implementations."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        # Test two-level system
        value = runner._execute_benchmark_simulation("two_level_rabi_frequency")
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 1.0, delta=0.01)
        
        # Test harmonic oscillator
        value = runner._execute_benchmark_simulation("harmonic_oscillator_ground_energy")
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 0.5, delta=0.01)
    
    def test_execute_benchmark_simulation_unknown(self):
        """Test benchmark simulation with unknown test name."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        with self.assertRaises(ValueError):
            runner._execute_benchmark_simulation("unknown_test")


class TestBenchmarkRunnerIntegration(unittest.TestCase):
    """Integration tests for BenchmarkRunner."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.reference_data_path = Path(self.temp_dir) / "integration_reference_data.json"
        self.output_dir = Path(self.temp_dir) / "integration_output"
        
        # Create comprehensive reference data for integration testing
        integration_reference_data = {
            "two_level_rabi_frequency": {
                "value": 1.0,
                "unit": "normalized",
                "source": "analytical_solution",
                "tolerance": 0.01
            },
            "harmonic_oscillator_ground_energy": {
                "value": 0.5,
                "unit": "hbar_omega",
                "source": "analytical_solution", 
                "tolerance": 0.01
            },
            "fmo_coherence_lifetime_fs": {
                "value": 660.0,
                "unit": "femtoseconds",
                "source": "Engel et al. Nature 2007",
                "tolerance": 0.15
            }
        }
        
        with open(self.reference_data_path, 'w') as f:
            json.dump(integration_reference_data, f)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow from start to finish."""
        runner = BenchmarkRunner(
            reference_data_path=str(self.reference_data_path),
            output_dir=str(self.output_dir)
        )
        
        # Run full validation suite
        results = runner.run_validation_suite("full")
        
        # Verify results
        self.assertEqual(results.total_tests, 3)
        self.assertGreater(results.pass_rate, 0)
        
        # Generate and save report
        report_path = runner.save_validation_report(results)
        self.assertTrue(Path(report_path).exists())
        
        # Verify report content
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        self.assertIn("# QBES Validation Report", report_content)
        self.assertIn("two_level_rabi_frequency", report_content)
        self.assertIn("harmonic_oscillator_ground_energy", report_content)
        self.assertIn("fmo_coherence_lifetime_fs", report_content)


if __name__ == '__main__':
    unittest.main()