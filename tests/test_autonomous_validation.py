"""
Integration tests for QBES autonomous validation system.

This module tests the QBESValidator class and its autonomous validation
capabilities including retry logic, accuracy assessment, and certification.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from qbes.validation.validator import (
    QBESValidator, ValidationConfig, ValidationSummary
)
from qbes.validation.accuracy_calculator import AccuracyResult
from qbes.benchmarks.benchmark_runner import ValidationResult, ValidationResults


class TestValidationConfig:
    """Test ValidationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        assert config.suite_type == "standard"
        assert config.accuracy_threshold == 98.0
        assert config.pass_rate_threshold == 100.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.output_dir is None
        assert config.tolerance == 0.02
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ValidationConfig(
            suite_type="full",
            accuracy_threshold=95.0,
            pass_rate_threshold=90.0,
            max_retries=5,
            retry_delay=2.0,
            output_dir="/tmp/validation",
            tolerance=0.01
        )
        
        assert config.suite_type == "full"
        assert config.accuracy_threshold == 95.0
        assert config.pass_rate_threshold == 90.0
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.output_dir == "/tmp/validation"
        assert config.tolerance == 0.01


class TestValidationSummary:
    """Test ValidationSummary dataclass."""
    
    def test_validation_summary_creation(self):
        """Test ValidationSummary creation with all fields."""
        summary = ValidationSummary(
            total_attempts=3,
            final_accuracy=98.5,
            final_pass_rate=100.0,
            certification_achieved=True,
            failed_tests=[],
            retry_history=[
                {'attempt': 1, 'accuracy': 95.0, 'passed': False},
                {'attempt': 2, 'accuracy': 97.0, 'passed': False},
                {'attempt': 3, 'accuracy': 98.5, 'passed': True}
            ],
            execution_time=45.2
        )
        
        assert summary.total_attempts == 3
        assert summary.final_accuracy == 98.5
        assert summary.final_pass_rate == 100.0
        assert summary.certification_achieved is True
        assert summary.failed_tests == []
        assert len(summary.retry_history) == 3
        assert summary.execution_time == 45.2


class TestQBESValidator:
    """Test QBESValidator class functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def mock_reference_data(self, temp_dir):
        """Create mock reference data file."""
        reference_data = {
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
        
        ref_file = Path(temp_dir) / "reference_data.json"
        with open(ref_file, 'w') as f:
            json.dump(reference_data, f)
        
        return str(ref_file)
    
    @pytest.fixture
    def validator_config(self, temp_dir):
        """Create test validation configuration."""
        return ValidationConfig(
            suite_type="standard",
            accuracy_threshold=98.0,
            pass_rate_threshold=100.0,
            max_retries=2,
            retry_delay=0.1,  # Short delay for testing
            output_dir=temp_dir,
            tolerance=0.02
        )
    
    def test_validator_initialization(self, validator_config):
        """Test QBESValidator initialization."""
        validator = QBESValidator(validator_config)
        
        assert validator.config == validator_config
        assert validator.accuracy_calculator is not None
        assert validator.benchmark_runner is not None
        assert validator.validation_history == []
        assert validator.retry_count == 0
    
    def test_validator_default_initialization(self):
        """Test QBESValidator initialization with default config."""
        validator = QBESValidator()
        
        assert validator.config.suite_type == "standard"
        assert validator.config.accuracy_threshold == 98.0
        assert validator.config.max_retries == 3
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    def test_successful_validation_first_attempt(self, mock_benchmark_runner, validator_config):
        """Test successful validation on first attempt."""
        # Mock successful validation results
        mock_results = ValidationResults()
        # Create results that will pass: relative error < tolerance
        mock_result1 = ValidationResult("test1", 1.001, 1.0, 0.02, 0.5)  # 0.1% error < 2% tolerance
        mock_result2 = ValidationResult("test2", 0.501, 0.5, 0.02, 0.3)  # 0.2% error < 2% tolerance
        mock_results.add_result(mock_result1)
        mock_results.add_result(mock_result2)
        
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.return_value = mock_results
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(validator_config)
        summary = validator.validate_against_benchmarks("standard")
        
        assert summary.certification_achieved is True
        assert summary.total_attempts == 1
        assert summary.final_accuracy >= 98.0
        assert summary.final_pass_rate == 100.0
        assert len(summary.failed_tests) == 0
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_validation_with_retries(self, mock_sleep, mock_benchmark_runner, validator_config):
        """Test validation with retry logic."""
        # Mock failing then successful validation results
        mock_failing_results = ValidationResults()
        mock_failing_result = ValidationResult("test1", 0.95, 1.0, 0.001, 0.5)  # High error
        mock_failing_results.add_result(mock_failing_result)
        
        mock_successful_results = ValidationResults()
        mock_successful_result = ValidationResult("test1", 1.0001, 1.0, 0.001, 0.5)
        mock_successful_results.add_result(mock_successful_result)
        
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.side_effect = [
            mock_failing_results,  # First attempt fails
            mock_successful_results  # Second attempt succeeds
        ]
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(validator_config)
        summary = validator.validate_against_benchmarks("standard")
        
        assert summary.total_attempts == 2
        assert summary.certification_achieved is True
        assert len(summary.retry_history) == 2
        assert mock_sleep.called  # Verify retry delay was used
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    @patch('time.sleep')
    def test_validation_max_retries_exceeded(self, mock_sleep, mock_benchmark_runner, validator_config):
        """Test validation when max retries are exceeded."""
        # Mock consistently failing validation results
        mock_failing_results = ValidationResults()
        mock_failing_result = ValidationResult("test1", 0.8, 1.0, 0.001, 0.5)  # High error
        mock_failing_results.add_result(mock_failing_result)
        
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.return_value = mock_failing_results
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(validator_config)
        summary = validator.validate_against_benchmarks("standard")
        
        assert summary.certification_achieved is False
        assert summary.total_attempts == validator_config.max_retries + 1
        assert summary.final_accuracy < 98.0
        assert len(summary.failed_tests) > 0
    
    def test_analyze_validation_results(self, validator_config):
        """Test validation results analysis."""
        validator = QBESValidator(validator_config)
        
        # Create test validation results
        results = ValidationResults()
        good_result = ValidationResult("test1", 1.0001, 1.0, 0.001, 0.5)
        bad_result = ValidationResult("test2", 0.8, 1.0, 0.001, 0.3)  # High error
        results.add_result(good_result)
        results.add_result(bad_result)
        
        analysis = validator._analyze_validation_results(results)
        
        assert 'accuracy' in analysis
        assert 'pass_rate' in analysis
        assert 'certification_achieved' in analysis
        assert 'failed_tests' in analysis
        assert analysis['certification_achieved'] is False  # Due to bad result
        assert 'test2' in analysis['failed_tests']
        assert 'test1' not in analysis['failed_tests']
    
    def test_calculate_accuracy_score(self, validator_config):
        """Test accuracy score calculation."""
        validator = QBESValidator(validator_config)
        
        # Create test validation results
        results = ValidationResults()
        result1 = ValidationResult("test1", 1.001, 1.0, 0.001, 0.5)  # 0.1% error
        result2 = ValidationResult("test2", 0.502, 0.5, 0.001, 0.3)  # 0.4% error
        results.add_result(result1)
        results.add_result(result2)
        
        accuracy = validator.calculate_accuracy_score(results)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 100.0
        assert accuracy > 99.0  # Should be high accuracy for small errors
    
    def test_assess_pass_fail_status(self, validator_config):
        """Test pass/fail status assessment."""
        validator = QBESValidator(validator_config)
        
        # Create test validation results
        results = ValidationResults()
        passing_result = ValidationResult("test1", 1.001, 1.0, 0.01, 0.5)  # Within tolerance
        failing_result = ValidationResult("test2", 0.8, 1.0, 0.01, 0.3)  # Outside tolerance
        results.add_result(passing_result)
        results.add_result(failing_result)
        
        status = validator.assess_pass_fail_status(results)
        
        assert 'test1' in status
        assert 'test2' in status
        assert 'overall' in status
        assert status['test1'] is True
        assert status['test2'] is False
        assert status['overall'] is False  # Overall fails if any test fails
    
    def test_execute_validation_loop(self, validator_config):
        """Test validation loop execution."""
        with patch.object(QBESValidator, 'validate_against_benchmarks') as mock_validate:
            mock_summary = ValidationSummary(
                total_attempts=1,
                final_accuracy=98.5,
                final_pass_rate=100.0,
                certification_achieved=True,
                failed_tests=[],
                retry_history=[],
                execution_time=10.0
            )
            mock_validate.return_value = mock_summary
            
            validator = QBESValidator(validator_config)
            summary = validator.execute_validation_loop("standard", max_iterations=5)
            
            assert summary == mock_summary
            mock_validate.assert_called_once_with("standard")
    
    def test_generate_certification_report(self, validator_config):
        """Test certification report generation."""
        validator = QBESValidator(validator_config)
        
        # Create test summary
        summary = ValidationSummary(
            total_attempts=2,
            final_accuracy=98.5,
            final_pass_rate=100.0,
            certification_achieved=True,
            failed_tests=[],
            retry_history=[
                {'attempt': 1, 'accuracy': 95.0, 'pass_rate': 100.0, 'passed': False},
                {'attempt': 2, 'accuracy': 98.5, 'pass_rate': 100.0, 'passed': True}
            ],
            execution_time=30.5
        )
        
        report = validator.generate_certification_report(summary)
        
        assert isinstance(report, str)
        assert "CERTIFIED" in report
        assert "98.5%" in report
        assert "100.0%" in report
        assert "2" in report  # Total attempts
        assert "30.5" in report  # Execution time
    
    def test_generate_certification_report_failed(self, validator_config):
        """Test certification report for failed validation."""
        validator = QBESValidator(validator_config)
        
        # Create test summary for failed validation
        summary = ValidationSummary(
            total_attempts=3,
            final_accuracy=95.0,
            final_pass_rate=80.0,
            certification_achieved=False,
            failed_tests=["test1", "test2"],
            retry_history=[
                {'attempt': 1, 'accuracy': 90.0, 'pass_rate': 60.0, 'passed': False},
                {'attempt': 2, 'accuracy': 92.0, 'pass_rate': 70.0, 'passed': False},
                {'attempt': 3, 'accuracy': 95.0, 'pass_rate': 80.0, 'passed': False}
            ],
            execution_time=60.0
        )
        
        report = validator.generate_certification_report(summary)
        
        assert isinstance(report, str)
        assert "NOT CERTIFIED" in report
        assert "95.0%" in report
        assert "80.0%" in report
        assert "test1" in report
        assert "test2" in report
        assert "❌ NOT MET" in report
    
    def test_validation_history_tracking(self, validator_config):
        """Test validation history tracking."""
        validator = QBESValidator(validator_config)
        
        # Initially empty
        assert len(validator.get_validation_history()) == 0
        
        # Mock some validation results
        results1 = ValidationResults()
        results2 = ValidationResults()
        
        validator.validation_history.append(results1)
        validator.validation_history.append(results2)
        
        history = validator.get_validation_history()
        assert len(history) == 2
        assert history[0] == results1
        assert history[1] == results2
    
    def test_reset_validation_state(self, validator_config):
        """Test validation state reset."""
        validator = QBESValidator(validator_config)
        
        # Add some state
        validator.validation_history.append(ValidationResults())
        validator.retry_count = 5
        
        # Reset
        validator.reset_validation_state()
        
        assert len(validator.validation_history) == 0
        assert validator.retry_count == 0
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    def test_validation_with_exception_handling(self, mock_benchmark_runner, validator_config):
        """Test validation with exception handling during execution."""
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.side_effect = Exception("Simulation failed")
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(validator_config)
        
        # Should raise exception after max retries
        with pytest.raises(Exception, match="Simulation failed"):
            validator.validate_against_benchmarks("standard")
    
    def test_prepare_for_retry_logging(self, validator_config, caplog):
        """Test retry preparation logging."""
        validator = QBESValidator(validator_config)
        
        # Create analysis with failed tests
        analysis = {
            'failed_tests': ['test1', 'test2'],
            'accuracy': 95.0,
            'pass_rate': 80.0,
            'meets_accuracy_threshold': False,
            'meets_pass_rate_threshold': False,
            'accuracy_results': [
                AccuracyResult(
                    test_name='test1',
                    computed_value=0.8,
                    reference_value=1.0,
                    relative_error=0.2,
                    absolute_error=0.2,
                    passed=False,
                    tolerance=0.02
                )
            ]
        }
        
        validator._prepare_for_retry(analysis)
        
        assert validator.retry_count == 1
        assert "Failed tests identified" in caplog.text
        assert "test1" in caplog.text
        assert "Accuracy threshold not met" in caplog.text
    
    def test_empty_validation_results_handling(self, validator_config):
        """Test handling of empty validation results."""
        validator = QBESValidator(validator_config)
        
        # Create empty validation results
        empty_results = ValidationResults()
        
        # Test accuracy calculation
        accuracy = validator.calculate_accuracy_score(empty_results)
        assert accuracy == 0.0
        
        # Test pass/fail assessment
        status = validator.assess_pass_fail_status(empty_results)
        assert status == {'overall': False}
        
        # Test analysis
        analysis = validator._analyze_validation_results(empty_results)
        assert analysis['accuracy'] == 0.0
        assert analysis['pass_rate'] == 0.0
        assert analysis['certification_achieved'] is False


class TestIntegrationScenarios:
    """Integration test scenarios for autonomous validation."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration tests."""
        return ValidationConfig(
            suite_type="quick",
            accuracy_threshold=95.0,
            pass_rate_threshold=100.0,
            max_retries=1,
            retry_delay=0.1,
            tolerance=0.05
        )
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    def test_full_autonomous_validation_success(self, mock_benchmark_runner, integration_config):
        """Test complete autonomous validation success scenario."""
        # Mock successful validation
        mock_results = ValidationResults()
        mock_result = ValidationResult("test1", 1.02, 1.0, 0.05, 1.0)  # 2% error, within 5% tolerance
        mock_results.add_result(mock_result)
        
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.return_value = mock_results
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(integration_config)
        summary = validator.execute_validation_loop()
        
        # Verify successful certification
        assert summary.certification_achieved is True
        assert summary.total_attempts == 1
        assert summary.final_accuracy >= 95.0
        assert summary.final_pass_rate == 100.0
        
        # Generate and verify report
        report = validator.generate_certification_report(summary)
        assert "CERTIFIED" in report
        assert "✅ MET" in report
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    @patch('time.sleep')
    def test_full_autonomous_validation_with_recovery(self, mock_sleep, mock_benchmark_runner, integration_config):
        """Test autonomous validation with failure and recovery."""
        # Mock failing then successful validation
        mock_failing_results = ValidationResults()
        mock_failing_result = ValidationResult("test1", 0.8, 1.0, 0.05, 1.0)  # 20% error
        mock_failing_results.add_result(mock_failing_result)
        
        mock_successful_results = ValidationResults()
        mock_successful_result = ValidationResult("test1", 1.02, 1.0, 0.05, 1.0)  # 2% error
        mock_successful_results.add_result(mock_successful_result)
        
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.side_effect = [
            mock_failing_results,
            mock_successful_results
        ]
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(integration_config)
        summary = validator.execute_validation_loop()
        
        # Verify recovery and certification
        assert summary.certification_achieved is True
        assert summary.total_attempts == 2
        assert len(summary.retry_history) == 2
        assert summary.retry_history[0]['passed'] is False
        assert summary.retry_history[1]['passed'] is True
        
        # Verify validation history
        history = validator.get_validation_history()
        assert len(history) == 2
    
    @patch('qbes.validation.validator.BenchmarkRunner')
    def test_validation_state_management(self, mock_benchmark_runner, integration_config):
        """Test validation state management across multiple runs."""
        mock_results = ValidationResults()
        mock_result = ValidationResult("test1", 1.01, 1.0, 0.05, 1.0)
        mock_results.add_result(mock_result)
        
        mock_runner_instance = Mock()
        mock_runner_instance.run_validation_suite.return_value = mock_results
        mock_benchmark_runner.return_value = mock_runner_instance
        
        validator = QBESValidator(integration_config)
        
        # First validation run
        summary1 = validator.execute_validation_loop()
        assert len(validator.get_validation_history()) == 1
        
        # Reset and run again
        validator.reset_validation_state()
        assert len(validator.get_validation_history()) == 0
        
        summary2 = validator.execute_validation_loop()
        assert len(validator.get_validation_history()) == 1
        
        # Both should be successful
        assert summary1.certification_achieved is True
        assert summary2.certification_achieved is True


if __name__ == "__main__":
    pytest.main([__file__])