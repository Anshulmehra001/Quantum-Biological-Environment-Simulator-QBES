"""
Tests for AccuracyCalculator class

This module contains comprehensive tests for the accuracy calculation and
statistical analysis functionality in the QBES validation system.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import logging

from qbes.validation.accuracy_calculator import (
    AccuracyCalculator, 
    AccuracyResult, 
    StatisticalSummary
)


class TestAccuracyCalculator:
    """Test suite for AccuracyCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = AccuracyCalculator(default_tolerance=0.02)
    
    def test_init_default_tolerance(self):
        """Test AccuracyCalculator initialization with default tolerance."""
        calc = AccuracyCalculator()
        assert calc.default_tolerance == 0.02
    
    def test_init_custom_tolerance(self):
        """Test AccuracyCalculator initialization with custom tolerance."""
        calc = AccuracyCalculator(default_tolerance=0.05)
        assert calc.default_tolerance == 0.05
    
    def test_calculate_relative_error_normal_case(self):
        """Test relative error calculation for normal values."""
        computed = 1.02
        reference = 1.0
        expected_error = 0.02
        
        error = self.calculator.calculate_relative_error(computed, reference)
        assert abs(error - expected_error) < 1e-10
    
    def test_calculate_relative_error_zero_difference(self):
        """Test relative error calculation when values are identical."""
        computed = 1.0
        reference = 1.0
        
        error = self.calculator.calculate_relative_error(computed, reference)
        assert error == 0.0
    
    def test_calculate_relative_error_negative_values(self):
        """Test relative error calculation with negative values."""
        computed = -1.02
        reference = -1.0
        expected_error = 0.02
        
        error = self.calculator.calculate_relative_error(computed, reference)
        assert abs(error - expected_error) < 1e-10
    
    def test_calculate_relative_error_zero_reference_both_zero(self):
        """Test relative error when both values are zero."""
        computed = 0.0
        reference = 0.0
        
        error = self.calculator.calculate_relative_error(computed, reference)
        assert error == 0.0
    
    def test_calculate_relative_error_zero_reference_nonzero_computed(self):
        """Test relative error raises error when reference is zero but computed is not."""
        computed = 1.0
        reference = 0.0
        
        with pytest.raises(ValueError, match="Cannot calculate relative error with zero reference value"):
            self.calculator.calculate_relative_error(computed, reference)
    
    def test_calculate_relative_error_infinite_values(self):
        """Test relative error raises error with infinite values."""
        with pytest.raises(ValueError, match="Cannot calculate relative error with non-finite values"):
            self.calculator.calculate_relative_error(np.inf, 1.0)
        
        with pytest.raises(ValueError, match="Cannot calculate relative error with non-finite values"):
            self.calculator.calculate_relative_error(1.0, np.inf)
        
        with pytest.raises(ValueError, match="Cannot calculate relative error with non-finite values"):
            self.calculator.calculate_relative_error(np.nan, 1.0)
    
    def test_calculate_absolute_error(self):
        """Test absolute error calculation."""
        computed = 1.02
        reference = 1.0
        expected_error = 0.02
        
        error = self.calculator.calculate_absolute_error(computed, reference)
        assert abs(error - expected_error) < 1e-10
    
    def test_calculate_absolute_error_negative_difference(self):
        """Test absolute error with negative difference."""
        computed = 0.98
        reference = 1.0
        expected_error = 0.02
        
        error = self.calculator.calculate_absolute_error(computed, reference)
        assert abs(error - expected_error) < 1e-10
    
    def test_calculate_statistical_metrics_normal_data(self):
        """Test statistical metrics calculation for normal dataset."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        stats = self.calculator.calculate_statistical_metrics(data)
        
        assert stats.mean == 3.0
        assert abs(stats.std_dev - np.std(data, ddof=1)) < 1e-10
        assert stats.min_value == 1.0
        assert stats.max_value == 5.0
        assert stats.median == 3.0
        assert stats.count == 5
    
    def test_calculate_statistical_metrics_single_value(self):
        """Test statistical metrics for single value dataset."""
        data = [2.5]
        
        stats = self.calculator.calculate_statistical_metrics(data)
        
        assert stats.mean == 2.5
        assert stats.std_dev == 0.0  # Standard deviation is 0 for single value
        assert stats.min_value == 2.5
        assert stats.max_value == 2.5
        assert stats.median == 2.5
        assert stats.count == 1
    
    def test_calculate_statistical_metrics_empty_data(self):
        """Test statistical metrics raises error for empty dataset."""
        data = []
        
        with pytest.raises(ValueError, match="Cannot calculate statistics for empty dataset"):
            self.calculator.calculate_statistical_metrics(data)
    
    def test_calculate_statistical_metrics_non_finite_data(self):
        """Test statistical metrics raises error for non-finite values."""
        data = [1.0, 2.0, np.inf, 4.0]
        
        with pytest.raises(ValueError, match="Dataset contains non-finite values"):
            self.calculator.calculate_statistical_metrics(data)
    
    def test_assess_single_test_passing(self):
        """Test single test assessment for passing test."""
        result = self.calculator.assess_single_test(
            test_name="test_1",
            computed=1.01,
            reference=1.0,
            tolerance=0.02,
            weight=1.5
        )
        
        assert result.test_name == "test_1"
        assert result.computed_value == 1.01
        assert result.reference_value == 1.0
        assert abs(result.relative_error - 0.01) < 1e-10
        assert abs(result.absolute_error - 0.01) < 1e-10
        assert result.passed is True
        assert result.tolerance == 0.02
        assert result.weight == 1.5
    
    def test_assess_single_test_failing(self):
        """Test single test assessment for failing test."""
        result = self.calculator.assess_single_test(
            test_name="test_2",
            computed=1.05,
            reference=1.0,
            tolerance=0.02
        )
        
        assert result.test_name == "test_2"
        assert result.computed_value == 1.05
        assert result.reference_value == 1.0
        assert abs(result.relative_error - 0.05) < 1e-10
        assert result.passed is False
        assert result.tolerance == 0.02
        assert result.weight == 1.0  # Default weight
    
    def test_assess_single_test_default_tolerance(self):
        """Test single test assessment uses default tolerance when not specified."""
        result = self.calculator.assess_single_test(
            test_name="test_3",
            computed=1.01,
            reference=1.0
        )
        
        assert result.tolerance == 0.02  # Default tolerance
        assert result.passed is True
    
    def test_determine_overall_accuracy_all_perfect(self):
        """Test overall accuracy calculation with perfect results."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test2", 2.0, 2.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test3", 3.0, 3.0, 0.0, 0.0, True, 0.02, 1.0)
        ]
        
        accuracy = self.calculator.determine_overall_accuracy(results)
        assert accuracy == 100.0
    
    def test_determine_overall_accuracy_mixed_results(self):
        """Test overall accuracy calculation with mixed results."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),      # 100% accuracy
            AccuracyResult("test2", 1.02, 1.0, 0.02, 0.02, True, 0.02, 1.0),   # 98% accuracy
            AccuracyResult("test3", 1.05, 1.0, 0.05, 0.05, False, 0.02, 1.0)   # 95% accuracy
        ]
        
        accuracy = self.calculator.determine_overall_accuracy(results)
        expected_accuracy = (100.0 + 98.0 + 95.0) / 3.0
        assert abs(accuracy - expected_accuracy) < 1e-10
    
    def test_determine_overall_accuracy_weighted(self):
        """Test overall accuracy calculation with weighted results."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 2.0),      # 100% accuracy, weight 2
            AccuracyResult("test2", 1.05, 1.0, 0.05, 0.05, False, 0.02, 1.0)   # 95% accuracy, weight 1
        ]
        
        accuracy = self.calculator.determine_overall_accuracy(results)
        expected_accuracy = (100.0 * 2.0 + 95.0 * 1.0) / 3.0  # (200 + 95) / 3
        assert abs(accuracy - expected_accuracy) < 1e-10
    
    def test_determine_overall_accuracy_empty_results(self):
        """Test overall accuracy raises error for empty results."""
        results = []
        
        with pytest.raises(ValueError, match="Cannot calculate overall accuracy for empty results"):
            self.calculator.determine_overall_accuracy(results)
    
    def test_determine_overall_accuracy_zero_weight(self):
        """Test overall accuracy raises error when total weight is zero."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 0.0)
        ]
        
        with pytest.raises(ValueError, match="Total weight cannot be zero"):
            self.calculator.determine_overall_accuracy(results)
    
    def test_calculate_pass_rate_all_pass(self):
        """Test pass rate calculation when all tests pass."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test2", 1.01, 1.0, 0.01, 0.01, True, 0.02, 1.0)
        ]
        
        pass_rate = self.calculator.calculate_pass_rate(results)
        assert pass_rate == 100.0
    
    def test_calculate_pass_rate_partial_pass(self):
        """Test pass rate calculation with partial passes."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test2", 1.05, 1.0, 0.05, 0.05, False, 0.02, 1.0)
        ]
        
        pass_rate = self.calculator.calculate_pass_rate(results)
        assert pass_rate == 50.0
    
    def test_calculate_pass_rate_empty_results(self):
        """Test pass rate calculation for empty results."""
        results = []
        
        pass_rate = self.calculator.calculate_pass_rate(results)
        assert pass_rate == 0.0
    
    def test_assess_pass_fail_status_all_pass(self):
        """Test pass/fail status assessment when all tests pass."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test2", 1.01, 1.0, 0.01, 0.01, True, 0.02, 1.0)
        ]
        
        status = self.calculator.assess_pass_fail_status(results)
        
        assert status["test1"] is True
        assert status["test2"] is True
        assert status["overall"] is True
    
    def test_assess_pass_fail_status_partial_pass(self):
        """Test pass/fail status assessment with partial passes."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test2", 1.05, 1.0, 0.05, 0.05, False, 0.02, 1.0)
        ]
        
        status = self.calculator.assess_pass_fail_status(results)
        
        assert status["test1"] is True
        assert status["test2"] is False
        assert status["overall"] is False  # Overall fails if any test fails
    
    def test_generate_accuracy_summary_comprehensive(self):
        """Test comprehensive accuracy summary generation."""
        results = [
            AccuracyResult("test1", 1.0, 1.0, 0.0, 0.0, True, 0.02, 1.0),
            AccuracyResult("test2", 1.02, 1.0, 0.02, 0.02, True, 0.02, 1.0),
            AccuracyResult("test3", 1.05, 1.0, 0.05, 0.05, False, 0.02, 1.0)
        ]
        
        summary = self.calculator.generate_accuracy_summary(results)
        
        # Check main metrics
        assert "overall_accuracy" in summary
        assert "pass_rate" in summary
        assert summary["total_tests"] == 3
        assert summary["passed_tests"] == 2
        assert summary["failed_tests"] == 1
        
        # Check pass/fail status
        assert "pass_fail_status" in summary
        assert summary["pass_fail_status"]["test1"] is True
        assert summary["pass_fail_status"]["test2"] is True
        assert summary["pass_fail_status"]["test3"] is False
        assert summary["pass_fail_status"]["overall"] is False
        
        # Check error statistics
        assert "error_statistics" in summary
        error_stats = summary["error_statistics"]
        assert "mean_relative_error" in error_stats
        assert "std_relative_error" in error_stats
        assert "min_relative_error" in error_stats
        assert "max_relative_error" in error_stats
        assert "median_relative_error" in error_stats
        
        # Check individual results
        assert "individual_results" in summary
        assert len(summary["individual_results"]) == 3
        
        # Verify error statistics values
        expected_mean_error = (0.0 + 0.02 + 0.05) / 3.0
        assert abs(error_stats["mean_relative_error"] - expected_mean_error) < 1e-10
        assert error_stats["min_relative_error"] == 0.0
        assert error_stats["max_relative_error"] == 0.05
    
    def test_generate_accuracy_summary_empty_results(self):
        """Test accuracy summary generation for empty results."""
        results = []
        
        summary = self.calculator.generate_accuracy_summary(results)
        
        assert summary["overall_accuracy"] == 0.0
        assert summary["pass_rate"] == 0.0
        assert summary["total_tests"] == 0
        assert summary["passed_tests"] == 0
        assert summary["failed_tests"] == 0
        assert summary["pass_fail_status"] == {}
        assert summary["error_statistics"] is None
        assert summary["individual_results"] == []
    
    def test_logging_integration(self):
        """Test that logging is properly integrated."""
        # Test that logger is properly initialized
        assert hasattr(self.calculator, 'logger')
        
        # Test that methods can be called without logging errors
        # (actual logging output testing would require more complex setup)
        self.calculator.calculate_relative_error(1.02, 1.0)
        self.calculator.assess_single_test("test", 1.02, 1.0)
    
    def test_accuracy_result_dataclass(self):
        """Test AccuracyResult dataclass functionality."""
        result = AccuracyResult(
            test_name="test",
            computed_value=1.02,
            reference_value=1.0,
            relative_error=0.02,
            absolute_error=0.02,
            passed=True,
            tolerance=0.02,
            weight=1.5
        )
        
        assert result.test_name == "test"
        assert result.computed_value == 1.02
        assert result.reference_value == 1.0
        assert result.relative_error == 0.02
        assert result.absolute_error == 0.02
        assert result.passed is True
        assert result.tolerance == 0.02
        assert result.weight == 1.5
    
    def test_statistical_summary_dataclass(self):
        """Test StatisticalSummary dataclass functionality."""
        summary = StatisticalSummary(
            mean=2.5,
            std_dev=1.2,
            min_value=1.0,
            max_value=4.0,
            median=2.5,
            count=5
        )
        
        assert summary.mean == 2.5
        assert summary.std_dev == 1.2
        assert summary.min_value == 1.0
        assert summary.max_value == 4.0
        assert summary.median == 2.5
        assert summary.count == 5


class TestAccuracyCalculatorIntegration:
    """Integration tests for AccuracyCalculator with realistic scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = AccuracyCalculator(default_tolerance=0.02)
    
    def test_fmo_benchmark_scenario(self):
        """Test accuracy calculation for realistic FMO benchmark scenario."""
        # Simulate FMO coherence lifetime benchmark results
        results = [
            self.calculator.assess_single_test(
                "fmo_coherence_lifetime",
                computed=650.0,  # fs
                reference=660.0,  # fs (literature value)
                tolerance=0.15,   # 15% tolerance for biological systems
                weight=2.0        # High importance
            ),
            self.calculator.assess_single_test(
                "fmo_energy_transfer_efficiency",
                computed=0.95,
                reference=0.98,
                tolerance=0.05,
                weight=1.5
            ),
            self.calculator.assess_single_test(
                "fmo_excitation_energy",
                computed=12100.0,  # cm^-1
                reference=12000.0,  # cm^-1
                tolerance=0.02,
                weight=1.0
            )
        ]
        
        summary = self.calculator.generate_accuracy_summary(results)
        
        # All tests should pass with these tolerances
        assert summary["passed_tests"] == 3
        assert summary["pass_rate"] == 100.0
        assert summary["pass_fail_status"]["overall"] is True
        assert summary["overall_accuracy"] > 90.0  # Should be high accuracy
    
    def test_analytical_benchmark_scenario(self):
        """Test accuracy calculation for analytical benchmark scenario."""
        # Simulate analytical system benchmark results (should be very accurate)
        results = [
            self.calculator.assess_single_test(
                "two_level_rabi_oscillation",
                computed=1.0000001,
                reference=1.0,
                tolerance=1e-6,
                weight=1.0
            ),
            self.calculator.assess_single_test(
                "harmonic_oscillator_ground_energy",
                computed=0.5000000001,
                reference=0.5,
                tolerance=1e-9,
                weight=1.0
            ),
            self.calculator.assess_single_test(
                "coherent_state_evolution",
                computed=0.9999999,
                reference=1.0,
                tolerance=1e-6,
                weight=1.0
            )
        ]
        
        summary = self.calculator.generate_accuracy_summary(results)
        
        # All analytical tests should pass with very high accuracy
        assert summary["passed_tests"] == 3
        assert summary["pass_rate"] == 100.0
        assert summary["overall_accuracy"] > 99.9
        assert summary["error_statistics"]["max_relative_error"] < 1e-6