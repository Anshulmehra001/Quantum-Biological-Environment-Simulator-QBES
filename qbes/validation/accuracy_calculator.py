"""
Accuracy Calculator for QBES Validation System

This module provides comprehensive accuracy calculation and statistical analysis
for validation benchmarks, including relative error computation, statistical
metrics, and pass/fail assessment.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalSummary:
    """Statistical summary of validation results."""
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    median: float
    count: int


@dataclass
class AccuracyResult:
    """Result of accuracy calculation for a single test."""
    test_name: str
    computed_value: float
    reference_value: float
    relative_error: float
    absolute_error: float
    passed: bool
    tolerance: float
    weight: float = 1.0


class AccuracyCalculator:
    """
    Calculates accuracy metrics and statistical summaries for validation results.
    
    This class provides methods for computing relative errors, statistical metrics,
    overall accuracy scores with weighted averaging, and pass/fail assessments
    based on tolerance thresholds.
    """
    
    def __init__(self, default_tolerance: float = 0.02):
        """
        Initialize the AccuracyCalculator.
        
        Args:
            default_tolerance: Default tolerance for pass/fail assessment (default: 2%)
        """
        self.default_tolerance = default_tolerance
        self.logger = logging.getLogger(__name__)
    
    def calculate_relative_error(self, computed: float, reference: float) -> float:
        """
        Calculate relative error between computed and reference values.
        
        Args:
            computed: Computed value from simulation
            reference: Reference value from literature/analytical solution
            
        Returns:
            Relative error as a fraction (not percentage)
            
        Raises:
            ValueError: If reference value is zero or invalid
        """
        if reference == 0.0:
            if computed == 0.0:
                return 0.0
            else:
                raise ValueError("Cannot calculate relative error with zero reference value")
        
        if not np.isfinite(computed) or not np.isfinite(reference):
            raise ValueError("Cannot calculate relative error with non-finite values")
        
        relative_error = abs(computed - reference) / abs(reference)
        
        self.logger.debug(f"Relative error calculation: |{computed} - {reference}| / |{reference}| = {relative_error}")
        
        return relative_error
    
    def calculate_absolute_error(self, computed: float, reference: float) -> float:
        """
        Calculate absolute error between computed and reference values.
        
        Args:
            computed: Computed value from simulation
            reference: Reference value from literature/analytical solution
            
        Returns:
            Absolute error
        """
        return abs(computed - reference)
    
    def calculate_statistical_metrics(self, data: List[float]) -> StatisticalSummary:
        """
        Calculate comprehensive statistical metrics for a dataset.
        
        Args:
            data: List of numerical values
            
        Returns:
            StatisticalSummary containing mean, std dev, min, max, median, count
            
        Raises:
            ValueError: If data is empty or contains non-finite values
        """
        if not data:
            raise ValueError("Cannot calculate statistics for empty dataset")
        
        data_array = np.array(data)
        
        if not np.all(np.isfinite(data_array)):
            raise ValueError("Dataset contains non-finite values")
        
        summary = StatisticalSummary(
            mean=float(np.mean(data_array)),
            std_dev=float(np.std(data_array, ddof=1)) if len(data) > 1 else 0.0,
            min_value=float(np.min(data_array)),
            max_value=float(np.max(data_array)),
            median=float(np.median(data_array)),
            count=len(data)
        )
        
        self.logger.debug(f"Statistical summary: mean={summary.mean:.6f}, "
                         f"std={summary.std_dev:.6f}, count={summary.count}")
        
        return summary
    
    def assess_single_test(self, 
                          test_name: str,
                          computed: float, 
                          reference: float,
                          tolerance: Optional[float] = None,
                          weight: float = 1.0) -> AccuracyResult:
        """
        Assess accuracy for a single validation test.
        
        Args:
            test_name: Name of the test
            computed: Computed value from simulation
            reference: Reference value
            tolerance: Tolerance threshold (uses default if None)
            weight: Weight for this test in overall scoring
            
        Returns:
            AccuracyResult with all metrics and pass/fail status
        """
        if tolerance is None:
            tolerance = self.default_tolerance
        
        relative_error = self.calculate_relative_error(computed, reference)
        absolute_error = self.calculate_absolute_error(computed, reference)
        passed = relative_error <= tolerance
        
        result = AccuracyResult(
            test_name=test_name,
            computed_value=computed,
            reference_value=reference,
            relative_error=relative_error,
            absolute_error=absolute_error,
            passed=passed,
            tolerance=tolerance,
            weight=weight
        )
        
        self.logger.info(f"Test '{test_name}': {computed:.6f} vs {reference:.6f}, "
                        f"rel_error={relative_error:.4f}, passed={passed}")
        
        return result
    
    def determine_overall_accuracy(self, results: List[AccuracyResult]) -> float:
        """
        Calculate overall accuracy score using weighted averaging.
        
        Args:
            results: List of AccuracyResult objects
            
        Returns:
            Overall accuracy score as percentage (0-100)
            
        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("Cannot calculate overall accuracy for empty results")
        
        total_weight = sum(result.weight for result in results)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        
        # Calculate weighted accuracy score
        # Accuracy = (1 - relative_error) * 100, weighted by test importance
        weighted_accuracy_sum = 0.0
        
        for result in results:
            # Convert relative error to accuracy percentage
            test_accuracy = max(0.0, (1.0 - result.relative_error)) * 100.0
            weighted_accuracy_sum += test_accuracy * result.weight
        
        overall_accuracy = weighted_accuracy_sum / total_weight
        
        self.logger.info(f"Overall accuracy: {overall_accuracy:.2f}% "
                        f"(weighted average of {len(results)} tests)")
        
        return overall_accuracy
    
    def calculate_pass_rate(self, results: List[AccuracyResult]) -> float:
        """
        Calculate the pass rate as percentage of tests that passed.
        
        Args:
            results: List of AccuracyResult objects
            
        Returns:
            Pass rate as percentage (0-100)
        """
        if not results:
            return 0.0
        
        passed_count = sum(1 for result in results if result.passed)
        pass_rate = (passed_count / len(results)) * 100.0
        
        self.logger.info(f"Pass rate: {pass_rate:.1f}% ({passed_count}/{len(results)} tests passed)")
        
        return pass_rate
    
    def assess_pass_fail_status(self, results: List[AccuracyResult]) -> Dict[str, bool]:
        """
        Assess pass/fail status for individual tests and overall suite.
        
        Args:
            results: List of AccuracyResult objects
            
        Returns:
            Dictionary with test names as keys and pass/fail status as values,
            plus 'overall' key for suite-level status
        """
        status_dict = {}
        
        # Individual test status
        for result in results:
            status_dict[result.test_name] = result.passed
        
        # Overall status - all tests must pass
        overall_passed = all(result.passed for result in results)
        status_dict['overall'] = overall_passed
        
        passed_count = sum(1 for result in results if result.passed)
        self.logger.info(f"Pass/fail assessment: {passed_count}/{len(results)} tests passed, "
                        f"overall={'PASS' if overall_passed else 'FAIL'}")
        
        return status_dict
    
    def generate_accuracy_summary(self, results: List[AccuracyResult]) -> Dict[str, Any]:
        """
        Generate comprehensive accuracy summary for validation results.
        
        Args:
            results: List of AccuracyResult objects
            
        Returns:
            Dictionary containing all accuracy metrics and statistics
        """
        if not results:
            return {
                'overall_accuracy': 0.0,
                'pass_rate': 0.0,
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'pass_fail_status': {},
                'error_statistics': None,
                'individual_results': []
            }
        
        overall_accuracy = self.determine_overall_accuracy(results)
        pass_rate = self.calculate_pass_rate(results)
        pass_fail_status = self.assess_pass_fail_status(results)
        
        # Calculate error statistics
        relative_errors = [result.relative_error for result in results]
        error_stats = self.calculate_statistical_metrics(relative_errors)
        
        passed_count = sum(1 for result in results if result.passed)
        failed_count = len(results) - passed_count
        
        summary = {
            'overall_accuracy': overall_accuracy,
            'pass_rate': pass_rate,
            'total_tests': len(results),
            'passed_tests': passed_count,
            'failed_tests': failed_count,
            'pass_fail_status': pass_fail_status,
            'error_statistics': {
                'mean_relative_error': error_stats.mean,
                'std_relative_error': error_stats.std_dev,
                'min_relative_error': error_stats.min_value,
                'max_relative_error': error_stats.max_value,
                'median_relative_error': error_stats.median
            },
            'individual_results': [
                {
                    'test_name': result.test_name,
                    'computed_value': result.computed_value,
                    'reference_value': result.reference_value,
                    'relative_error': result.relative_error,
                    'absolute_error': result.absolute_error,
                    'passed': result.passed,
                    'tolerance': result.tolerance,
                    'weight': result.weight
                }
                for result in results
            ]
        }
        
        self.logger.info(f"Generated accuracy summary: {overall_accuracy:.2f}% accuracy, "
                        f"{pass_rate:.1f}% pass rate")
        
        return summary