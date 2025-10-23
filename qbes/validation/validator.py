"""
QBES Autonomous Validation System

This module implements the QBESValidator class that provides autonomous execution
of validation suites with result analysis, accuracy threshold checking, and
retry logic for failed tests.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .accuracy_calculator import AccuracyCalculator, AccuracyResult
from ..benchmarks.benchmark_runner import BenchmarkRunner, ValidationResults


@dataclass
class ValidationConfig:
    """Configuration for autonomous validation execution."""
    suite_type: str = "standard"
    accuracy_threshold: float = 98.0
    pass_rate_threshold: float = 100.0
    max_retries: int = 3
    retry_delay: float = 1.0
    output_dir: Optional[str] = None
    tolerance: float = 0.02


@dataclass
class ValidationSummary:
    """Summary of autonomous validation execution."""
    total_attempts: int
    final_accuracy: float
    final_pass_rate: float
    certification_achieved: bool
    failed_tests: List[str]
    retry_history: List[Dict[str, Any]]
    execution_time: float


class QBESValidator:
    """
    Autonomous validation execution and analysis system for QBES.
    
    This class provides comprehensive validation capabilities including:
    - Autonomous execution of validation suites
    - Accuracy threshold checking and pass/fail determination
    - Retry logic for failed tests with systematic debugging
    - Result analysis and certification assessment
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the QBES validator.
        
        Args:
            config: ValidationConfig object with validation parameters
        """
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.accuracy_calculator = AccuracyCalculator(default_tolerance=self.config.tolerance)
        self.benchmark_runner = BenchmarkRunner(output_dir=self.config.output_dir)
        
        # Validation state
        self.validation_history: List[ValidationResults] = []
        self.retry_count = 0
        
        self.logger.info(f"QBESValidator initialized with config: {self.config}")
    
    def validate_against_benchmarks(self, suite: str = None) -> ValidationSummary:
        """
        Execute autonomous validation against benchmark suite.
        
        Args:
            suite: Suite type to run ('quick', 'standard', 'full')
                  If None, uses config default
        
        Returns:
            ValidationSummary with complete validation results
        """
        start_time = time.time()
        suite = suite or self.config.suite_type
        
        self.logger.info(f"Starting autonomous validation with suite: {suite}")
        
        retry_history = []
        final_results = None
        certification_achieved = False
        
        for attempt in range(self.config.max_retries + 1):
            self.logger.info(f"Validation attempt {attempt + 1}/{self.config.max_retries + 1}")
            
            try:
                # Execute validation suite
                results = self.benchmark_runner.run_validation_suite(suite)
                self.validation_history.append(results)
                
                # Analyze results
                analysis = self._analyze_validation_results(results)
                retry_history.append({
                    'attempt': attempt + 1,
                    'accuracy': analysis['accuracy'],
                    'pass_rate': analysis['pass_rate'],
                    'passed': analysis['certification_achieved'],
                    'failed_tests': analysis['failed_tests']
                })
                
                self.logger.info(f"Attempt {attempt + 1} results: "
                               f"Accuracy={analysis['accuracy']:.1f}%, "
                               f"Pass Rate={analysis['pass_rate']:.1f}%")
                
                # Check if certification criteria met
                if analysis['certification_achieved']:
                    self.logger.info("🎉 Certification criteria achieved!")
                    certification_achieved = True
                    final_results = results
                    break
                
                # If not the last attempt, prepare for retry
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Certification not achieved. Preparing retry {attempt + 2}...")
                    self._prepare_for_retry(analysis)
                    time.sleep(self.config.retry_delay)
                else:
                    self.logger.error("Maximum retries reached. Certification not achieved.")
                    final_results = results
                
            except Exception as e:
                self.logger.error(f"Validation attempt {attempt + 1} failed: {e}")
                retry_history.append({
                    'attempt': attempt + 1,
                    'error': str(e),
                    'passed': False
                })
                
                if attempt == self.config.max_retries:
                    raise
        
        execution_time = time.time() - start_time
        
        # Create final summary
        if final_results:
            final_analysis = self._analyze_validation_results(final_results)
            summary = ValidationSummary(
                total_attempts=len(retry_history),
                final_accuracy=final_analysis['accuracy'],
                final_pass_rate=final_analysis['pass_rate'],
                certification_achieved=certification_achieved,
                failed_tests=final_analysis['failed_tests'],
                retry_history=retry_history,
                execution_time=execution_time
            )
        else:
            summary = ValidationSummary(
                total_attempts=len(retry_history),
                final_accuracy=0.0,
                final_pass_rate=0.0,
                certification_achieved=False,
                failed_tests=[],
                retry_history=retry_history,
                execution_time=execution_time
            )
        
        self._log_validation_summary(summary)
        return summary
    
    def _analyze_validation_results(self, results: ValidationResults) -> Dict[str, Any]:
        """
        Analyze validation results against certification criteria.
        
        Args:
            results: ValidationResults from benchmark execution
            
        Returns:
            Dictionary with analysis results
        """
        # Convert ValidationResults to AccuracyResult format
        accuracy_results = []
        for result in results.results:
            accuracy_result = AccuracyResult(
                test_name=result.test_name,
                computed_value=result.computed_value,
                reference_value=result.reference_value,
                relative_error=result.relative_error,
                absolute_error=abs(result.computed_value - result.reference_value),
                passed=result.passed,
                tolerance=result.tolerance
            )
            accuracy_results.append(accuracy_result)
        
        # Calculate comprehensive accuracy metrics
        if accuracy_results:
            overall_accuracy = self.accuracy_calculator.determine_overall_accuracy(accuracy_results)
            pass_rate = self.accuracy_calculator.calculate_pass_rate(accuracy_results)
            pass_fail_status = self.accuracy_calculator.assess_pass_fail_status(accuracy_results)
        else:
            overall_accuracy = 0.0
            pass_rate = 0.0
            pass_fail_status = {'overall': False}
        
        # Determine certification status
        certification_achieved = (
            overall_accuracy >= self.config.accuracy_threshold and
            pass_rate >= self.config.pass_rate_threshold
        )
        
        # Identify failed tests
        failed_tests = [
            result.test_name for result in accuracy_results 
            if not result.passed
        ]
        
        analysis = {
            'accuracy': overall_accuracy,
            'pass_rate': pass_rate,
            'certification_achieved': certification_achieved,
            'failed_tests': failed_tests,
            'pass_fail_status': pass_fail_status,
            'accuracy_results': accuracy_results,
            'meets_accuracy_threshold': overall_accuracy >= self.config.accuracy_threshold,
            'meets_pass_rate_threshold': pass_rate >= self.config.pass_rate_threshold
        }
        
        return analysis
    
    def _prepare_for_retry(self, analysis: Dict[str, Any]):
        """
        Prepare system for retry attempt based on analysis results.
        
        Args:
            analysis: Analysis results from previous attempt
        """
        failed_tests = analysis['failed_tests']
        
        if failed_tests:
            self.logger.warning(f"Failed tests identified: {failed_tests}")
            
            # Log specific failure analysis
            for result in analysis['accuracy_results']:
                if not result.passed:
                    self.logger.warning(
                        f"Test '{result.test_name}' failed: "
                        f"computed={result.computed_value:.6f}, "
                        f"reference={result.reference_value:.6f}, "
                        f"error={result.relative_error:.4f} "
                        f"(tolerance={result.tolerance:.4f})"
                    )
        
        # Log accuracy analysis
        if not analysis['meets_accuracy_threshold']:
            self.logger.warning(
                f"Accuracy threshold not met: {analysis['accuracy']:.1f}% < "
                f"{self.config.accuracy_threshold:.1f}%"
            )
        
        if not analysis['meets_pass_rate_threshold']:
            self.logger.warning(
                f"Pass rate threshold not met: {analysis['pass_rate']:.1f}% < "
                f"{self.config.pass_rate_threshold:.1f}%"
            )
        
        # Increment retry counter
        self.retry_count += 1
        
        # Log retry preparation
        self.logger.info(f"Preparing for retry {self.retry_count + 1}...")
        self.logger.info("Potential improvements:")
        self.logger.info("- Check numerical precision settings")
        self.logger.info("- Verify simulation parameters")
        self.logger.info("- Review algorithm implementations")
    
    def calculate_accuracy_score(self, results: ValidationResults) -> float:
        """
        Calculate overall accuracy score for validation results.
        
        Args:
            results: ValidationResults from benchmark execution
            
        Returns:
            Overall accuracy score as percentage (0-100)
        """
        if not results.results:
            return 0.0
        
        # Convert to AccuracyResult format
        accuracy_results = []
        for result in results.results:
            accuracy_result = AccuracyResult(
                test_name=result.test_name,
                computed_value=result.computed_value,
                reference_value=result.reference_value,
                relative_error=result.relative_error,
                absolute_error=abs(result.computed_value - result.reference_value),
                passed=result.passed,
                tolerance=result.tolerance
            )
            accuracy_results.append(accuracy_result)
        
        return self.accuracy_calculator.determine_overall_accuracy(accuracy_results)
    
    def assess_pass_fail_status(self, results: ValidationResults) -> Dict[str, bool]:
        """
        Assess pass/fail status for validation results.
        
        Args:
            results: ValidationResults from benchmark execution
            
        Returns:
            Dictionary with test names and pass/fail status
        """
        if not results.results:
            return {'overall': False}
        
        # Convert to AccuracyResult format
        accuracy_results = []
        for result in results.results:
            accuracy_result = AccuracyResult(
                test_name=result.test_name,
                computed_value=result.computed_value,
                reference_value=result.reference_value,
                relative_error=result.relative_error,
                absolute_error=abs(result.computed_value - result.reference_value),
                passed=result.passed,
                tolerance=result.tolerance
            )
            accuracy_results.append(accuracy_result)
        
        return self.accuracy_calculator.assess_pass_fail_status(accuracy_results)
    
    def execute_validation_loop(self, suite: str = None, max_iterations: int = None) -> ValidationSummary:
        """
        Execute validation loop with automatic retry until certification achieved.
        
        Args:
            suite: Suite type to run
            max_iterations: Maximum iterations (overrides config if provided)
            
        Returns:
            ValidationSummary with complete execution results
        """
        if max_iterations is not None:
            original_max_retries = self.config.max_retries
            self.config.max_retries = max_iterations - 1
        
        try:
            summary = self.validate_against_benchmarks(suite)
            return summary
        finally:
            if max_iterations is not None:
                self.config.max_retries = original_max_retries
    
    def generate_certification_report(self, summary: ValidationSummary) -> str:
        """
        Generate certification report based on validation summary.
        
        Args:
            summary: ValidationSummary from validation execution
            
        Returns:
            Markdown-formatted certification report
        """
        report = f"""# QBES Autonomous Validation Certification Report

**Validation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**QBES Version:** 1.2.0
**Validation Suite:** {self.config.suite_type}

## Certification Status

"""
        
        if summary.certification_achieved:
            report += "🎉 **CERTIFIED** - QBES meets all validation criteria\n\n"
        else:
            report += "❌ **NOT CERTIFIED** - QBES requires improvements\n\n"
        
        report += f"""## Final Results

- **Overall Accuracy:** {summary.final_accuracy:.2f}%
- **Pass Rate:** {summary.final_pass_rate:.1f}%
- **Total Attempts:** {summary.total_attempts}
- **Execution Time:** {summary.execution_time:.1f} seconds

## Certification Criteria

- **Accuracy Threshold:** {self.config.accuracy_threshold:.1f}% ({'✅ MET' if summary.final_accuracy >= self.config.accuracy_threshold else '❌ NOT MET'})
- **Pass Rate Threshold:** {self.config.pass_rate_threshold:.1f}% ({'✅ MET' if summary.final_pass_rate >= self.config.pass_rate_threshold else '❌ NOT MET'})

"""
        
        if summary.failed_tests:
            report += "## Failed Tests\n\n"
            for test in summary.failed_tests:
                report += f"- {test}\n"
            report += "\n"
        
        if summary.retry_history:
            report += "## Retry History\n\n"
            report += "| Attempt | Accuracy | Pass Rate | Status |\n"
            report += "|---------|----------|-----------|--------|\n"
            
            for entry in summary.retry_history:
                if 'error' in entry:
                    report += f"| {entry['attempt']} | - | - | ERROR |\n"
                else:
                    status = "✅ PASS" if entry['passed'] else "❌ FAIL"
                    report += f"| {entry['attempt']} | {entry['accuracy']:.1f}% | {entry['pass_rate']:.1f}% | {status} |\n"
        
        report += "\n## Recommendations\n\n"
        
        if summary.certification_achieved:
            report += "QBES has successfully achieved certification criteria. The system is validated for scientific use.\n"
        else:
            report += "QBES requires the following improvements:\n\n"
            
            if summary.final_accuracy < self.config.accuracy_threshold:
                report += f"- Improve numerical accuracy to reach {self.config.accuracy_threshold:.1f}% threshold\n"
            
            if summary.final_pass_rate < self.config.pass_rate_threshold:
                report += f"- Address failed tests to achieve {self.config.pass_rate_threshold:.1f}% pass rate\n"
            
            if summary.failed_tests:
                report += "- Focus on the following failed tests:\n"
                for test in summary.failed_tests:
                    report += f"  - {test}\n"
        
        return report
    
    def _log_validation_summary(self, summary: ValidationSummary):
        """Log comprehensive validation summary."""
        self.logger.info("=" * 60)
        self.logger.info("AUTONOMOUS VALIDATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Final Accuracy: {summary.final_accuracy:.2f}%")
        self.logger.info(f"Final Pass Rate: {summary.final_pass_rate:.1f}%")
        self.logger.info(f"Certification: {'ACHIEVED' if summary.certification_achieved else 'NOT ACHIEVED'}")
        self.logger.info(f"Total Attempts: {summary.total_attempts}")
        self.logger.info(f"Execution Time: {summary.execution_time:.1f} seconds")
        
        if summary.failed_tests:
            self.logger.warning(f"Failed Tests: {', '.join(summary.failed_tests)}")
        
        self.logger.info("=" * 60)
    
    def get_validation_history(self) -> List[ValidationResults]:
        """
        Get complete validation history.
        
        Returns:
            List of ValidationResults from all attempts
        """
        return self.validation_history.copy()
    
    def reset_validation_state(self):
        """Reset validation state for fresh execution."""
        self.validation_history.clear()
        self.retry_count = 0
        self.logger.info("Validation state reset")