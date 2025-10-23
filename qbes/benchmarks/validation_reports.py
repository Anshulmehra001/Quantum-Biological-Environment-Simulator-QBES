"""
Comprehensive validation report generation.

This module generates detailed validation reports combining literature validation,
cross-validation, and statistical analysis results.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from .literature_validation import LiteratureValidator, LiteratureValidationResult
from .cross_validation import CrossValidator, CrossValidationResult
from .statistical_testing import StatisticalTester, ComprehensiveStatisticalReport, perform_statistical_validation
from .benchmark_systems import BenchmarkRunner, BenchmarkResult


class ValidationSummary:
    """Summary of all validation results."""
    
    def __init__(self, timestamp: str = None, qbes_version: str = "1.1.0",
                 benchmark_tests_total: int = 0, benchmark_tests_passed: int = 0, benchmark_success_rate: float = 0.0,
                 literature_validations_total: int = 0, literature_validations_passed: int = 0, literature_success_rate: float = 0.0,
                 cross_validations_total: int = 0, cross_validations_passed: int = 0, cross_validation_success_rate: float = 0.0,
                 overall_validation_score: float = 0.0, validation_grade: str = "F",
                 critical_issues: List[str] = None, recommendations: List[str] = None,
                 # Legacy parameters for backward compatibility
                 benchmark_score: float = None, literature_score: float = None, 
                 cross_validation_score: float = None, statistical_score: float = None):
        """Initialize ValidationSummary with support for both new and legacy parameter formats."""
        
        # Handle legacy parameters if provided
        if any(param is not None for param in [benchmark_score, literature_score, cross_validation_score, statistical_score]):
            self._init_from_legacy_params(benchmark_score or 0.0, literature_score or 0.0, 
                                        cross_validation_score or 0.0, statistical_score or 0.0)
        else:
            # Use new parameter format
            self.timestamp = timestamp or datetime.now().isoformat()
            self.qbes_version = qbes_version
            self.benchmark_tests_total = benchmark_tests_total
            self.benchmark_tests_passed = benchmark_tests_passed
            self.benchmark_success_rate = benchmark_success_rate
            self.literature_validations_total = literature_validations_total
            self.literature_validations_passed = literature_validations_passed
            self.literature_success_rate = literature_success_rate
            self.cross_validations_total = cross_validations_total
            self.cross_validations_passed = cross_validations_passed
            self.cross_validation_success_rate = cross_validation_score
            self.overall_validation_score = overall_validation_score
            self.validation_grade = validation_grade
            self.critical_issues = critical_issues or []
            self.recommendations = recommendations or []
    
    def _init_from_legacy_params(self, benchmark_score: float, literature_score: float, 
                               cross_validation_score: float, statistical_score: float):
        """Initialize from legacy parameter format for backward compatibility."""
        # Calculate overall score from individual scores
        scores = [benchmark_score, literature_score, cross_validation_score, statistical_score]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine grade based on overall score
        if overall_score >= 0.9:
            grade = "A"
        elif overall_score >= 0.8:
            grade = "B"
        elif overall_score >= 0.7:
            grade = "C"
        elif overall_score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        # Convert scores to counts (assuming 10 tests each for simplicity)
        benchmark_total = 10
        benchmark_passed = int(benchmark_score * benchmark_total)
        
        literature_total = 10
        literature_passed = int(literature_score * literature_total)
        
        cross_validation_total = 10
        cross_validation_passed = int(cross_validation_score * cross_validation_total)
        
        # Set all attributes
        self.timestamp = datetime.now().isoformat()
        self.qbes_version = "1.1.0"
        self.benchmark_tests_total = benchmark_total
        self.benchmark_tests_passed = benchmark_passed
        self.benchmark_success_rate = benchmark_score
        self.literature_validations_total = literature_total
        self.literature_validations_passed = literature_passed
        self.literature_success_rate = literature_score
        self.cross_validations_total = cross_validation_total
        self.cross_validations_passed = cross_validation_passed
        self.cross_validation_success_rate = cross_validation_score
        self.overall_validation_score = overall_score
        self.validation_grade = grade
        self.critical_issues = []
        self.recommendations = []


class ComprehensiveValidationReporter:
    """
    Comprehensive validation report generator.
    
    Combines results from benchmark tests, literature validation,
    cross-validation, and statistical analysis into unified reports.
    """
    
    def __init__(self, output_dir: str = "validation_reports"):
        """
        Initialize validation reporter.
        
        Args:
            output_dir: Directory to save validation reports
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
        
        # Validation components
        self.benchmark_runner = None
        self.literature_validator = None
        self.cross_validator = None
        self.statistical_tester = StatisticalTester()
        
        # Results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.literature_results: List[LiteratureValidationResult] = []
        self.cross_validation_results: List[CrossValidationResult] = []
        self.statistical_reports: List[ComprehensiveStatisticalReport] = []
    
    def ensure_output_directory(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_complete_validation_suite(self, 
                                    include_benchmarks: bool = True,
                                    include_literature: bool = True,
                                    include_cross_validation: bool = True,
                                    max_performance_size: int = 16) -> ValidationSummary:
        """
        Run complete validation suite with all components.
        
        Args:
            include_benchmarks: Whether to run benchmark tests
            include_literature: Whether to run literature validation
            include_cross_validation: Whether to run cross-validation
            max_performance_size: Maximum system size for performance tests
            
        Returns:
            ValidationSummary with overall results
        """
        print("Running Complete QBES Validation Suite")
        print("=" * 50)
        
        # 1. Run benchmark tests
        if include_benchmarks:
            print("1. Running Benchmark Tests...")
            self.benchmark_runner = BenchmarkRunner()
            self.benchmark_runner.add_standard_benchmarks()
            self.benchmark_results = self.benchmark_runner.run_all_benchmarks()
            print(f"   Completed {len(self.benchmark_results)} benchmark tests")
        
        # 2. Run literature validation
        if include_literature:
            print("\n2. Running Literature Validation...")
            self.literature_validator = LiteratureValidator()
            self.literature_validator.add_standard_datasets()
            self.literature_results = self.literature_validator.validate_all_datasets()
            print(f"   Completed {len(self.literature_results)} literature validations")
        
        # 3. Run cross-validation
        if include_cross_validation:
            print("\n3. Running Cross-Validation...")
            self.cross_validator = CrossValidator()
            self.cross_validator.add_standard_interfaces()
            self.cross_validation_results = self.cross_validator.run_cross_validation_suite()
            print(f"   Completed {len(self.cross_validation_results)} cross-validations")
        
        # 4. Perform statistical analysis
        print("\n4. Performing Statistical Analysis...")
        self._perform_statistical_analysis()
        print(f"   Generated {len(self.statistical_reports)} statistical reports")
        
        # 5. Generate summary
        print("\n5. Generating Validation Summary...")
        summary = self._generate_validation_summary()
        
        # 6. Save all results
        self._save_all_results(summary)
        
        print(f"\nValidation suite completed!")
        print(f"Overall validation score: {summary.overall_validation_score:.1%}")
        print(f"Validation grade: {summary.validation_grade}")
        
        return summary
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis on validation results."""
        self.statistical_reports = []
        
        # Analyze benchmark results
        if self.benchmark_results:
            benchmark_observed = [r.numerical_result for r in self.benchmark_results if r.test_passed]
            benchmark_expected = [r.analytical_result for r in self.benchmark_results if r.test_passed]
            
            if len(benchmark_observed) >= 3:
                report = perform_statistical_validation(benchmark_observed, benchmark_expected)
                self.statistical_reports.append(report)
        
        # Analyze literature validation results
        if self.literature_results:
            for lit_result in self.literature_results:
                if lit_result.validation_passed and len(lit_result.simulated_values) >= 3:
                    report = perform_statistical_validation(
                        lit_result.simulated_values,
                        lit_result.experimental_values
                    )
                    self.statistical_reports.append(report)
        
        # Analyze cross-validation results
        if self.cross_validation_results:
            for cv_result in self.cross_validation_results:
                if cv_result.validation_passed and len(cv_result.qbes_results) >= 3:
                    report = perform_statistical_validation(
                        cv_result.qbes_results,
                        cv_result.reference_results
                    )
                    self.statistical_reports.append(report)
    
    def _generate_validation_summary(self) -> ValidationSummary:
        """Generate comprehensive validation summary."""
        
        # Benchmark statistics
        benchmark_total = len(self.benchmark_results)
        benchmark_passed = sum(1 for r in self.benchmark_results if r.test_passed)
        benchmark_success_rate = benchmark_passed / benchmark_total if benchmark_total > 0 else 0.0
        
        # Literature validation statistics
        literature_total = len(self.literature_results)
        literature_passed = sum(1 for r in self.literature_results if r.validation_passed)
        literature_success_rate = literature_passed / literature_total if literature_total > 0 else 0.0
        
        # Cross-validation statistics
        cross_validation_total = len(self.cross_validation_results)
        cross_validation_passed = sum(1 for r in self.cross_validation_results if r.validation_passed)
        cross_validation_success_rate = cross_validation_passed / cross_validation_total if cross_validation_total > 0 else 0.0
        
        # Calculate overall validation score (weighted average)
        weights = {'benchmark': 0.4, 'literature': 0.4, 'cross_validation': 0.2}
        
        overall_score = 0.0
        total_weight = 0.0
        
        if benchmark_total > 0:
            overall_score += weights['benchmark'] * benchmark_success_rate
            total_weight += weights['benchmark']
        
        if literature_total > 0:
            overall_score += weights['literature'] * literature_success_rate
            total_weight += weights['literature']
        
        if cross_validation_total > 0:
            overall_score += weights['cross_validation'] * cross_validation_success_rate
            total_weight += weights['cross_validation']
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Determine validation grade
        if overall_score >= 0.95:
            grade = "A+"
        elif overall_score >= 0.90:
            grade = "A"
        elif overall_score >= 0.85:
            grade = "A-"
        elif overall_score >= 0.80:
            grade = "B+"
        elif overall_score >= 0.75:
            grade = "B"
        elif overall_score >= 0.70:
            grade = "B-"
        elif overall_score >= 0.65:
            grade = "C+"
        elif overall_score >= 0.60:
            grade = "C"
        else:
            grade = "F"
        
        # Identify critical issues
        critical_issues = []
        
        if benchmark_success_rate < 0.8:
            critical_issues.append("Low benchmark test success rate")
        
        if literature_success_rate < 0.7:
            critical_issues.append("Poor agreement with literature data")
        
        if cross_validation_success_rate < 0.7:
            critical_issues.append("Inconsistent results compared to other packages")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            benchmark_success_rate, literature_success_rate, cross_validation_success_rate
        )
        
        return ValidationSummary(
            timestamp=datetime.now().isoformat(),
            qbes_version="1.0.0",  # TODO: Get from package metadata
            benchmark_tests_total=benchmark_total,
            benchmark_tests_passed=benchmark_passed,
            benchmark_success_rate=benchmark_success_rate,
            literature_validations_total=literature_total,
            literature_validations_passed=literature_passed,
            literature_success_rate=literature_success_rate,
            cross_validations_total=cross_validation_total,
            cross_validations_passed=cross_validation_passed,
            cross_validation_success_rate=cross_validation_success_rate,
            overall_validation_score=overall_score,
            validation_grade=grade,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, 
                                benchmark_rate: float,
                                literature_rate: float,
                                cross_validation_rate: float) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Benchmark recommendations
        if benchmark_rate < 0.9:
            recommendations.append("Review and improve core quantum mechanical algorithms")
        
        if benchmark_rate < 0.7:
            recommendations.append("Critical: Multiple benchmark failures indicate fundamental issues")
        
        # Literature recommendations
        if literature_rate < 0.8:
            recommendations.append("Improve noise models and environmental coupling parameters")
        
        if literature_rate < 0.6:
            recommendations.append("Critical: Poor literature agreement suggests model inadequacy")
        
        # Cross-validation recommendations
        if cross_validation_rate < 0.8:
            recommendations.append("Review implementation against established quantum simulation methods")
        
        # Statistical recommendations
        if self.statistical_reports:
            low_power_reports = [r for r in self.statistical_reports 
                               if r.power_analysis and r.power_analysis.get('statistical_power', 1.0) < 0.8]
            if low_power_reports:
                recommendations.append("Increase sample sizes for more robust statistical validation")
        
        # Overall recommendations
        if benchmark_rate > 0.9 and literature_rate > 0.8 and cross_validation_rate > 0.8:
            recommendations.append("Excellent validation results. QBES is ready for scientific use.")
        elif benchmark_rate > 0.8 and literature_rate > 0.7:
            recommendations.append("Good validation results with minor areas for improvement.")
        else:
            recommendations.append("Significant validation issues require attention before scientific use.")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive validation report."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("QBES COMPREHENSIVE VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"QBES Version: 1.0.0")
        lines.append("")
        
        # Executive Summary
        if hasattr(self, '_summary'):
            summary = self._summary
            lines.append("EXECUTIVE SUMMARY")
            lines.append("-" * 17)
            lines.append(f"Overall Validation Score: {summary.overall_validation_score:.1%}")
            lines.append(f"Validation Grade: {summary.validation_grade}")
            lines.append("")
            
            if summary.critical_issues:
                lines.append("Critical Issues:")
                for issue in summary.critical_issues:
                    lines.append(f"  ⚠️  {issue}")
                lines.append("")
        
        # Benchmark Results
        if self.benchmark_results:
            lines.append("BENCHMARK TEST RESULTS")
            lines.append("-" * 22)
            
            passed = sum(1 for r in self.benchmark_results if r.test_passed)
            total = len(self.benchmark_results)
            
            lines.append(f"Tests Run: {total}")
            lines.append(f"Passed: {passed}")
            lines.append(f"Failed: {total - passed}")
            lines.append(f"Success Rate: {(passed/total)*100:.1f}%")
            lines.append("")
            
            lines.append("Individual Test Results:")
            for result in self.benchmark_results:
                status = "✅ PASS" if result.test_passed else "❌ FAIL"
                lines.append(f"  {result.system_name}: {status}")
                lines.append(f"    Relative Error: {result.relative_error:.2e}")
                lines.append(f"    Computation Time: {result.computation_time:.3f}s")
                if result.error_message:
                    lines.append(f"    Error: {result.error_message}")
            lines.append("")
        
        # Literature Validation Results
        if self.literature_results:
            lines.append("LITERATURE VALIDATION RESULTS")
            lines.append("-" * 30)
            
            passed = sum(1 for r in self.literature_results if r.validation_passed)
            total = len(self.literature_results)
            
            lines.append(f"Validations Run: {total}")
            lines.append(f"Passed: {passed}")
            lines.append(f"Failed: {total - passed}")
            lines.append(f"Success Rate: {(passed/total)*100:.1f}%")
            lines.append("")
            
            lines.append("Individual Validation Results:")
            for result in self.literature_results:
                status = "✅ PASS" if result.validation_passed else "❌ FAIL"
                lines.append(f"  {result.system_name}: {status}")
                lines.append(f"    Reference: {result.reference.authors} ({result.reference.year})")
                lines.append(f"    Mean Deviation: {result.mean_relative_deviation:.1%}")
                lines.append(f"    P-value: {result.p_value:.3f}")
            lines.append("")
        
        # Cross-Validation Results
        if self.cross_validation_results:
            lines.append("CROSS-VALIDATION RESULTS")
            lines.append("-" * 25)
            
            passed = sum(1 for r in self.cross_validation_results if r.validation_passed)
            total = len(self.cross_validation_results)
            
            lines.append(f"Cross-Validations Run: {total}")
            lines.append(f"Passed: {passed}")
            lines.append(f"Failed: {total - passed}")
            lines.append(f"Success Rate: {(passed/total)*100:.1f}%")
            lines.append("")
            
            # Group by package
            packages = set(r.package_name for r in self.cross_validation_results)
            for package in sorted(packages):
                package_results = [r for r in self.cross_validation_results if r.package_name == package]
                package_passed = sum(1 for r in package_results if r.validation_passed)
                
                lines.append(f"Package: {package}")
                lines.append(f"  Tests: {len(package_results)}")
                lines.append(f"  Passed: {package_passed}")
                lines.append(f"  Success Rate: {(package_passed/len(package_results))*100:.1f}%")
                
                for result in package_results:
                    status = "✅ PASS" if result.validation_passed else "❌ FAIL"
                    lines.append(f"    {result.system_name}: {status}")
                    if result.validation_passed:
                        lines.append(f"      Mean Deviation: {result.mean_relative_difference:.1%}")
                        lines.append(f"      Correlation: {result.correlation_coefficient:.3f}")
            lines.append("")
        
        # Statistical Analysis Summary
        if self.statistical_reports:
            lines.append("STATISTICAL ANALYSIS SUMMARY")
            lines.append("-" * 28)
            
            # Aggregate statistics
            all_correlations = []
            all_effect_sizes = []
            
            for report in self.statistical_reports:
                if 'correlation' in report.effect_size_measures:
                    all_correlations.append(report.effect_size_measures['correlation'])
                if 'cohens_d' in report.effect_size_measures:
                    all_effect_sizes.append(abs(report.effect_size_measures['cohens_d']))
            
            if all_correlations:
                avg_correlation = np.mean(all_correlations)
                lines.append(f"Average Correlation: {avg_correlation:.3f}")
            
            if all_effect_sizes:
                avg_effect_size = np.mean(all_effect_sizes)
                lines.append(f"Average Effect Size (|Cohen's d|): {avg_effect_size:.3f}")
            
            lines.append("")
        
        # Recommendations
        if hasattr(self, '_summary') and self._summary.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 15)
            for i, rec in enumerate(self._summary.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Conclusion
        lines.append("CONCLUSION")
        lines.append("-" * 10)
        
        if hasattr(self, '_summary'):
            summary = self._summary
            if summary.overall_validation_score >= 0.9:
                lines.append("✅ QBES has passed comprehensive validation with excellent results.")
                lines.append("The software is ready for scientific research applications.")
            elif summary.overall_validation_score >= 0.8:
                lines.append("✅ QBES has passed validation with good results.")
                lines.append("Minor improvements recommended but suitable for research use.")
            elif summary.overall_validation_score >= 0.7:
                lines.append("⚠️  QBES has acceptable validation results.")
                lines.append("Some improvements needed before extensive research use.")
            else:
                lines.append("❌ QBES validation results indicate significant issues.")
                lines.append("Major improvements required before research use.")
        
        return "\n".join(lines)
    
    def _save_all_results(self, summary: ValidationSummary):
        """Save all validation results to files."""
        self._summary = summary
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "validation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        # Save comprehensive report
        report_path = os.path.join(self.output_dir, "comprehensive_validation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_comprehensive_report())
        
        # Save individual component results
        if self.benchmark_results:
            benchmark_path = os.path.join(self.output_dir, "benchmark_results.json")
            with open(benchmark_path, 'w') as f:
                json.dump([asdict(r) for r in self.benchmark_results], f, indent=2, default=str)
        
        if self.literature_results:
            literature_path = os.path.join(self.output_dir, "literature_validation_results.json")
            with open(literature_path, 'w') as f:
                json.dump([asdict(r) for r in self.literature_results], f, indent=2, default=str)
        
        if self.cross_validation_results:
            cross_val_path = os.path.join(self.output_dir, "cross_validation_results.json")
            with open(cross_val_path, 'w') as f:
                json.dump([asdict(r) for r in self.cross_validation_results], f, indent=2, default=str)
        
        if self.statistical_reports:
            stats_path = os.path.join(self.output_dir, "statistical_analysis_results.json")
            with open(stats_path, 'w') as f:
                json.dump([asdict(r) for r in self.statistical_reports], f, indent=2, default=str)
        
        print(f"\nAll validation results saved to: {self.output_dir}")
        print(f"  - Summary: validation_summary.json")
        print(f"  - Report: comprehensive_validation_report.txt")
        print(f"  - Individual results: *_results.json")


def run_comprehensive_validation(output_dir: str = "validation_reports") -> ValidationSummary:
    """
    Run comprehensive QBES validation suite.
    
    Args:
        output_dir: Directory to save validation reports
        
    Returns:
        ValidationSummary with overall results
    """
    reporter = ComprehensiveValidationReporter(output_dir)
    
    summary = reporter.run_complete_validation_suite(
        include_benchmarks=True,
        include_literature=True,
        include_cross_validation=True
    )
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"Overall Score: {summary.overall_validation_score:.1%}")
    print(f"Grade: {summary.validation_grade}")
    
    if summary.critical_issues:
        print("\nCritical Issues:")
        for issue in summary.critical_issues:
            print(f"  ⚠️  {issue}")
    
    print(f"\nDetailed reports available in: {output_dir}")
    
    return summary

class ValidationReportGenerator:
    """Simple validation report generator for CLI validate command."""
    
    def generate_markdown_report(self, results: List[Any], output_file: str):
        """Generate markdown validation report."""
        lines = []
        
        # Header
        lines.append("# QBES Validation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**QBES Version:** 1.2.0")
        lines.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if getattr(r, 'test_passed', False))
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Tests:** {total_tests}")
        lines.append(f"- **Passed:** {passed_tests}")
        lines.append(f"- **Failed:** {total_tests - passed_tests}")
        lines.append(f"- **Success Rate:** {success_rate:.1f}%")
        lines.append("")
        
        # Test Results
        lines.append("## Test Results")
        lines.append("")
        
        for result in results:
            test_passed = getattr(result, 'test_passed', False)
            system_name = getattr(result, 'system_name', 'Unknown Test')
            status = "✅ PASS" if test_passed else "❌ FAIL"
            
            lines.append(f"### {system_name}")
            lines.append(f"**Status:** {status}")
            
            # Add specific metrics based on result type
            if hasattr(result, 'relative_error'):
                lines.append(f"**Relative Error:** {result.relative_error:.2e}")
            if hasattr(result, 'computation_time'):
                lines.append(f"**Computation Time:** {result.computation_time:.3f}s")
            if hasattr(result, 'coherence_error'):
                lines.append(f"**Coherence Error:** {result.coherence_error:.1%}")
            if hasattr(result, 'efficiency_error'):
                lines.append(f"**Efficiency Error:** {result.efficiency_error:.1%}")
            if hasattr(result, 'temperature'):
                lines.append(f"**Temperature:** {result.temperature}K")
            if hasattr(result, 'error_message') and result.error_message:
                lines.append(f"**Error:** {result.error_message}")
            
            lines.append("")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def generate_json_report(self, results: List[Any], output_file: str):
        """Generate JSON validation report."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'qbes_version': '1.2.0',
            'summary': {
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if getattr(r, 'test_passed', False)),
                'failed_tests': sum(1 for r in results if not getattr(r, 'test_passed', False)),
                'success_rate': (sum(1 for r in results if getattr(r, 'test_passed', False)) / len(results)) * 100 if results else 0
            },
            'results': []
        }
        
        for result in results:
            result_dict = {}
            
            # Extract common attributes
            for attr in ['system_name', 'test_passed', 'relative_error', 'computation_time',
                        'coherence_error', 'efficiency_error', 'temperature', 'error_message']:
                if hasattr(result, attr):
                    value = getattr(result, attr)
                    if value is not None:
                        result_dict[attr] = value
            
            report_data['results'].append(result_dict)
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def generate_text_report(self, results: List[Any], output_file: str):
        """Generate plain text validation report."""
        lines = []
        
        # Header
        lines.append("QBES VALIDATION REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"QBES Version: 1.2.0")
        lines.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if getattr(r, 'test_passed', False))
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Total Tests:  {total_tests}")
        lines.append(f"Passed:       {passed_tests}")
        lines.append(f"Failed:       {total_tests - passed_tests}")
        lines.append(f"Success Rate: {success_rate:.1f}%")
        lines.append("")
        
        # Test Results
        lines.append("TEST RESULTS")
        lines.append("-" * 20)
        
        for result in results:
            test_passed = getattr(result, 'test_passed', False)
            system_name = getattr(result, 'system_name', 'Unknown Test')
            status = "PASS" if test_passed else "FAIL"
            
            lines.append(f"{system_name}: {status}")
            
            if hasattr(result, 'relative_error'):
                lines.append(f"  Relative Error: {result.relative_error:.2e}")
            if hasattr(result, 'computation_time'):
                lines.append(f"  Computation Time: {result.computation_time:.3f}s")
            if hasattr(result, 'error_message') and result.error_message:
                lines.append(f"  Error: {result.error_message}")
            lines.append("")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))