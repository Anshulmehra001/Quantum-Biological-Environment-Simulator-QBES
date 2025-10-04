"""
Statistical significance testing for benchmark results.

This module provides statistical tests to assess the significance
of differences between QBES results and reference data.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings

try:
    from scipy.stats import kstest, anderson, shapiro, ttest_1samp, ttest_ind, wilcoxon, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical tests will be unavailable.")


@dataclass
class StatisticalTestResult:
    """Results from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    significance_level: float = 0.05
    null_hypothesis_rejected: bool = False
    interpretation: str = ""
    additional_info: Dict[str, Any] = None


@dataclass
class ComprehensiveStatisticalReport:
    """Comprehensive statistical analysis report."""
    sample_size: int
    mean_difference: float
    std_difference: float
    normality_tests: List[StatisticalTestResult]
    significance_tests: List[StatisticalTestResult]
    effect_size_measures: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    power_analysis: Optional[Dict[str, float]] = None
    recommendations: List[str] = None


class StatisticalTester:
    """
    Statistical testing suite for benchmark validation.
    
    Provides various statistical tests to assess the significance
    of differences between simulated and reference results.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            significance_level: Alpha level for statistical tests
        """
        self.significance_level = significance_level
    
    def test_normality(self, data: np.ndarray) -> List[StatisticalTestResult]:
        """
        Test normality of data using multiple tests.
        
        Args:
            data: Data array to test
            
        Returns:
            List of normality test results
        """
        results = []
        
        if not SCIPY_AVAILABLE:
            return [StatisticalTestResult(
                test_name="Normality Test",
                statistic=0.0,
                p_value=1.0,
                interpretation="SciPy not available for normality testing"
            )]
        
        # Shapiro-Wilk test (good for small samples)
        if len(data) >= 3 and len(data) <= 5000:
            try:
                stat, p_val = shapiro(data)
                results.append(StatisticalTestResult(
                    test_name="Shapiro-Wilk",
                    statistic=stat,
                    p_value=p_val,
                    significance_level=self.significance_level,
                    null_hypothesis_rejected=p_val < self.significance_level,
                    interpretation=f"Data {'does not follow' if p_val < self.significance_level else 'follows'} normal distribution"
                ))
            except Exception as e:
                results.append(StatisticalTestResult(
                    test_name="Shapiro-Wilk",
                    statistic=0.0,
                    p_value=1.0,
                    interpretation=f"Test failed: {str(e)}"
                ))
        
        # Anderson-Darling test
        if len(data) >= 8:
            try:
                result = anderson(data, dist='norm')
                # Use 5% significance level critical value
                critical_val = result.critical_values[2] if len(result.critical_values) > 2 else result.critical_values[-1]
                
                results.append(StatisticalTestResult(
                    test_name="Anderson-Darling",
                    statistic=result.statistic,
                    p_value=0.05 if result.statistic > critical_val else 0.1,  # Approximate p-value
                    critical_value=critical_val,
                    significance_level=self.significance_level,
                    null_hypothesis_rejected=result.statistic > critical_val,
                    interpretation=f"Data {'does not follow' if result.statistic > critical_val else 'follows'} normal distribution"
                ))
            except Exception as e:
                results.append(StatisticalTestResult(
                    test_name="Anderson-Darling",
                    statistic=0.0,
                    p_value=1.0,
                    interpretation=f"Test failed: {str(e)}"
                ))
        
        # Kolmogorov-Smirnov test against normal distribution
        if len(data) >= 5:
            try:
                # Standardize data
                standardized = (data - np.mean(data)) / np.std(data)
                stat, p_val = kstest(standardized, 'norm')
                
                results.append(StatisticalTestResult(
                    test_name="Kolmogorov-Smirnov",
                    statistic=stat,
                    p_value=p_val,
                    significance_level=self.significance_level,
                    null_hypothesis_rejected=p_val < self.significance_level,
                    interpretation=f"Data {'does not follow' if p_val < self.significance_level else 'follows'} normal distribution"
                ))
            except Exception as e:
                results.append(StatisticalTestResult(
                    test_name="Kolmogorov-Smirnov",
                    statistic=0.0,
                    p_value=1.0,
                    interpretation=f"Test failed: {str(e)}"
                ))
        
        return results
    
    def test_mean_difference(self, 
                           observed: np.ndarray, 
                           expected: np.ndarray,
                           paired: bool = True) -> List[StatisticalTestResult]:
        """
        Test significance of mean difference between observed and expected values.
        
        Args:
            observed: Observed values
            expected: Expected values
            paired: Whether observations are paired
            
        Returns:
            List of significance test results
        """
        results = []
        
        if not SCIPY_AVAILABLE:
            # Simple z-test approximation
            diff = observed - expected
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            n = len(diff)
            
            if std_diff > 0:
                z_stat = mean_diff / (std_diff / np.sqrt(n))
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0.0
                p_val = 1.0
            
            results.append(StatisticalTestResult(
                test_name="Approximate Z-test",
                statistic=z_stat,
                p_value=p_val,
                significance_level=self.significance_level,
                null_hypothesis_rejected=p_val < self.significance_level,
                interpretation=f"Mean difference {'is' if p_val < self.significance_level else 'is not'} statistically significant"
            ))
            
            return results
        
        if paired and len(observed) == len(expected):
            # Paired tests
            differences = observed - expected
            
            # Paired t-test
            try:
                stat, p_val = ttest_1samp(differences, 0.0)
                results.append(StatisticalTestResult(
                    test_name="Paired t-test",
                    statistic=stat,
                    p_value=p_val,
                    significance_level=self.significance_level,
                    null_hypothesis_rejected=p_val < self.significance_level,
                    interpretation=f"Mean difference {'is' if p_val < self.significance_level else 'is not'} statistically significant"
                ))
            except Exception as e:
                results.append(StatisticalTestResult(
                    test_name="Paired t-test",
                    statistic=0.0,
                    p_value=1.0,
                    interpretation=f"Test failed: {str(e)}"
                ))
            
            # Wilcoxon signed-rank test (non-parametric)
            if len(differences) >= 6:
                try:
                    stat, p_val = wilcoxon(differences)
                    results.append(StatisticalTestResult(
                        test_name="Wilcoxon signed-rank",
                        statistic=stat,
                        p_value=p_val,
                        significance_level=self.significance_level,
                        null_hypothesis_rejected=p_val < self.significance_level,
                        interpretation=f"Median difference {'is' if p_val < self.significance_level else 'is not'} statistically significant"
                    ))
                except Exception as e:
                    results.append(StatisticalTestResult(
                        test_name="Wilcoxon signed-rank",
                        statistic=0.0,
                        p_value=1.0,
                        interpretation=f"Test failed: {str(e)}"
                    ))
        
        else:
            # Independent samples tests
            try:
                stat, p_val = ttest_ind(observed, expected)
                results.append(StatisticalTestResult(
                    test_name="Independent t-test",
                    statistic=stat,
                    p_value=p_val,
                    significance_level=self.significance_level,
                    null_hypothesis_rejected=p_val < self.significance_level,
                    interpretation=f"Mean difference {'is' if p_val < self.significance_level else 'is not'} statistically significant"
                ))
            except Exception as e:
                results.append(StatisticalTestResult(
                    test_name="Independent t-test",
                    statistic=0.0,
                    p_value=1.0,
                    interpretation=f"Test failed: {str(e)}"
                ))
            
            # Mann-Whitney U test (non-parametric)
            if len(observed) >= 3 and len(expected) >= 3:
                try:
                    stat, p_val = mannwhitneyu(observed, expected, alternative='two-sided')
                    results.append(StatisticalTestResult(
                        test_name="Mann-Whitney U",
                        statistic=stat,
                        p_value=p_val,
                        significance_level=self.significance_level,
                        null_hypothesis_rejected=p_val < self.significance_level,
                        interpretation=f"Distribution difference {'is' if p_val < self.significance_level else 'is not'} statistically significant"
                    ))
                except Exception as e:
                    results.append(StatisticalTestResult(
                        test_name="Mann-Whitney U",
                        statistic=0.0,
                        p_value=1.0,
                        interpretation=f"Test failed: {str(e)}"
                    ))
        
        return results
    
    def calculate_effect_sizes(self, 
                              observed: np.ndarray, 
                              expected: np.ndarray) -> Dict[str, float]:
        """
        Calculate various effect size measures.
        
        Args:
            observed: Observed values
            expected: Expected values
            
        Returns:
            Dictionary of effect size measures
        """
        effect_sizes = {}
        
        # Cohen's d (standardized mean difference)
        if len(observed) == len(expected):
            differences = observed - expected
            pooled_std = np.sqrt((np.var(observed, ddof=1) + np.var(expected, ddof=1)) / 2)
            
            if pooled_std > 0:
                cohens_d = np.mean(differences) / pooled_std
                effect_sizes['cohens_d'] = cohens_d
            else:
                effect_sizes['cohens_d'] = 0.0
        
        # Pearson correlation coefficient
        if len(observed) == len(expected) and len(observed) > 1:
            try:
                correlation, _ = stats.pearsonr(observed, expected)
                effect_sizes['correlation'] = correlation
            except:
                effect_sizes['correlation'] = 0.0
        
        # Mean absolute percentage error (MAPE)
        if len(observed) == len(expected):
            mape = np.mean(np.abs((observed - expected) / (expected + 1e-12))) * 100
            effect_sizes['mape'] = mape
        
        # Root mean square error (RMSE)
        if len(observed) == len(expected):
            rmse = np.sqrt(np.mean((observed - expected) ** 2))
            effect_sizes['rmse'] = rmse
        
        # Normalized RMSE
        if len(observed) == len(expected):
            expected_range = np.max(expected) - np.min(expected)
            if expected_range > 0:
                nrmse = rmse / expected_range
                effect_sizes['nrmse'] = nrmse
        
        return effect_sizes
    
    def calculate_confidence_intervals(self, 
                                     observed: np.ndarray, 
                                     expected: np.ndarray,
                                     confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for various statistics.
        
        Args:
            observed: Observed values
            expected: Expected values
            confidence_level: Confidence level (default 0.95)
            
        Returns:
            Dictionary of confidence intervals
        """
        intervals = {}
        alpha = 1 - confidence_level
        
        if len(observed) == len(expected):
            differences = observed - expected
            n = len(differences)
            
            if n > 1:
                # Confidence interval for mean difference
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                
                if SCIPY_AVAILABLE:
                    t_critical = stats.t.ppf(1 - alpha/2, n - 1)
                else:
                    # Approximate with normal distribution for large samples
                    t_critical = stats.norm.ppf(1 - alpha/2) if n >= 30 else 2.0
                
                margin_error = t_critical * std_diff / np.sqrt(n)
                intervals['mean_difference'] = (mean_diff - margin_error, mean_diff + margin_error)
                
                # Confidence interval for correlation (if applicable)
                if len(observed) > 2:
                    try:
                        correlation, _ = stats.pearsonr(observed, expected)
                        # Fisher's z-transformation
                        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
                        z_se = 1 / np.sqrt(n - 3)
                        
                        if SCIPY_AVAILABLE:
                            z_critical = stats.norm.ppf(1 - alpha/2)
                        else:
                            z_critical = 1.96
                        
                        z_lower = z - z_critical * z_se
                        z_upper = z + z_critical * z_se
                        
                        # Transform back to correlation scale
                        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                        
                        intervals['correlation'] = (r_lower, r_upper)
                    except:
                        pass
        
        return intervals
    
    def perform_power_analysis(self, 
                              observed: np.ndarray, 
                              expected: np.ndarray,
                              effect_size: Optional[float] = None) -> Dict[str, float]:
        """
        Perform statistical power analysis.
        
        Args:
            observed: Observed values
            expected: Expected values
            effect_size: Expected effect size (if None, calculated from data)
            
        Returns:
            Dictionary with power analysis results
        """
        power_results = {}
        
        if len(observed) != len(expected):
            return power_results
        
        n = len(observed)
        differences = observed - expected
        
        # Calculate effect size if not provided
        if effect_size is None:
            pooled_std = np.sqrt((np.var(observed, ddof=1) + np.var(expected, ddof=1)) / 2)
            if pooled_std > 0:
                effect_size = abs(np.mean(differences)) / pooled_std
            else:
                effect_size = 0.0
        
        power_results['effect_size'] = effect_size
        power_results['sample_size'] = n
        
        # Approximate power calculation for t-test
        if SCIPY_AVAILABLE and n > 1:
            try:
                # Non-centrality parameter
                ncp = effect_size * np.sqrt(n)
                
                # Critical t-value
                t_critical = stats.t.ppf(1 - self.significance_level/2, n - 1)
                
                # Power calculation (approximate)
                power = 1 - stats.t.cdf(t_critical, n - 1, ncp) + stats.t.cdf(-t_critical, n - 1, ncp)
                power_results['statistical_power'] = power
                
                # Sample size for desired power (80%)
                desired_power = 0.8
                if effect_size > 0:
                    # Approximate sample size calculation
                    z_alpha = stats.norm.ppf(1 - self.significance_level/2)
                    z_beta = stats.norm.ppf(desired_power)
                    required_n = ((z_alpha + z_beta) / effect_size) ** 2
                    power_results['required_sample_size_80_power'] = int(np.ceil(required_n))
                
            except Exception as e:
                power_results['error'] = str(e)
        
        return power_results
    
    def comprehensive_analysis(self, 
                              observed: np.ndarray, 
                              expected: np.ndarray,
                              paired: bool = True) -> ComprehensiveStatisticalReport:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            observed: Observed values
            expected: Expected values
            paired: Whether observations are paired
            
        Returns:
            ComprehensiveStatisticalReport with all analyses
        """
        if len(observed) != len(expected) and paired:
            raise ValueError("Observed and expected arrays must have same length for paired analysis")
        
        differences = observed - expected if paired else observed
        
        # Basic statistics
        sample_size = len(observed)
        mean_difference = np.mean(differences)
        std_difference = np.std(differences, ddof=1) if len(differences) > 1 else 0.0
        
        # Normality tests
        normality_tests = self.test_normality(differences)
        
        # Significance tests
        significance_tests = self.test_mean_difference(observed, expected, paired)
        
        # Effect sizes
        effect_sizes = self.calculate_effect_sizes(observed, expected)
        
        # Confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(observed, expected)
        
        # Power analysis
        power_analysis = self.perform_power_analysis(observed, expected)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            sample_size, mean_difference, std_difference,
            normality_tests, significance_tests, effect_sizes, power_analysis
        )
        
        return ComprehensiveStatisticalReport(
            sample_size=sample_size,
            mean_difference=mean_difference,
            std_difference=std_difference,
            normality_tests=normality_tests,
            significance_tests=significance_tests,
            effect_size_measures=effect_sizes,
            confidence_intervals=confidence_intervals,
            power_analysis=power_analysis,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self,
                                sample_size: int,
                                mean_difference: float,
                                std_difference: float,
                                normality_tests: List[StatisticalTestResult],
                                significance_tests: List[StatisticalTestResult],
                                effect_sizes: Dict[str, float],
                                power_analysis: Dict[str, float]) -> List[str]:
        """Generate statistical recommendations based on analysis results."""
        recommendations = []
        
        # Sample size recommendations
        if sample_size < 10:
            recommendations.append("Small sample size (n<10). Results should be interpreted with caution.")
        elif sample_size < 30:
            recommendations.append("Moderate sample size. Consider non-parametric tests if normality is violated.")
        
        # Normality recommendations
        normality_violated = any(test.null_hypothesis_rejected for test in normality_tests)
        if normality_violated:
            recommendations.append("Data may not be normally distributed. Consider non-parametric tests.")
        
        # Significance recommendations
        significant_results = [test for test in significance_tests if test.null_hypothesis_rejected]
        if significant_results:
            recommendations.append("Statistically significant differences detected. Examine practical significance.")
        else:
            recommendations.append("No statistically significant differences found.")
        
        # Effect size recommendations
        if 'cohens_d' in effect_sizes:
            d = abs(effect_sizes['cohens_d'])
            if d < 0.2:
                recommendations.append("Small effect size (Cohen's d < 0.2). Differences may not be practically significant.")
            elif d < 0.5:
                recommendations.append("Medium effect size (Cohen's d ≈ 0.2-0.5). Moderate practical significance.")
            elif d < 0.8:
                recommendations.append("Large effect size (Cohen's d ≈ 0.5-0.8). High practical significance.")
            else:
                recommendations.append("Very large effect size (Cohen's d > 0.8). Very high practical significance.")
        
        # Power recommendations
        if 'statistical_power' in power_analysis:
            power = power_analysis['statistical_power']
            if power < 0.8:
                recommendations.append(f"Low statistical power ({power:.2f}). Consider increasing sample size.")
            else:
                recommendations.append(f"Adequate statistical power ({power:.2f}).")
        
        # Correlation recommendations
        if 'correlation' in effect_sizes:
            r = effect_sizes['correlation']
            if abs(r) > 0.9:
                recommendations.append("Very high correlation. Results are highly consistent.")
            elif abs(r) > 0.7:
                recommendations.append("High correlation. Results are reasonably consistent.")
            elif abs(r) > 0.5:
                recommendations.append("Moderate correlation. Some consistency in results.")
            else:
                recommendations.append("Low correlation. Results show poor consistency.")
        
        return recommendations
    
    def generate_statistical_report(self, report: ComprehensiveStatisticalReport) -> str:
        """Generate a formatted statistical analysis report."""
        lines = []
        lines.append("=" * 60)
        lines.append("Statistical Analysis Report")
        lines.append("=" * 60)
        
        # Basic statistics
        lines.append(f"Sample Size: {report.sample_size}")
        lines.append(f"Mean Difference: {report.mean_difference:.6f}")
        lines.append(f"Standard Deviation of Differences: {report.std_difference:.6f}")
        lines.append("")
        
        # Normality tests
        lines.append("Normality Tests:")
        lines.append("-" * 16)
        for test in report.normality_tests:
            status = "REJECTED" if test.null_hypothesis_rejected else "NOT REJECTED"
            lines.append(f"{test.test_name}: statistic={test.statistic:.4f}, p={test.p_value:.4f} ({status})")
        lines.append("")
        
        # Significance tests
        lines.append("Significance Tests:")
        lines.append("-" * 18)
        for test in report.significance_tests:
            status = "SIGNIFICANT" if test.null_hypothesis_rejected else "NOT SIGNIFICANT"
            lines.append(f"{test.test_name}: statistic={test.statistic:.4f}, p={test.p_value:.4f} ({status})")
        lines.append("")
        
        # Effect sizes
        lines.append("Effect Size Measures:")
        lines.append("-" * 20)
        for measure, value in report.effect_size_measures.items():
            lines.append(f"{measure}: {value:.4f}")
        lines.append("")
        
        # Confidence intervals
        if report.confidence_intervals:
            lines.append("95% Confidence Intervals:")
            lines.append("-" * 25)
            for measure, (lower, upper) in report.confidence_intervals.items():
                lines.append(f"{measure}: [{lower:.4f}, {upper:.4f}]")
            lines.append("")
        
        # Power analysis
        if report.power_analysis:
            lines.append("Power Analysis:")
            lines.append("-" * 15)
            for measure, value in report.power_analysis.items():
                if isinstance(value, (int, float)):
                    lines.append(f"{measure}: {value:.4f}")
                else:
                    lines.append(f"{measure}: {value}")
            lines.append("")
        
        # Recommendations
        if report.recommendations:
            lines.append("Recommendations:")
            lines.append("-" * 15)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        return "\n".join(lines)


def perform_statistical_validation(observed_values: List[float],
                                 expected_values: List[float],
                                 significance_level: float = 0.05) -> ComprehensiveStatisticalReport:
    """
    Perform comprehensive statistical validation of benchmark results.
    
    Args:
        observed_values: Observed (simulated) values
        expected_values: Expected (reference) values
        significance_level: Alpha level for statistical tests
        
    Returns:
        ComprehensiveStatisticalReport with full analysis
    """
    tester = StatisticalTester(significance_level)
    
    observed = np.array(observed_values)
    expected = np.array(expected_values)
    
    return tester.comprehensive_analysis(observed, expected, paired=True)