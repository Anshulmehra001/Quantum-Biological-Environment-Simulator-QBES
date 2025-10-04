"""
Tests for statistical analysis tools.
"""

import pytest
import numpy as np
from qbes.analysis import ResultsAnalyzer
from qbes.core.data_models import (
    DensityMatrix, SimulationResults, StatisticalSummary, SimulationConfig
)


class TestStatisticalAnalysis:
    """Test statistical analysis methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ResultsAnalyzer()
        self.basis_2 = ['|0>', '|1>']
        
        # Create dummy simulation config
        self.config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="all",
            noise_model_type="ohmic",
            output_directory="test_output"
        )
    
    def create_density_matrix(self, matrix: np.ndarray, time: float = 0.0) -> DensityMatrix:
        """Helper to create density matrix."""
        return DensityMatrix(matrix=matrix, basis_labels=self.basis_2, time=time)
    
    def create_test_results(self, n_points: int = 10) -> SimulationResults:
        """Create test simulation results."""
        # Create trajectory with gradual decoherence
        trajectory = []
        energies = []
        
        for i in range(n_points):
            t = float(i)
            coherence = 0.5 * np.exp(-0.1 * t)
            rho = np.array([[0.5, coherence], [coherence, 0.5]])
            trajectory.append(self.create_density_matrix(rho, t))
            energies.append(1.0 + 0.01 * np.sin(t))  # Small oscillations
        
        return SimulationResults(
            state_trajectory=trajectory,
            coherence_measures={'coherence_lifetime': [10.0, 9.5, 10.5]},
            energy_trajectory=energies,
            decoherence_rates={'dephasing': 0.1, 'relaxation': 0.05},
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=n_points
            ),
            simulation_config=self.config
        )
    
    def test_generate_statistical_summary(self):
        """Test statistical summary generation."""
        results = self.create_test_results(10)
        
        summary = self.analyzer.generate_statistical_summary(results)
        
        # Check that summary contains expected keys
        assert 'energy' in summary.mean_values
        assert 'purity' in summary.mean_values
        assert 'entropy' in summary.mean_values
        assert 'coherence_lifetime' in summary.mean_values
        assert 'dephasing_rate' in summary.mean_values
        
        # Check that statistics are reasonable
        assert summary.sample_size == 10
        assert summary.mean_values['energy'] > 0
        assert 0 <= summary.mean_values['purity'] <= 1
        assert summary.mean_values['entropy'] >= 0
        
        # Check confidence intervals
        assert 'energy' in summary.confidence_intervals
        ci_lower, ci_upper = summary.confidence_intervals['energy']
        assert ci_lower <= summary.mean_values['energy'] <= ci_upper
    
    def test_generate_statistical_summary_single_point(self):
        """Test statistical summary with single data point."""
        results = self.create_test_results(1)
        
        summary = self.analyzer.generate_statistical_summary(results)
        
        # Should handle single point gracefully
        assert summary.sample_size == 1
        assert summary.std_deviations['energy'] == 0.0
        
        # Confidence intervals should be point estimates
        ci_lower, ci_upper = summary.confidence_intervals['energy']
        assert ci_lower == ci_upper == summary.mean_values['energy']
    
    def test_generate_statistical_summary_empty_data(self):
        """Test statistical summary with empty data."""
        # Create minimal results with matching trajectory lengths
        trajectory = [self.create_density_matrix(np.eye(2)/2, 0.0)]
        
        results = SimulationResults(
            state_trajectory=trajectory,
            coherence_measures={},  # Empty
            energy_trajectory=[1.0],   # Match trajectory length
            decoherence_rates={},   # Empty
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=1
            ),
            simulation_config=self.config
        )
        
        summary = self.analyzer.generate_statistical_summary(results)
        
        # Should still calculate state-based quantities
        assert 'purity' in summary.mean_values
        assert 'entropy' in summary.mean_values
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        # Create data with known outliers
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, -50])  # 100 and -50 are outliers
        
        outliers = self.analyzer.detect_outliers(data, method='iqr')
        
        # Should detect the extreme values
        assert len(outliers) >= 2
        assert 5 in outliers  # Index of 100
        assert 10 in outliers  # Index of -50
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        # Create normal data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        data = np.concatenate([normal_data, [10, -10]])  # Add extreme outliers
        
        outliers = self.analyzer.detect_outliers(data, method='zscore')
        
        # Should detect the extreme values
        assert len(outliers) >= 2
        # The outliers should be at the end of the array
        assert 100 in outliers or 101 in outliers
    
    def test_detect_outliers_modified_zscore(self):
        """Test outlier detection using modified Z-score method."""
        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        
        outliers = self.analyzer.detect_outliers(data, method='modified_zscore')
        
        # Should detect the extreme value
        assert 9 in outliers  # Index of 100
    
    def test_detect_outliers_edge_cases(self):
        """Test outlier detection edge cases."""
        # Empty data
        outliers = self.analyzer.detect_outliers(np.array([]))
        assert len(outliers) == 0
        
        # Single point
        outliers = self.analyzer.detect_outliers(np.array([1.0]))
        assert len(outliers) == 0
        
        # Two points
        outliers = self.analyzer.detect_outliers(np.array([1.0, 2.0]))
        assert len(outliers) == 0
        
        # Constant data
        outliers = self.analyzer.detect_outliers(np.array([5.0, 5.0, 5.0, 5.0]))
        assert len(outliers) == 0
    
    def test_detect_outliers_invalid_method(self):
        """Test outlier detection with invalid method."""
        data = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError):
            self.analyzer.detect_outliers(data, method='invalid_method')
    
    def test_calculate_uncertainty_estimates(self):
        """Test uncertainty estimation."""
        # Create sample data
        np.random.seed(42)
        data = np.random.normal(10.0, 2.0, 50)
        
        uncertainties = self.analyzer.calculate_uncertainty_estimates(data)
        
        # Check that all expected keys are present
        expected_keys = ['mean', 'std_dev', 'std_error', 'confidence_interval_95', 
                        'bootstrap_ci_95', 'relative_uncertainty', 'sample_size']
        for key in expected_keys:
            assert key in uncertainties
        
        # Check that values are reasonable
        assert abs(uncertainties['mean'] - 10.0) < 1.0  # Should be close to true mean
        assert uncertainties['std_dev'] > 0
        assert uncertainties['std_error'] > 0
        assert uncertainties['std_error'] < uncertainties['std_dev']  # Standard error < standard deviation
        assert uncertainties['sample_size'] == 50
        
        # Check confidence interval
        ci_lower, ci_upper = uncertainties['confidence_interval_95']
        assert ci_lower < uncertainties['mean'] < ci_upper
        
        # Check bootstrap confidence interval
        boot_ci_lower, boot_ci_upper = uncertainties['bootstrap_ci_95']
        assert boot_ci_lower < uncertainties['mean'] < boot_ci_upper
    
    def test_calculate_uncertainty_estimates_single_point(self):
        """Test uncertainty estimation with single data point."""
        data = np.array([5.0])
        
        uncertainties = self.analyzer.calculate_uncertainty_estimates(data)
        
        assert uncertainties['mean'] == 5.0
        assert uncertainties['std_error'] == 0.0
        assert uncertainties['confidence_interval_95'] == (5.0, 5.0)
        assert uncertainties['relative_uncertainty'] == 0.0
    
    def test_calculate_uncertainty_estimates_empty_data(self):
        """Test uncertainty estimation with empty data."""
        data = np.array([])
        
        uncertainties = self.analyzer.calculate_uncertainty_estimates(data)
        
        assert len(uncertainties) == 0
    
    def test_assess_data_quality(self):
        """Test data quality assessment."""
        results = self.create_test_results(20)
        
        quality = self.analyzer.assess_data_quality(results)
        
        # Check basic quality metrics
        assert quality['has_state_trajectory'] == True
        assert quality['has_energy_trajectory'] == True
        assert quality['trajectory_length'] == 20
        assert quality['energy_has_nan'] == False
        assert quality['energy_has_inf'] == False
        
        # Check outlier detection
        assert 'purity_outliers' in quality
        assert 'energy_outliers' in quality
        assert quality['purity_outlier_fraction'] >= 0
        assert quality['energy_outlier_fraction'] >= 0
        
        # Check temporal consistency
        assert 'uniform_time_steps' in quality
        assert 'time_step_variation' in quality
        
        # Check overall quality score
        assert 0 <= quality['overall_quality_score'] <= 1
    
    def test_assess_data_quality_poor_data(self):
        """Test data quality assessment with poor quality data."""
        # Create results with problematic data - need matching trajectory lengths
        trajectory = [
            self.create_density_matrix(np.eye(2)/2, 0.0),
            self.create_density_matrix(np.eye(2)/2, 1.0),
            self.create_density_matrix(np.eye(2)/2, 2.0)
        ]
        
        results = SimulationResults(
            state_trajectory=trajectory,
            coherence_measures={},
            energy_trajectory=[np.nan, np.inf, 1.0],  # Contains NaN and Inf
            decoherence_rates={},
            statistical_summary=StatisticalSummary(
                mean_values={}, std_deviations={}, 
                confidence_intervals={}, sample_size=3
            ),
            simulation_config=self.config
        )
        
        quality = self.analyzer.assess_data_quality(results)
        
        # Should detect problems
        assert quality['energy_has_nan'] == True
        assert quality['energy_has_inf'] == True
        assert quality['overall_quality_score'] < 1.0
    
    def test_perform_trend_analysis(self):
        """Test trend analysis."""
        # Create data with known trend
        times = np.linspace(0, 10, 50)
        trend_slope = 0.5
        noise = np.random.normal(0, 0.1, 50)
        data = trend_slope * times + noise
        
        trend_analysis = self.analyzer.perform_trend_analysis(data, times)
        
        # Check trend detection
        assert 'trend_slope' in trend_analysis
        assert 'trend_r_squared' in trend_analysis
        assert 'trend_significance' in trend_analysis
        assert 'is_stationary' in trend_analysis
        
        # Should detect the trend
        assert abs(trend_analysis['trend_slope'] - trend_slope) < 0.1
        assert trend_analysis['trend_r_squared'] > 0.8  # Should have good fit
        assert trend_analysis['is_stationary'] == False  # Has trend
    
    def test_perform_trend_analysis_stationary(self):
        """Test trend analysis with stationary data."""
        # Create stationary data (no trend)
        np.random.seed(42)
        data = np.random.normal(5.0, 1.0, 50)
        
        trend_analysis = self.analyzer.perform_trend_analysis(data)
        
        # Should detect no significant trend
        assert abs(trend_analysis['trend_slope']) < 0.1
        assert trend_analysis['is_stationary'] == True
    
    def test_perform_trend_analysis_edge_cases(self):
        """Test trend analysis edge cases."""
        # Too few points
        data = np.array([1.0, 2.0])
        trend_analysis = self.analyzer.perform_trend_analysis(data)
        
        assert trend_analysis['trend_slope'] == 0.0
        assert trend_analysis['trend_significance'] == 0.0
        
        # Single point
        data = np.array([1.0])
        trend_analysis = self.analyzer.perform_trend_analysis(data)
        
        assert trend_analysis['trend_slope'] == 0.0
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation helper."""
        # Test with known data
        data = np.array([1, 2, 3, 4, 5])
        
        ci = self.analyzer._calculate_confidence_interval(data, 0.95)
        
        # Should be a tuple
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        
        # Lower bound should be less than upper bound
        assert ci[0] <= ci[1]
        
        # Mean should be within interval
        mean = np.mean(data)
        assert ci[0] <= mean <= ci[1]
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        np.random.seed(42)  # For reproducibility
        data = np.random.normal(10, 2, 100)
        
        boot_ci = self.analyzer._bootstrap_confidence_interval(data, 0.95, 1000)
        
        # Should be a tuple
        assert isinstance(boot_ci, tuple)
        assert len(boot_ci) == 2
        
        # Lower bound should be less than upper bound
        assert boot_ci[0] <= boot_ci[1]
        
        # Should be reasonably close to true mean
        assert abs((boot_ci[0] + boot_ci[1]) / 2 - 10) < 1.0


class TestStatisticalMethodAccuracy:
    """Test accuracy of statistical methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ResultsAnalyzer()
    
    def test_outlier_detection_accuracy(self):
        """Test accuracy of outlier detection methods."""
        # Create data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers_true = [50, 75]  # Indices where we'll add outliers
        
        data = normal_data.copy()
        data[outliers_true[0]] = 10  # Strong outlier
        data[outliers_true[1]] = -8  # Strong outlier
        
        # Test different methods
        for method in ['iqr', 'zscore', 'modified_zscore']:
            detected = self.analyzer.detect_outliers(data, method=method)
            
            # Should detect at least one of the true outliers
            overlap = set(detected) & set(outliers_true)
            assert len(overlap) >= 1, f"Method {method} failed to detect outliers"
    
    def test_uncertainty_estimation_accuracy(self):
        """Test accuracy of uncertainty estimation."""
        # Test with known distribution
        np.random.seed(42)
        true_mean = 5.0
        true_std = 2.0
        data = np.random.normal(true_mean, true_std, 1000)
        
        uncertainties = self.analyzer.calculate_uncertainty_estimates(data)
        
        # Mean should be close to true mean
        assert abs(uncertainties['mean'] - true_mean) < 0.2
        
        # Standard deviation should be close to true std
        assert abs(uncertainties['std_dev'] - true_std) < 0.2
        
        # Confidence interval should contain true mean
        ci_lower, ci_upper = uncertainties['confidence_interval_95']
        assert ci_lower <= true_mean <= ci_upper
    
    def test_trend_analysis_accuracy(self):
        """Test accuracy of trend analysis."""
        # Create data with known linear trend
        times = np.linspace(0, 10, 100)
        true_slope = 1.5
        true_intercept = 2.0
        noise_std = 0.5
        
        np.random.seed(42)
        noise = np.random.normal(0, noise_std, 100)
        data = true_intercept + true_slope * times + noise
        
        trend_analysis = self.analyzer.perform_trend_analysis(data, times)
        
        # Should accurately detect slope
        assert abs(trend_analysis['trend_slope'] - true_slope) < 0.1
        
        # Should have high R-squared due to strong trend
        assert trend_analysis['trend_r_squared'] > 0.8
        
        # Should detect non-stationarity
        assert trend_analysis['is_stationary'] == False


if __name__ == "__main__":
    pytest.main([__file__])