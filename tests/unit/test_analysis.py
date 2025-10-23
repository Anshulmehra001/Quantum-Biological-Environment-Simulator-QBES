"""
Unit tests for the analysis module.
"""

import pytest
import numpy as np
from qbes.analysis import ResultsAnalyzer, CoherenceAnalyzer, StatisticalAnalyzer
from qbes.core.data_models import DensityMatrix, SimulationResults


class TestResultsAnalyzer:
    """Test suite for ResultsAnalyzer functionality."""
    
    def test_initialization(self):
        """Test ResultsAnalyzer initialization."""
        analyzer = ResultsAnalyzer()
        assert analyzer is not None
        assert analyzer.analysis_cache == {}
    
    def test_calculate_coherence_lifetime_simple(self):
        """Test coherence lifetime calculation with simple trajectory."""
        analyzer = ResultsAnalyzer()
        
        # Create a simple trajectory with exponential decay
        times = np.linspace(0, 1, 11)
        states = []
        
        for t in times:
            # Create density matrix with decaying coherence
            coherence = np.exp(-t / 0.3)  # Lifetime = 0.3
            matrix = np.array([
                [0.5, coherence * 0.1j],
                [-coherence * 0.1j, 0.5]
            ], dtype=complex)
            
            state = DensityMatrix(
                matrix=matrix,
                basis_labels=["ground", "excited"],
                time=t
            )
            states.append(state)
        
        lifetime = analyzer.calculate_coherence_lifetime(states)
        
        # Should be close to 0.3 (the decay constant we used)
        assert 0.2 < lifetime < 0.4
    
    def test_calculate_coherence_lifetime_insufficient_data(self):
        """Test coherence lifetime calculation with insufficient data."""
        analyzer = ResultsAnalyzer()
        
        # Single state
        state = DensityMatrix(
            matrix=np.array([[0.5, 0.1j], [-0.1j, 0.5]], dtype=complex),
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        with pytest.raises(ValueError):
            analyzer.calculate_coherence_lifetime([state])
    
    def test_validate_energy_conservation_good(self):
        """Test energy conservation validation with good data."""
        analyzer = ResultsAnalyzer()
        
        # Create energy trajectory with small fluctuations
        base_energy = -100.0
        energies = [base_energy + 0.001 * np.sin(i * 0.1) for i in range(100)]
        
        result = analyzer.validate_energy_conservation(energies)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_energy_conservation_bad(self):
        """Test energy conservation validation with bad data."""
        analyzer = ResultsAnalyzer()
        
        # Create energy trajectory with large drift
        energies = [i * 0.1 for i in range(100)]  # Linear increase
        
        result = analyzer.validate_energy_conservation(energies)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_validate_probability_conservation(self):
        """Test probability conservation validation."""
        analyzer = ResultsAnalyzer()
        
        # Create valid states
        states = []
        for i in range(5):
            matrix = np.array([[0.6, 0.1j], [-0.1j, 0.4]], dtype=complex)
            state = DensityMatrix(
                matrix=matrix,
                basis_labels=["ground", "excited"],
                time=i * 0.1
            )
            states.append(state)
        
        result = analyzer.validate_probability_conservation(states)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_generate_statistical_summary(self):
        """Test statistical summary generation."""
        analyzer = ResultsAnalyzer()
        
        # Create mock simulation results
        results = SimulationResults(
            state_trajectory=[],
            coherence_measures={"purity": [0.9, 0.8, 0.7, 0.6]},
            energy_trajectory=[-100.0, -99.9, -100.1, -100.0],
            decoherence_rates={"total": 0.1},
            statistical_summary=None,
            simulation_config=None,
            computation_time=10.0
        )
        
        summary = analyzer.generate_statistical_summary(results)
        
        assert summary is not None
        assert "energy" in summary.mean_values
        assert "purity" in summary.mean_values
        assert summary.sample_size > 0
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        analyzer = ResultsAnalyzer()
        
        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
        
        outliers = analyzer.detect_outliers(data, method="iqr")
        
        assert 5 in outliers  # Index of value 100
        assert len(outliers) >= 1
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        analyzer = ResultsAnalyzer()
        
        # Create data with outliers
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
        
        outliers = analyzer.detect_outliers(data, method="zscore")
        
        assert 5 in outliers  # Index of value 100
        assert len(outliers) >= 1
    
    def test_calculate_uncertainty_estimates(self):
        """Test uncertainty estimation."""
        analyzer = ResultsAnalyzer()
        
        # Create sample data
        data = np.random.normal(10.0, 2.0, 100)
        
        estimates = analyzer.calculate_uncertainty_estimates(data)
        
        assert "mean" in estimates
        assert "std_dev" in estimates
        assert "std_error" in estimates
        assert "confidence_interval_95" in estimates
        assert estimates["sample_size"] == 100
        
        # Check that mean is reasonable
        assert 8.0 < estimates["mean"] < 12.0
    
    def test_calculate_purity(self):
        """Test purity calculation."""
        analyzer = ResultsAnalyzer()
        
        # Pure state
        pure_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        pure_state = DensityMatrix(
            matrix=pure_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        purity = analyzer.calculate_purity(pure_state)
        assert np.isclose(purity, 1.0)
        
        # Mixed state
        mixed_matrix = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        mixed_state = DensityMatrix(
            matrix=mixed_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        purity = analyzer.calculate_purity(mixed_state)
        assert np.isclose(purity, 0.5)
    
    def test_calculate_von_neumann_entropy(self):
        """Test von Neumann entropy calculation."""
        analyzer = ResultsAnalyzer()
        
        # Pure state (entropy = 0)
        pure_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        pure_state = DensityMatrix(
            matrix=pure_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        entropy = analyzer.calculate_von_neumann_entropy(pure_state)
        assert np.isclose(entropy, 0.0, atol=1e-10)
        
        # Maximally mixed state (entropy = log(2))
        mixed_matrix = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
        mixed_state = DensityMatrix(
            matrix=mixed_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        entropy = analyzer.calculate_von_neumann_entropy(mixed_state)
        assert np.isclose(entropy, 1.0, rtol=1e-10)  # log2(2) = 1


class TestCoherenceAnalyzer:
    """Test suite for CoherenceAnalyzer functionality."""
    
    def test_initialization(self):
        """Test CoherenceAnalyzer initialization."""
        analyzer = CoherenceAnalyzer()
        assert analyzer is not None
        assert analyzer.analyzer is not None
    
    def test_analyze_decoherence(self):
        """Test decoherence analysis."""
        analyzer = CoherenceAnalyzer()
        
        # Create mock trajectory
        states = []
        for i in range(10):
            coherence = np.exp(-i * 0.1)
            matrix = np.array([
                [0.5, coherence * 0.1j],
                [-coherence * 0.1j, 0.5]
            ], dtype=complex)
            
            state = DensityMatrix(
                matrix=matrix,
                basis_labels=["ground", "excited"],
                time=i * 0.1
            )
            states.append(state)
        
        result = analyzer.analyze_decoherence(states)
        
        assert "decoherence_rate" in result
        assert "coherence_time" in result
        assert "decay_type" in result
        assert result["decoherence_rate"] > 0


class TestStatisticalAnalyzer:
    """Test suite for StatisticalAnalyzer functionality."""
    
    def test_initialization(self):
        """Test StatisticalAnalyzer initialization."""
        analyzer = StatisticalAnalyzer()
        assert analyzer is not None
        assert analyzer.analyzer is not None
    
    def test_calculate_statistics(self):
        """Test basic statistics calculation."""
        analyzer = StatisticalAnalyzer()
        
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats = analyzer.calculate_statistics(data)
        
        assert stats["mean"] == 5.5
        assert stats["median"] == 5.5
        assert stats["min"] == 1
        assert stats["max"] == 10
        assert stats["count"] == 10
    
    def test_calculate_statistics_empty(self):
        """Test statistics calculation with empty data."""
        analyzer = StatisticalAnalyzer()
        
        stats = analyzer.calculate_statistics([])
        
        assert stats["mean"] == 0.0
        assert stats["count"] == 0
    
    def test_generate_confidence_intervals(self):
        """Test confidence interval generation."""
        analyzer = StatisticalAnalyzer()
        
        # Generate sample data
        data = np.random.normal(10.0, 2.0, 100)
        
        ci = analyzer.generate_confidence_intervals(data, confidence_level=0.95)
        
        assert "mean_ci" in ci
        assert "confidence_level" in ci
        assert "sample_size" in ci
        assert ci["confidence_level"] == 0.95
        assert ci["sample_size"] == 100
        
        # Check that confidence interval contains reasonable values
        lower, upper = ci["mean_ci"]
        assert lower < upper
        assert 6.0 < lower < 14.0
        assert 6.0 < upper < 14.0