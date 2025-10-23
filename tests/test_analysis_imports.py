"""
Unit tests for analysis module imports and basic functionality.
"""

import pytest
import numpy as np
from qbes.analysis import CoherenceAnalyzer, StatisticalAnalyzer


class TestAnalysisImports:
    """Test cases for analysis module imports and basic functionality."""
    
    def test_coherence_analyzer_import(self):
        """Test that CoherenceAnalyzer can be imported and instantiated."""
        analyzer = CoherenceAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'calculate_coherence_lifetime')
        assert hasattr(analyzer, 'analyze_decoherence')
    
    def test_statistical_analyzer_import(self):
        """Test that StatisticalAnalyzer can be imported and instantiated."""
        analyzer = StatisticalAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'calculate_statistics')
        assert hasattr(analyzer, 'generate_confidence_intervals')
    
    def test_coherence_analyzer_analyze_decoherence(self):
        """Test CoherenceAnalyzer analyze_decoherence method."""
        analyzer = CoherenceAnalyzer()
        
        # Test with empty trajectory
        result = analyzer.analyze_decoherence([])
        assert isinstance(result, dict)
        assert "decoherence_rate" in result
        assert "coherence_time" in result
        assert result["decoherence_rate"] == 0.0
        assert result["coherence_time"] == float('inf')
    
    def test_statistical_analyzer_calculate_statistics(self):
        """Test StatisticalAnalyzer calculate_statistics method."""
        analyzer = StatisticalAnalyzer()
        
        # Test with sample data
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.calculate_statistics(data)
        
        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "variance" in result
        assert "count" in result
        
        assert result["mean"] == 3.0
        assert result["count"] == 5
        
        # Test with empty data
        empty_result = analyzer.calculate_statistics([])
        assert empty_result["mean"] == 0.0
        assert empty_result["count"] == 0
    
    def test_statistical_analyzer_confidence_intervals(self):
        """Test StatisticalAnalyzer generate_confidence_intervals method."""
        analyzer = StatisticalAnalyzer()
        
        # Test with sample data
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.generate_confidence_intervals(data)
        
        assert isinstance(result, dict)
        assert "mean_ci" in result
        assert "confidence_level" in result
        assert "sample_size" in result
        
        assert result["confidence_level"] == 0.95
        assert result["sample_size"] == 5
        assert isinstance(result["mean_ci"], tuple)
        assert len(result["mean_ci"]) == 2
        
        # Test with empty data
        empty_result = analyzer.generate_confidence_intervals([])
        assert empty_result["mean_ci"] == (0.0, 0.0)
        assert empty_result["confidence_level"] == 0.95
    
    def test_all_required_methods_exist(self):
        """Test that all required methods exist as expected by the test suite."""
        # Test CoherenceAnalyzer
        ca = CoherenceAnalyzer()
        assert hasattr(ca, 'calculate_coherence_lifetime')
        assert hasattr(ca, 'analyze_decoherence')
        
        # Test StatisticalAnalyzer
        sa = StatisticalAnalyzer()
        assert hasattr(sa, 'calculate_statistics')
        assert hasattr(sa, 'generate_confidence_intervals')


if __name__ == "__main__":
    pytest.main([__file__])