"""
Unit tests for ValidationSummary backward compatibility.
"""

import pytest
from qbes.benchmarks.validation_reports import ValidationSummary


class TestValidationSummary:
    """Test cases for ValidationSummary backward compatibility."""
    
    def test_legacy_parameter_format(self):
        """Test ValidationSummary with legacy parameter format."""
        summary = ValidationSummary(
            benchmark_score=0.9,
            literature_score=0.8,
            cross_validation_score=0.7,
            statistical_score=0.85
        )
        
        assert summary is not None
        assert hasattr(summary, 'overall_validation_score')
        assert hasattr(summary, 'validation_grade')
        assert summary.overall_validation_score == 0.8125  # (0.9 + 0.8 + 0.7 + 0.85) / 4
        assert summary.validation_grade == "B"  # 0.8125 is in B range (0.8-0.9)
        
        # Check that derived values are set correctly
        assert summary.benchmark_success_rate == 0.9
        assert summary.literature_success_rate == 0.8
        assert summary.cross_validation_success_rate == 0.7
        
        # Check that counts are derived correctly
        assert summary.benchmark_tests_total == 10
        assert summary.benchmark_tests_passed == 9  # 0.9 * 10
        assert summary.literature_validations_total == 10
        assert summary.literature_validations_passed == 8  # 0.8 * 10
    
    def test_new_parameter_format(self):
        """Test ValidationSummary with new parameter format."""
        summary = ValidationSummary(
            benchmark_tests_total=15,
            benchmark_tests_passed=12,
            benchmark_success_rate=0.8,
            literature_validations_total=20,
            literature_validations_passed=18,
            literature_success_rate=0.9,
            overall_validation_score=0.85,
            validation_grade="B"
        )
        
        assert summary is not None
        assert summary.benchmark_tests_total == 15
        assert summary.benchmark_tests_passed == 12
        assert summary.benchmark_success_rate == 0.8
        assert summary.literature_validations_total == 20
        assert summary.literature_validations_passed == 18
        assert summary.literature_success_rate == 0.9
        assert summary.overall_validation_score == 0.85
        assert summary.validation_grade == "B"
    
    def test_default_values(self):
        """Test ValidationSummary with default values."""
        summary = ValidationSummary()
        
        assert summary is not None
        assert summary.benchmark_tests_total == 0
        assert summary.benchmark_tests_passed == 0
        assert summary.benchmark_success_rate == 0.0
        assert summary.overall_validation_score == 0.0
        assert summary.validation_grade == "F"
        assert isinstance(summary.critical_issues, list)
        assert isinstance(summary.recommendations, list)
        assert len(summary.critical_issues) == 0
        assert len(summary.recommendations) == 0
    
    def test_grade_calculation(self):
        """Test that grade calculation works correctly for legacy format."""
        test_cases = [
            (0.95, "A"),
            (0.85, "B"),
            (0.75, "C"),
            (0.65, "D"),
            (0.55, "F")
        ]
        
        for score, expected_grade in test_cases:
            summary = ValidationSummary(
                benchmark_score=score,
                literature_score=score,
                cross_validation_score=score,
                statistical_score=score
            )
            assert summary.validation_grade == expected_grade
            assert summary.overall_validation_score == score
    
    def test_mixed_legacy_parameters(self):
        """Test ValidationSummary with partial legacy parameters."""
        summary = ValidationSummary(
            benchmark_score=0.9,
            literature_score=0.8
            # cross_validation_score and statistical_score will default to 0.0
        )
        
        assert summary is not None
        assert summary.benchmark_success_rate == 0.9
        assert summary.literature_success_rate == 0.8
        assert summary.cross_validation_success_rate == 0.0
        # Overall score should be (0.9 + 0.8 + 0.0 + 0.0) / 4 = 0.425
        assert abs(summary.overall_validation_score - 0.425) < 1e-10
        assert summary.validation_grade == "F"  # 0.425 < 0.6
    
    def test_timestamp_and_version(self):
        """Test that timestamp and version are set correctly."""
        summary = ValidationSummary(benchmark_score=0.8)
        
        assert summary.qbes_version == "1.1.0"
        assert summary.timestamp is not None
        assert len(summary.timestamp) > 0
        
        # Test with explicit timestamp
        custom_timestamp = "2023-01-01T00:00:00"
        summary2 = ValidationSummary(timestamp=custom_timestamp)
        assert summary2.timestamp == custom_timestamp


if __name__ == "__main__":
    pytest.main([__file__])