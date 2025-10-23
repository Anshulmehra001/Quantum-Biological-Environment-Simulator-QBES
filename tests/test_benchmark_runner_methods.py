"""
Tests for BenchmarkRunner methods added in task 1.6.
"""

import pytest
from unittest.mock import Mock, patch
from qbes.benchmarks.benchmark_systems import BenchmarkRunner, BenchmarkResult, TwoLevelSystemBenchmark


class TestBenchmarkRunnerMethods:
    """Test BenchmarkRunner methods for task 1.6."""
    
    def test_run_benchmarks_method_exists(self):
        """Test that run_benchmarks method exists."""
        runner = BenchmarkRunner()
        assert hasattr(runner, 'run_benchmarks')
        assert callable(getattr(runner, 'run_benchmarks'))
    
    def test_run_benchmarks_alias_functionality(self):
        """Test that run_benchmarks works as alias for run_all_benchmarks."""
        runner = BenchmarkRunner()
        
        # Mock the run_all_benchmarks method
        mock_result = [BenchmarkResult(
            system_name="Test",
            test_passed=True,
            numerical_result=1.0,
            analytical_result=1.0,
            relative_error=0.0,
            absolute_error=0.0,
            computation_time=0.1,
            tolerance=1e-6
        )]
        
        with patch.object(runner, 'run_all_benchmarks', return_value=mock_result) as mock_run_all:
            result = runner.run_benchmarks(final_time=0.5, time_step=0.01)
            
            # Verify the alias calls the original method with correct parameters
            mock_run_all.assert_called_once_with(0.5, 0.01)
            assert result == mock_result
    
    def test_run_benchmarks_with_default_parameters(self):
        """Test run_benchmarks with default parameters."""
        runner = BenchmarkRunner()
        
        with patch.object(runner, 'run_all_benchmarks', return_value=[]) as mock_run_all:
            runner.run_benchmarks()
            
            # Verify default parameters are passed correctly
            mock_run_all.assert_called_once_with(1.0, 0.01)
    
    def test_add_benchmark_method_works(self):
        """Test that add_benchmark method works correctly."""
        runner = BenchmarkRunner()
        benchmark = TwoLevelSystemBenchmark()
        
        # Initially no benchmarks
        assert len(runner.benchmarks) == 0
        
        # Add benchmark
        runner.add_benchmark(benchmark)
        
        # Verify benchmark was added
        assert len(runner.benchmarks) == 1
        assert runner.benchmarks[0] == benchmark
    
    def test_benchmark_runner_error_handling(self):
        """Test error handling in benchmark execution."""
        runner = BenchmarkRunner()
        
        # Create a mock benchmark that raises an exception
        mock_benchmark = Mock()
        mock_benchmark.name = "Failing Benchmark"
        mock_benchmark.tolerance = 1e-6
        mock_benchmark.run_benchmark.side_effect = Exception("Test error")
        
        runner.add_benchmark(mock_benchmark)
        
        # Run benchmarks and verify error handling
        results = runner.run_benchmarks()
        
        assert len(results) == 1
        result = results[0]
        assert not result.test_passed
        assert result.system_name == "Failing Benchmark"
        assert result.error_message == "Test error"
        assert result.relative_error == float('inf')


if __name__ == "__main__":
    pytest.main([__file__])