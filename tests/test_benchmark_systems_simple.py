"""
Tests for benchmark systems and validation suite (simplified version).
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from qbes.benchmarks.benchmark_systems import (
    BenchmarkSystem, BenchmarkResult, TwoLevelSystemBenchmark,
    BenchmarkRunner, run_quick_benchmarks
)
from qbes.core.data_models import DensityMatrix, Hamiltonian, LindbladOperator


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating benchmark result."""
        result = BenchmarkResult(
            system_name="Test System",
            test_passed=True,
            numerical_result=1.0,
            analytical_result=1.1,
            relative_error=0.1,
            absolute_error=0.1,
            computation_time=0.5,
            tolerance=1e-6
        )
        
        assert result.system_name == "Test System"
        assert result.test_passed is True
        assert result.numerical_result == 1.0
        assert result.analytical_result == 1.1
        assert result.relative_error == 0.1
        assert result.computation_time == 0.5
        assert result.error_message is None


class TestTwoLevelSystemBenchmark:
    """Test two-level system benchmark."""
    
    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = TwoLevelSystemBenchmark(energy_gap=2.0, tolerance=1e-5)
        
        assert benchmark.name == "Two-Level System"
        assert benchmark.energy_gap == 2.0
        assert benchmark.tolerance == 1e-5
    
    def test_analytical_solution(self):
        """Test analytical solution calculation."""
        benchmark = TwoLevelSystemBenchmark(energy_gap=1.0)
        
        # At t=0, population should be 0 (starts in ground state)
        result_t0 = benchmark.get_analytical_solution(0.0)
        assert abs(result_t0 - 0.0) < 1e-10
        
        # At t=π, population should be 1 (full oscillation)
        result_tpi = benchmark.get_analytical_solution(np.pi)
        expected = np.sin(np.pi / 2) ** 2  # sin²(π/2) = 1
        assert abs(result_tpi - expected) < 1e-10
    
    def test_system_setup(self):
        """Test quantum system setup."""
        benchmark = TwoLevelSystemBenchmark(energy_gap=1.0)
        hamiltonian, lindblad_ops, initial_state = benchmark.setup_system()
        
        # Check Hamiltonian
        assert hamiltonian.matrix.shape == (2, 2)
        assert hamiltonian.basis_labels == ["ground", "excited"]
        assert not hamiltonian.time_dependent
        
        # Check no Lindblad operators for isolated system
        assert len(lindblad_ops) == 0
        
        # Check initial state (ground state)
        assert initial_state.matrix.shape == (2, 2)
        assert abs(initial_state.matrix[0, 0] - 1.0) < 1e-10  # Ground state population
        assert abs(initial_state.matrix[1, 1] - 0.0) < 1e-10  # Excited state population
    
    def test_extract_observable(self):
        """Test observable extraction."""
        benchmark = TwoLevelSystemBenchmark()
        
        # Create test state with 0.3 excited state population
        test_matrix = np.array([[0.7, 0], [0, 0.3]], dtype=complex)
        test_state = DensityMatrix(
            matrix=test_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        observable = benchmark.extract_observable(test_state)
        assert abs(observable - 0.3) < 1e-10


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""
    
    def test_initialization(self):
        """Test runner initialization."""
        runner = BenchmarkRunner()
        
        assert len(runner.benchmarks) == 0
        assert len(runner.results) == 0
    
    def test_add_benchmark(self):
        """Test adding benchmarks."""
        runner = BenchmarkRunner()
        benchmark = TwoLevelSystemBenchmark()
        
        runner.add_benchmark(benchmark)
        
        assert len(runner.benchmarks) == 1
        assert runner.benchmarks[0] == benchmark
    
    def test_add_standard_benchmarks(self):
        """Test adding standard benchmark suite."""
        runner = BenchmarkRunner()
        runner.add_standard_benchmarks()
        
        assert len(runner.benchmarks) == 1  # Only TwoLevelSystemBenchmark for now
        
        # Check that expected benchmark is present
        benchmark_names = [b.name for b in runner.benchmarks]
        assert "Two-Level System" in benchmark_names


class TestQuickBenchmarks:
    """Test quick benchmark execution function."""
    
    def test_run_quick_benchmarks(self):
        """Test quick benchmark execution."""
        runner = run_quick_benchmarks()
        
        assert isinstance(runner, BenchmarkRunner)
        assert len(runner.benchmarks) == 1  # Standard benchmarks


if __name__ == "__main__":
    pytest.main([__file__])