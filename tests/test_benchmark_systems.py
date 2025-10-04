"""
Tests for benchmark systems and validation suite.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from qbes.benchmarks.benchmark_systems import (
    BenchmarkSystem, BenchmarkResult, TwoLevelSystemBenchmark,
    HarmonicOscillatorBenchmark, DampedTwoLevelSystemBenchmark,
    PhotosyntheticComplexBenchmark, BenchmarkRunner, run_quick_benchmarks
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


class TestHarmonicOscillatorBenchmark:
    """Test harmonic oscillator benchmark."""
    
    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = HarmonicOscillatorBenchmark(frequency=2.0, n_levels=5)
        
        assert benchmark.name == "Harmonic Oscillator"
        assert benchmark.frequency == 2.0
        assert benchmark.n_levels == 5
    
    def test_analytical_solution(self):
        """Test analytical solution for coherent state."""
        benchmark = HarmonicOscillatorBenchmark(frequency=1.0)
        
        # At t=0, position should be √2 * Re(α) = √2 for α=1
        result_t0 = benchmark.get_analytical_solution(0.0, alpha=1.0)
        expected = np.sqrt(2) * np.real(1.0)
        assert abs(result_t0 - expected) < 1e-10
        
        # At t=π/2, position should be √2 * Re(i) = 0
        result_tpi2 = benchmark.get_analytical_solution(np.pi/2, alpha=1.0)
        expected = np.sqrt(2) * np.real(1j)
        assert abs(result_tpi2 - expected) < 1e-10
    
    def test_coherent_state_creation(self):
        """Test coherent state density matrix creation."""
        benchmark = HarmonicOscillatorBenchmark(n_levels=3)
        
        alpha = 0.5
        rho = benchmark._create_coherent_state_density_matrix(alpha)
        
        # Check normalization
        assert abs(np.trace(rho) - 1.0) < 1e-10
        
        # Check Hermiticity
        assert np.allclose(rho, rho.conj().T)
        
        # Check positive semidefinite (eigenvalues >= 0)
        eigenvals = np.linalg.eigvals(rho)
        assert all(eigenvals >= -1e-10)


class TestDampedTwoLevelSystemBenchmark:
    """Test damped two-level system benchmark."""
    
    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = DampedTwoLevelSystemBenchmark(
            energy_gap=1.0, decay_rate=0.2
        )
        
        assert benchmark.name == "Damped Two-Level System"
        assert benchmark.energy_gap == 1.0
        assert benchmark.decay_rate == 0.2
    
    def test_analytical_solution(self):
        """Test analytical solution for population decay."""
        benchmark = DampedTwoLevelSystemBenchmark(decay_rate=0.1)
        
        # At t=0, population should be initial value
        result_t0 = benchmark.get_analytical_solution(0.0, initial_population=1.0)
        assert abs(result_t0 - 1.0) < 1e-10
        
        # At t=10 (10 decay times), population should be ~e^(-1) ≈ 0.368
        result_t10 = benchmark.get_analytical_solution(10.0, initial_population=1.0)
        expected = np.exp(-1.0)
        assert abs(result_t10 - expected) < 1e-10
    
    def test_system_setup(self):
        """Test system setup with Lindblad operators."""
        benchmark = DampedTwoLevelSystemBenchmark(decay_rate=0.1)
        hamiltonian, lindblad_ops, initial_state = benchmark.setup_system()
        
        # Check Lindblad operators
        assert len(lindblad_ops) == 1
        assert lindblad_ops[0].description == "spontaneous_emission"
        assert abs(lindblad_ops[0].coupling_strength - np.sqrt(0.1)) < 1e-10
        
        # Check initial state (excited state)
        assert abs(initial_state.matrix[1, 1] - 1.0) < 1e-10


class TestPhotosyntheticComplexBenchmark:
    """Test photosynthetic complex benchmark."""
    
    def test_initialization(self):
        """Test benchmark initialization."""
        benchmark = PhotosyntheticComplexBenchmark(
            site_energy_1=12000, site_energy_2=12200,
            coupling=100, temperature=300
        )
        
        assert benchmark.name == "Photosynthetic Complex"
        assert benchmark.site_energy_1 == 12000
        assert benchmark.site_energy_2 == 12200
        assert benchmark.coupling == 100
        assert benchmark.temperature == 300
    
    def test_analytical_solution(self):
        """Test analytical solution for coherence decay."""
        benchmark = PhotosyntheticComplexBenchmark(
            coupling=100, reorganization_energy=35, temperature=300
        )
        
        # At t=0, coherence should be initial value
        result_t0 = benchmark.get_analytical_solution(0.0, initial_coherence=0.5)
        assert abs(result_t0 - 0.5) < 1e-10
        
        # At later times, coherence should decay
        result_t1 = benchmark.get_analytical_solution(1.0, initial_coherence=0.5)
        assert result_t1 < 0.5  # Should have decayed
    
    def test_system_setup(self):
        """Test photosynthetic system setup."""
        benchmark = PhotosyntheticComplexBenchmark()
        hamiltonian, lindblad_ops, initial_state = benchmark.setup_system()
        
        # Check Hamiltonian (2x2 for dimer)
        assert hamiltonian.matrix.shape == (2, 2)
        assert hamiltonian.basis_labels == ["site_1", "site_2"]
        
        # Check Lindblad operators for dephasing
        assert len(lindblad_ops) == 2
        
        # Check initial state has coherence
        assert abs(initial_state.matrix[0, 1]) > 0  # Off-diagonal element


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
        
        assert len(runner.benchmarks) == 4  # All standard benchmarks
        
        # Check that all expected benchmarks are present
        benchmark_names = [b.name for b in runner.benchmarks]
        expected_names = [
            "Two-Level System", "Harmonic Oscillator",
            "Damped Two-Level System", "Photosynthetic Complex"
        ]
        
        for name in expected_names:
            assert name in benchmark_names
    
    @patch('qbes.benchmarks.benchmark_systems.BenchmarkSystem.run_benchmark')
    def test_run_all_benchmarks(self, mock_run_benchmark):
        """Test running all benchmarks."""
        # Mock successful benchmark result
        mock_result = BenchmarkResult(
            system_name="Test",
            test_passed=True,
            numerical_result=1.0,
            analytical_result=1.0,
            relative_error=0.0,
            absolute_error=0.0,
            computation_time=0.1,
            tolerance=1e-6
        )
        mock_run_benchmark.return_value = mock_result
        
        runner = BenchmarkRunner()
        runner.add_benchmark(TwoLevelSystemBenchmark())
        
        results = runner.run_all_benchmarks()
        
        assert len(results) == 1
        assert results[0].test_passed is True
        mock_run_benchmark.assert_called_once()
    
    def test_generate_report(self):
        """Test report generation."""
        runner = BenchmarkRunner()
        
        # Add mock results
        runner.results = [
            BenchmarkResult(
                system_name="Test1",
                test_passed=True,
                numerical_result=1.0,
                analytical_result=1.0,
                relative_error=1e-8,
                absolute_error=1e-8,
                computation_time=0.1,
                tolerance=1e-6
            ),
            BenchmarkResult(
                system_name="Test2",
                test_passed=False,
                numerical_result=1.0,
                analytical_result=2.0,
                relative_error=0.5,
                absolute_error=1.0,
                computation_time=0.2,
                tolerance=1e-6,
                error_message="Test error"
            )
        ]
        
        report = runner.generate_report()
        
        assert "QBES Benchmark Test Report" in report
        assert "Total Tests: 2" in report
        assert "Passed: 1" in report
        assert "Failed: 1" in report
        assert "Success Rate: 50.0%" in report
        assert "Test1" in report
        assert "Test2" in report
        assert "Test error" in report
    
    def test_get_failed_tests(self):
        """Test getting failed tests."""
        runner = BenchmarkRunner()
        
        # Add mock results
        passed_result = BenchmarkResult(
            system_name="Passed",
            test_passed=True,
            numerical_result=1.0,
            analytical_result=1.0,
            relative_error=0.0,
            absolute_error=0.0,
            computation_time=0.1,
            tolerance=1e-6
        )
        
        failed_result = BenchmarkResult(
            system_name="Failed",
            test_passed=False,
            numerical_result=1.0,
            analytical_result=2.0,
            relative_error=0.5,
            absolute_error=1.0,
            computation_time=0.1,
            tolerance=1e-6
        )
        
        runner.results = [passed_result, failed_result]
        
        failed_tests = runner.get_failed_tests()
        
        assert len(failed_tests) == 1
        assert failed_tests[0].system_name == "Failed"
    
    def test_performance_scaling_test(self):
        """Test performance scaling analysis."""
        runner = BenchmarkRunner()
        
        # Test with small system sizes to avoid long computation
        scaling_results = runner.performance_scaling_test(system_sizes=[2, 4])
        
        assert len(scaling_results) == 2
        assert 2 in scaling_results
        assert 4 in scaling_results
        
        # Check that computation times are positive
        for size, time in scaling_results.items():
            assert time >= 0


class TestQuickBenchmarks:
    """Test quick benchmark execution function."""
    
    @patch('qbes.benchmarks.benchmark_systems.BenchmarkRunner.run_all_benchmarks')
    def test_run_quick_benchmarks(self, mock_run_all):
        """Test quick benchmark execution."""
        mock_run_all.return_value = []
        
        runner = run_quick_benchmarks()
        
        assert isinstance(runner, BenchmarkRunner)
        assert len(runner.benchmarks) == 4  # Standard benchmarks
        mock_run_all.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])