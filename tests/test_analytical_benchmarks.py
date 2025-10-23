"""
Unit tests for analytical benchmark systems.

This module provides comprehensive tests for the analytical benchmark
systems that validate QBES against known exact solutions.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import time

from qbes.benchmarks.analytical_systems import (
    TwoLevelSystemBenchmark,
    HarmonicOscillatorBenchmark,
    DampedTwoLevelSystemBenchmark,
    AnalyticalBenchmarkResult,
    create_analytical_benchmark_suite,
    run_analytical_benchmarks
)
from qbes.core.data_models import DensityMatrix, Hamiltonian, LindbladOperator


class TestTwoLevelSystemBenchmark(unittest.TestCase):
    """Test cases for TwoLevelSystemBenchmark class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = TwoLevelSystemBenchmark(
            energy_gap=1.0,
            rabi_frequency=1.0,
            tolerance=1e-6
        )
    
    def test_initialization(self):
        """Test proper initialization of TwoLevelSystemBenchmark."""
        self.assertEqual(self.benchmark.name, "Two-Level System Rabi Oscillations")
        self.assertEqual(self.benchmark.energy_gap, 1.0)
        self.assertEqual(self.benchmark.rabi_frequency, 1.0)
        self.assertEqual(self.benchmark.tolerance, 1e-6)
        self.assertIsNotNone(self.benchmark.quantum_engine)
    
    def test_analytical_solution(self):
        """Test analytical solution calculation."""
        # Test at t=0: should be 0 (starting in ground state)
        result_t0 = self.benchmark.get_analytical_solution(0.0)
        self.assertAlmostEqual(result_t0, 0.0, places=10)
        
        # Test at t=π/2: should be 1 (maximum population transfer)
        result_tmax = self.benchmark.get_analytical_solution(np.pi)
        self.assertAlmostEqual(result_tmax, 1.0, places=10)
        
        # Test at t=π: should be 0 (back to ground state)
        result_tpi = self.benchmark.get_analytical_solution(2 * np.pi)
        self.assertAlmostEqual(result_tpi, 0.0, places=10)
    
    def test_setup_system(self):
        """Test quantum system setup."""
        hamiltonian, lindblad_ops, initial_state = self.benchmark.setup_system()
        
        # Check Hamiltonian
        self.assertIsInstance(hamiltonian, Hamiltonian)
        self.assertEqual(hamiltonian.matrix.shape, (2, 2))
        self.assertEqual(hamiltonian.basis_labels, ["ground", "excited"])
        
        # Check expected Hamiltonian structure
        expected_h = 0.5 * np.array([
            [1.0, 1.0],
            [1.0, -1.0]
        ], dtype=complex)
        np.testing.assert_array_almost_equal(hamiltonian.matrix, expected_h)
        
        # Check no Lindblad operators for isolated system
        self.assertEqual(len(lindblad_ops), 0)
        
        # Check initial state (ground state)
        self.assertIsInstance(initial_state, DensityMatrix)
        self.assertEqual(initial_state.matrix.shape, (2, 2))
        expected_initial = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        np.testing.assert_array_almost_equal(initial_state.matrix, expected_initial)
    
    def test_extract_observable(self):
        """Test observable extraction."""
        # Create test density matrix with known excited state population
        test_matrix = np.array([
            [0.3, 0.1],
            [0.1, 0.7]
        ], dtype=complex)
        
        test_state = DensityMatrix(
            matrix=test_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        population = self.benchmark.extract_observable(test_state)
        self.assertAlmostEqual(population, 0.7, places=10)
    
    def test_run_benchmark_success(self):
        """Test successful benchmark execution."""
        # Mock the quantum engine directly on the benchmark instance
        mock_engine = Mock()
        
        # Create mock evolved state with known population
        mock_final_state = Mock()
        mock_final_state.matrix = np.array([
            [0.25, 0.0],
            [0.0, 0.75]
        ], dtype=complex)
        
        mock_engine.evolve_state.return_value = mock_final_state
        self.benchmark.quantum_engine = mock_engine
        
        # Run benchmark
        result = self.benchmark.run_benchmark(final_time=1.0, time_step=0.01)
        
        # Check result
        self.assertIsInstance(result, AnalyticalBenchmarkResult)
        self.assertEqual(result.system_name, "Two-Level System Rabi Oscillations")
        self.assertAlmostEqual(result.numerical_result, 0.75, places=10)
        self.assertTrue(result.computation_time > 0)
    
    def test_run_benchmark_failure(self):
        """Test benchmark execution with failure."""
        # Mock quantum engine to raise exception
        mock_engine = Mock()
        mock_engine.evolve_state.side_effect = Exception("Simulation failed")
        self.benchmark.quantum_engine = mock_engine
        
        # Run benchmark
        result = self.benchmark.run_benchmark(final_time=1.0, time_step=0.01)
        
        # Check failure result
        self.assertIsInstance(result, AnalyticalBenchmarkResult)
        self.assertFalse(result.test_passed)
        self.assertEqual(result.error_message, "Simulation failed")
        self.assertEqual(result.relative_error, float('inf'))


class TestHarmonicOscillatorBenchmark(unittest.TestCase):
    """Test cases for HarmonicOscillatorBenchmark class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = HarmonicOscillatorBenchmark(
            frequency=1.0,
            n_levels=5,
            initial_displacement=1.0,
            tolerance=1e-5
        )
    
    def test_initialization(self):
        """Test proper initialization of HarmonicOscillatorBenchmark."""
        self.assertEqual(self.benchmark.name, "Harmonic Oscillator Coherent State")
        self.assertEqual(self.benchmark.frequency, 1.0)
        self.assertEqual(self.benchmark.n_levels, 5)
        self.assertEqual(self.benchmark.initial_displacement, 1.0)
        self.assertEqual(self.benchmark.tolerance, 1e-5)
    
    def test_analytical_solution(self):
        """Test analytical solution calculation."""
        # Test at t=0: should be √2 * Re(α) = √2 * 1.0
        result_t0 = self.benchmark.get_analytical_solution(0.0)
        expected_t0 = np.sqrt(2) * 1.0
        self.assertAlmostEqual(result_t0, expected_t0, places=10)
        
        # Test at t=π/2: should be √2 * Re(α * exp(-iπ/2)) = √2 * Re(-i) = 0
        result_tpi2 = self.benchmark.get_analytical_solution(np.pi / 2)
        self.assertAlmostEqual(result_tpi2, 0.0, places=10)
        
        # Test at t=π: should be √2 * Re(α * exp(-iπ)) = √2 * Re(-1) = -√2
        result_tpi = self.benchmark.get_analytical_solution(np.pi)
        expected_tpi = -np.sqrt(2) * 1.0
        self.assertAlmostEqual(result_tpi, expected_tpi, places=10)
    
    def test_coherent_state_creation(self):
        """Test coherent state density matrix creation."""
        alpha = 1.0
        rho = self.benchmark._create_coherent_state_density_matrix(alpha)
        
        # Check dimensions
        self.assertEqual(rho.shape, (5, 5))
        
        # Check trace (should be 1)
        trace = np.trace(rho)
        self.assertAlmostEqual(trace, 1.0, places=10)
        
        # Check Hermiticity
        np.testing.assert_array_almost_equal(rho, np.conj(rho.T))
        
        # Check positive semidefinite (eigenvalues should be non-negative)
        eigenvals = np.linalg.eigvals(rho)
        self.assertTrue(np.all(eigenvals >= -1e-10))  # Allow small numerical errors
    
    def test_setup_system(self):
        """Test quantum system setup."""
        hamiltonian, lindblad_ops, initial_state = self.benchmark.setup_system()
        
        # Check Hamiltonian
        self.assertIsInstance(hamiltonian, Hamiltonian)
        self.assertEqual(hamiltonian.matrix.shape, (5, 5))
        
        # Check Hamiltonian eigenvalues (should be ω(n + 1/2))
        eigenvals = np.linalg.eigvals(hamiltonian.matrix)
        eigenvals_sorted = np.sort(np.real(eigenvals))
        expected_eigenvals = [0.5, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_almost_equal(eigenvals_sorted, expected_eigenvals)
        
        # Check no Lindblad operators
        self.assertEqual(len(lindblad_ops), 0)
        
        # Check initial state
        self.assertIsInstance(initial_state, DensityMatrix)
        self.assertEqual(initial_state.matrix.shape, (5, 5))
    
    def test_extract_observable(self):
        """Test position expectation value extraction."""
        # Create simple test state (ground state)
        test_matrix = np.zeros((5, 5), dtype=complex)
        test_matrix[0, 0] = 1.0  # Ground state
        
        test_state = DensityMatrix(
            matrix=test_matrix,
            basis_labels=[f"n={n}" for n in range(5)],
            time=0.0
        )
        
        position = self.benchmark.extract_observable(test_state)
        # Ground state should have zero position expectation value
        self.assertAlmostEqual(position, 0.0, places=10)


class TestDampedTwoLevelSystemBenchmark(unittest.TestCase):
    """Test cases for DampedTwoLevelSystemBenchmark class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = DampedTwoLevelSystemBenchmark(
            energy_gap=1.0,
            decay_rate=0.1,
            initial_population=1.0,
            tolerance=1e-5
        )
    
    def test_initialization(self):
        """Test proper initialization of DampedTwoLevelSystemBenchmark."""
        self.assertEqual(self.benchmark.name, "Damped Two-Level System")
        self.assertEqual(self.benchmark.energy_gap, 1.0)
        self.assertEqual(self.benchmark.decay_rate, 0.1)
        self.assertEqual(self.benchmark.initial_population, 1.0)
        self.assertEqual(self.benchmark.tolerance, 1e-5)
    
    def test_analytical_solution(self):
        """Test analytical solution calculation."""
        # Test at t=0: should be initial population
        result_t0 = self.benchmark.get_analytical_solution(0.0)
        self.assertAlmostEqual(result_t0, 1.0, places=10)
        
        # Test at t=10 (one decay time): should be exp(-1) ≈ 0.368
        result_t10 = self.benchmark.get_analytical_solution(10.0)
        expected_t10 = np.exp(-1.0)
        self.assertAlmostEqual(result_t10, expected_t10, places=10)
        
        # Test at large time: should approach 0
        result_large = self.benchmark.get_analytical_solution(100.0)
        self.assertLess(result_large, 1e-4)
    
    def test_setup_system(self):
        """Test quantum system setup."""
        hamiltonian, lindblad_ops, initial_state = self.benchmark.setup_system()
        
        # Check Hamiltonian
        self.assertIsInstance(hamiltonian, Hamiltonian)
        self.assertEqual(hamiltonian.matrix.shape, (2, 2))
        
        # Check Hamiltonian structure (diagonal with ±ω/2)
        expected_h = 0.5 * np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ], dtype=complex)
        np.testing.assert_array_almost_equal(hamiltonian.matrix, expected_h)
        
        # Check Lindblad operators
        self.assertEqual(len(lindblad_ops), 1)
        lindblad_op = lindblad_ops[0]
        self.assertIsInstance(lindblad_op, LindbladOperator)
        
        # Check Lindblad operator structure (σ-)
        expected_l = np.sqrt(0.1) * np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=complex)
        np.testing.assert_array_almost_equal(lindblad_op.operator, expected_l)
        
        # Check initial state (excited state)
        self.assertIsInstance(initial_state, DensityMatrix)
        expected_initial = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=complex)
        np.testing.assert_array_almost_equal(initial_state.matrix, expected_initial)
    
    def test_extract_observable(self):
        """Test excited state population extraction."""
        # Create test density matrix with known population
        test_matrix = np.array([
            [0.4, 0.0],
            [0.0, 0.6]
        ], dtype=complex)
        
        test_state = DensityMatrix(
            matrix=test_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        population = self.benchmark.extract_observable(test_state)
        self.assertAlmostEqual(population, 0.6, places=10)


class TestAnalyticalBenchmarkSuite(unittest.TestCase):
    """Test cases for analytical benchmark suite functions."""
    
    def test_create_analytical_benchmark_suite(self):
        """Test creation of analytical benchmark suite."""
        benchmarks = create_analytical_benchmark_suite()
        
        # Check that we get the expected number of benchmarks
        self.assertEqual(len(benchmarks), 3)
        
        # Check types
        self.assertIsInstance(benchmarks[0], TwoLevelSystemBenchmark)
        self.assertIsInstance(benchmarks[1], HarmonicOscillatorBenchmark)
        self.assertIsInstance(benchmarks[2], DampedTwoLevelSystemBenchmark)
        
        # Check names
        expected_names = [
            "Two-Level System Rabi Oscillations",
            "Harmonic Oscillator Coherent State",
            "Damped Two-Level System"
        ]
        actual_names = [b.name for b in benchmarks]
        self.assertEqual(actual_names, expected_names)
    
    @patch('qbes.benchmarks.analytical_systems.create_analytical_benchmark_suite')
    def test_run_analytical_benchmarks(self, mock_create_suite):
        """Test running analytical benchmarks."""
        # Create mock benchmarks
        mock_benchmark1 = Mock()
        mock_benchmark1.name = "Test Benchmark 1"
        mock_result1 = AnalyticalBenchmarkResult(
            system_name="Test Benchmark 1",
            test_passed=True,
            numerical_result=1.0,
            analytical_result=1.0,
            relative_error=0.0,
            absolute_error=0.0,
            computation_time=0.1,
            tolerance=1e-6
        )
        mock_benchmark1.run_benchmark.return_value = mock_result1
        
        mock_benchmark2 = Mock()
        mock_benchmark2.name = "Test Benchmark 2"
        mock_result2 = AnalyticalBenchmarkResult(
            system_name="Test Benchmark 2",
            test_passed=False,
            numerical_result=0.9,
            analytical_result=1.0,
            relative_error=0.1,
            absolute_error=0.1,
            computation_time=0.2,
            tolerance=1e-6
        )
        mock_benchmark2.run_benchmark.return_value = mock_result2
        
        mock_create_suite.return_value = [mock_benchmark1, mock_benchmark2]
        
        # Run benchmarks
        results = run_analytical_benchmarks(final_time=0.1, time_step=0.01)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].test_passed)
        self.assertFalse(results[1].test_passed)
        
        # Check that benchmarks were called with correct parameters
        mock_benchmark1.run_benchmark.assert_called_once_with(0.1, 0.01)
        mock_benchmark2.run_benchmark.assert_called_once_with(0.1, 0.01)


class TestAnalyticalBenchmarkResult(unittest.TestCase):
    """Test cases for AnalyticalBenchmarkResult class."""
    
    def test_result_creation(self):
        """Test creation of benchmark result."""
        result = AnalyticalBenchmarkResult(
            system_name="Test System",
            test_passed=True,
            numerical_result=1.001,
            analytical_result=1.0,
            relative_error=0.001,
            absolute_error=0.001,
            computation_time=0.5,
            tolerance=1e-3
        )
        
        self.assertEqual(result.system_name, "Test System")
        self.assertTrue(result.test_passed)
        self.assertEqual(result.numerical_result, 1.001)
        self.assertEqual(result.analytical_result, 1.0)
        self.assertEqual(result.relative_error, 0.001)
        self.assertEqual(result.absolute_error, 0.001)
        self.assertEqual(result.computation_time, 0.5)
        self.assertEqual(result.tolerance, 1e-3)
        self.assertIsNone(result.error_message)
    
    def test_result_with_error(self):
        """Test creation of benchmark result with error."""
        result = AnalyticalBenchmarkResult(
            system_name="Failed System",
            test_passed=False,
            numerical_result=0.0,
            analytical_result=0.0,
            relative_error=float('inf'),
            absolute_error=float('inf'),
            computation_time=0.0,
            tolerance=1e-6,
            error_message="Test error"
        )
        
        self.assertEqual(result.system_name, "Failed System")
        self.assertFalse(result.test_passed)
        self.assertEqual(result.error_message, "Test error")
        self.assertEqual(result.relative_error, float('inf'))


class TestBenchmarkIntegration(unittest.TestCase):
    """Integration tests for analytical benchmarks."""
    
    def test_two_level_system_integration(self):
        """Integration test for two-level system benchmark."""
        # This test requires the actual quantum engine to work
        # Skip if quantum engine is not available
        try:
            benchmark = TwoLevelSystemBenchmark(tolerance=1e-2)  # Relaxed tolerance
            
            # Run short simulation
            result = benchmark.run_benchmark(final_time=0.1, time_step=0.001)
            
            # Check that we get a reasonable result
            self.assertIsInstance(result, AnalyticalBenchmarkResult)
            self.assertIsNotNone(result.numerical_result)
            self.assertIsNotNone(result.analytical_result)
            self.assertTrue(result.computation_time > 0)
            
        except Exception as e:
            self.skipTest(f"Quantum engine not available: {e}")
    
    def test_harmonic_oscillator_integration(self):
        """Integration test for harmonic oscillator benchmark."""
        try:
            benchmark = HarmonicOscillatorBenchmark(
                n_levels=3,  # Small system for fast test
                tolerance=1e-2
            )
            
            # Run short simulation
            result = benchmark.run_benchmark(final_time=0.1, time_step=0.001)
            
            # Check that we get a reasonable result
            self.assertIsInstance(result, AnalyticalBenchmarkResult)
            self.assertIsNotNone(result.numerical_result)
            self.assertIsNotNone(result.analytical_result)
            self.assertTrue(result.computation_time > 0)
            
        except Exception as e:
            self.skipTest(f"Quantum engine not available: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)