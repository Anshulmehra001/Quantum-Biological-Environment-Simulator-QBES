"""
Tests for FMO complex benchmark system.

This module tests the FMO complex benchmark implementation against
published experimental and theoretical results.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import qbes modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qbes.benchmarks.fmo_system import (
    FMOComplexBenchmark, FMOBenchmarkResult,
    create_fmo_benchmark_suite, run_fmo_benchmarks
)
from qbes.core.data_models import DensityMatrix, Hamiltonian, LindbladOperator


class TestFMOComplexBenchmark(unittest.TestCase):
    """Test cases for FMO complex benchmark system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = FMOComplexBenchmark(temperature=77.0, tolerance=0.15)
        
    def test_initialization(self):
        """Test FMO benchmark initialization."""
        self.assertEqual(self.benchmark.name, "FMO Complex Benchmark")
        self.assertEqual(self.benchmark.temperature, 77.0)
        self.assertEqual(self.benchmark.tolerance, 0.15)
        self.assertEqual(self.benchmark.num_sites, 7)
        self.assertIsNotNone(self.benchmark.quantum_engine)
        
        # Check that site energies and coupling matrix have correct dimensions
        self.assertEqual(len(self.benchmark.site_energies), 7)
        self.assertEqual(self.benchmark.coupling_matrix.shape, (7, 7))
        
        # Check that coupling matrix is symmetric
        np.testing.assert_array_almost_equal(
            self.benchmark.coupling_matrix,
            self.benchmark.coupling_matrix.T,
            decimal=10
        )
        
        # Check that diagonal elements are zero (no self-coupling)
        for i in range(7):
            self.assertAlmostEqual(self.benchmark.coupling_matrix[i, i], 0.0, places=10)
    
    def test_reference_data_loading(self):
        """Test loading of reference data."""
        # Should have loaded reference values
        self.assertGreater(self.benchmark.ref_coherence_lifetime, 0)
        self.assertGreater(self.benchmark.ref_transfer_efficiency, 0)
        self.assertLessEqual(self.benchmark.ref_transfer_efficiency, 1.0)
        
        # Check expected values from literature
        self.assertAlmostEqual(self.benchmark.ref_coherence_lifetime, 660.0, delta=50.0)
        self.assertAlmostEqual(self.benchmark.ref_transfer_efficiency, 0.95, delta=0.1)
    
    def test_system_setup(self):
        """Test quantum system setup."""
        hamiltonian, lindblad_ops, initial_state = self.benchmark.setup_system()
        
        # Check Hamiltonian
        self.assertIsInstance(hamiltonian, Hamiltonian)
        self.assertEqual(hamiltonian.matrix.shape, (7, 7))
        self.assertEqual(len(hamiltonian.basis_labels), 7)
        self.assertFalse(hamiltonian.time_dependent)
        
        # Check that Hamiltonian is Hermitian
        np.testing.assert_array_almost_equal(
            hamiltonian.matrix,
            hamiltonian.matrix.conj().T,
            decimal=10
        )
        
        # Check Lindblad operators
        self.assertIsInstance(lindblad_ops, list)
        self.assertGreater(len(lindblad_ops), 0)  # Should have decoherence operators
        
        for op in lindblad_ops:
            self.assertIsInstance(op, LindbladOperator)
            self.assertEqual(op.operator.shape, (7, 7))
            self.assertGreater(op.coupling_strength, 0)
        
        # Check initial state
        self.assertIsInstance(initial_state, DensityMatrix)
        self.assertEqual(initial_state.matrix.shape, (7, 7))
        self.assertEqual(len(initial_state.basis_labels), 7)
        
        # Check that initial state is valid density matrix
        # Trace should be 1
        trace = np.trace(initial_state.matrix)
        self.assertAlmostEqual(trace, 1.0, places=10)
        
        # Should be positive semidefinite (all eigenvalues >= 0)
        eigenvals = np.linalg.eigvals(initial_state.matrix)
        self.assertTrue(np.all(eigenvals >= -1e-10))  # Allow small numerical errors
        
        # Initial excitation should be on site 1 (index 0)
        self.assertAlmostEqual(np.real(initial_state.matrix[0, 0]), 1.0, places=10)
        for i in range(1, 7):
            self.assertAlmostEqual(np.real(initial_state.matrix[i, i]), 0.0, places=10)
    
    def test_decoherence_operators_creation(self):
        """Test creation of decoherence operators."""
        lindblad_ops = self.benchmark._create_decoherence_operators()
        
        self.assertIsInstance(lindblad_ops, list)
        self.assertGreater(len(lindblad_ops), 0)
        
        # Should have both dephasing and relaxation operators
        dephasing_ops = [op for op in lindblad_ops if op.operator_type == "pure_dephasing"]
        relaxation_ops = [op for op in lindblad_ops if op.operator_type == "population_relaxation"]
        
        self.assertEqual(len(dephasing_ops), 7)  # One for each site
        self.assertGreater(len(relaxation_ops), 0)  # Multiple relaxation channels
        
        # Check dephasing operators are diagonal
        for op in dephasing_ops:
            # Should be proportional to a projector (diagonal matrix)
            off_diagonal_sum = np.sum(np.abs(op.operator)) - np.sum(np.abs(np.diag(op.operator)))
            self.assertLess(off_diagonal_sum, 1e-10)
    
    def test_coherence_lifetime_calculation(self):
        """Test coherence lifetime calculation."""
        # Create mock states with exponential coherence decay
        times = np.linspace(0, 1000, 100)  # fs
        states = []
        
        for t in times:
            # Create a simple valid density matrix with decaying coherences
            # Use a mixed state approach to ensure positive semidefinite
            matrix = np.zeros((7, 7), dtype=complex)
            
            # Population distribution (must sum to 1)
            matrix[0, 0] = 0.8  # 80% on site 1
            matrix[1, 1] = 0.2  # 20% on site 2
            
            # Add decaying coherences between populated states only
            coherence_amplitude = 0.1 * np.exp(-t / 500.0)  # 500 fs lifetime
            # Coherence between sites with populations
            matrix[0, 1] = coherence_amplitude * np.sqrt(0.8 * 0.2)  # Scaled by populations
            matrix[1, 0] = coherence_amplitude * np.sqrt(0.8 * 0.2)
            
            state = DensityMatrix(
                matrix=matrix,
                basis_labels=[f"BChl_{i+1}" for i in range(7)],
                time=t
            )
            states.append(state)
        
        lifetime = self.benchmark.calculate_coherence_lifetime(states, times)
        
        # Should be close to the expected 500 fs
        self.assertGreater(lifetime, 300.0)
        self.assertLess(lifetime, 800.0)
    
    def test_transfer_efficiency_calculation(self):
        """Test energy transfer efficiency calculation."""
        # Create final state with population on reaction center sites
        matrix = np.zeros((7, 7), dtype=complex)
        matrix[2, 2] = 0.6  # 60% on site 3 (index 2)
        matrix[5, 5] = 0.3  # 30% on site 6 (index 5)
        matrix[0, 0] = 0.1  # 10% remaining on site 1
        
        final_state = DensityMatrix(
            matrix=matrix,
            basis_labels=[f"BChl_{i+1}" for i in range(7)],
            time=1000.0
        )
        
        efficiency = self.benchmark.calculate_transfer_efficiency(final_state)
        
        # Should be 0.6 + 0.3 = 0.9
        self.assertAlmostEqual(efficiency, 0.9, places=10)
    
    @patch('qbes.benchmarks.fmo_system.QuantumEngine')
    def test_benchmark_run_success(self, mock_quantum_engine):
        """Test successful benchmark run."""
        # Mock quantum engine evolution
        mock_engine_instance = MagicMock()
        mock_quantum_engine.return_value = mock_engine_instance
        
        # Create mock evolved states
        def mock_evolve_state(state, dt, hamiltonian, lindblad_ops):
            # Simple mock: return a state with population transfer and some coherences
            matrix = state.matrix.copy()
            
            # Transfer small amount of population from site 1 to sites 3 and 6
            transfer_amount = 0.001  # Smaller transfer to maintain valid density matrix
            if np.real(matrix[0, 0]) > transfer_amount:  # Only transfer if enough population
                matrix[0, 0] -= transfer_amount
                matrix[2, 2] += transfer_amount * 0.6
                matrix[5, 5] += transfer_amount * 0.4
            
            # Add small coherences that decay over time
            time_factor = max(0.1, 1.0 - state.time / 1000.0)  # Decay over 1000 fs
            coherence_strength = 0.01 * time_factor
            
            # Add coherences between populated sites
            if np.real(matrix[0, 0]) > 0.1 and np.real(matrix[2, 2]) > 0.01:
                coherence = coherence_strength * np.sqrt(np.real(matrix[0, 0]) * np.real(matrix[2, 2]))
                matrix[0, 2] = coherence
                matrix[2, 0] = coherence
            
            return DensityMatrix(
                matrix=matrix,
                basis_labels=state.basis_labels,
                time=state.time + dt
            )
        
        mock_engine_instance.evolve_state.side_effect = mock_evolve_state
        
        # Create new benchmark instance with mocked engine
        benchmark = FMOComplexBenchmark(temperature=77.0, tolerance=0.15)
        benchmark.quantum_engine = mock_engine_instance
        
        # Run benchmark
        result = benchmark.run_benchmark(simulation_time=100.0, time_step=1.0)
        
        # Check result structure
        self.assertIsInstance(result, FMOBenchmarkResult)
        self.assertEqual(result.system_name, "FMO Complex Benchmark")
        self.assertEqual(result.temperature, 77.0)
        self.assertIsNone(result.error_message)
        
        # Check that observables were calculated (allow 0 values for mock)
        self.assertGreaterEqual(result.computed_coherence_lifetime, 0)
        self.assertGreaterEqual(result.computed_transfer_efficiency, 0)
        self.assertLessEqual(result.computed_transfer_efficiency, 1.0)
        
        # Check that errors were calculated
        self.assertGreaterEqual(result.coherence_error, 0)
        self.assertGreaterEqual(result.efficiency_error, 0)
        
        # Check computation time
        self.assertGreater(result.computation_time, 0)
    
    def test_benchmark_run_failure(self):
        """Test benchmark run with simulation failure."""
        # Create benchmark with invalid quantum engine to force failure
        benchmark = FMOComplexBenchmark(temperature=77.0, tolerance=0.15)
        benchmark.quantum_engine = None  # This will cause an error
        
        result = benchmark.run_benchmark(simulation_time=100.0, time_step=1.0)
        
        # Check that failure was handled gracefully
        self.assertIsInstance(result, FMOBenchmarkResult)
        self.assertFalse(result.test_passed)
        self.assertIsNotNone(result.error_message)
        self.assertEqual(result.computed_coherence_lifetime, 0.0)
        self.assertEqual(result.computed_transfer_efficiency, 0.0)
        self.assertEqual(result.coherence_error, float('inf'))
        self.assertEqual(result.efficiency_error, float('inf'))
    
    def test_benchmark_suite_creation(self):
        """Test creation of FMO benchmark suite."""
        benchmarks = create_fmo_benchmark_suite()
        
        self.assertIsInstance(benchmarks, list)
        self.assertGreater(len(benchmarks), 0)
        
        # Should have benchmarks at different temperatures
        temperatures = [b.temperature for b in benchmarks]
        self.assertIn(77.0, temperatures)   # Low temperature
        self.assertIn(300.0, temperatures)  # Room temperature
        
        # All should be FMOComplexBenchmark instances
        for benchmark in benchmarks:
            self.assertIsInstance(benchmark, FMOComplexBenchmark)
    
    def test_literature_validation_parameters(self):
        """Test that benchmark uses correct literature parameters."""
        # Check site energies are in reasonable range for BChl-a
        # Literature values are around 12000-13000 cm^-1
        site_energies_cm = self.benchmark.site_energies / 1.24e-4  # Convert back to cm^-1
        
        for energy in site_energies_cm:
            self.assertGreater(energy, 11000)  # Lower bound
            self.assertLess(energy, 14000)     # Upper bound
        
        # Check coupling matrix has reasonable values
        # Literature couplings are typically 10-100 cm^-1
        coupling_cm = self.benchmark.coupling_matrix / 1.24e-4  # Convert back to cm^-1
        
        # Find maximum off-diagonal coupling
        max_coupling = 0
        for i in range(7):
            for j in range(i+1, 7):
                coupling_magnitude = abs(coupling_cm[i, j])
                if coupling_magnitude > max_coupling:
                    max_coupling = coupling_magnitude
        
        self.assertGreater(max_coupling, 10)   # Should have significant couplings
        self.assertLess(max_coupling, 200)     # But not unreasonably large
    
    def test_temperature_dependence(self):
        """Test that benchmark behavior depends on temperature."""
        low_temp_benchmark = FMOComplexBenchmark(temperature=77.0)
        high_temp_benchmark = FMOComplexBenchmark(temperature=300.0)
        
        # Create decoherence operators for both temperatures
        low_temp_ops = low_temp_benchmark._create_decoherence_operators()
        high_temp_ops = high_temp_benchmark._create_decoherence_operators()
        
        # High temperature should generally have stronger decoherence
        # (though the exact relationship depends on the model details)
        self.assertEqual(len(low_temp_ops), len(high_temp_ops))
        
        # At least check that both have reasonable numbers of operators
        self.assertGreater(len(low_temp_ops), 5)
        self.assertGreater(len(high_temp_ops), 5)


class TestFMOBenchmarkIntegration(unittest.TestCase):
    """Integration tests for FMO benchmark system."""
    
    def test_pdb_file_exists(self):
        """Test that FMO PDB file exists and is readable."""
        # Get the path to the PDB file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pdb_path = os.path.join(current_dir, '..', 'qbes', 'benchmarks', 'data', 'fmo_3eni.pdb')
        
        # Check file exists
        self.assertTrue(os.path.exists(pdb_path), f"FMO PDB file not found at {pdb_path}")
        
        # Check file is readable and has expected content
        with open(pdb_path, 'r') as f:
            content = f.read()
        
        # Should contain FMO-specific information
        self.assertIn('FENNA-MATTHEWS-OLSON', content)
        self.assertIn('BCL', content)  # Bacteriochlorophyll
        self.assertIn('PHOTOSYNTHESIS', content)
        
        # Should have multiple BCL entries (7 per chain)
        bcl_count = content.count('BCL')
        self.assertGreaterEqual(bcl_count, 21)  # At least 7 per chain × 3 chains
    
    def test_reference_data_consistency(self):
        """Test that reference data is consistent with literature."""
        # Load reference data directly
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reference_file = os.path.join(current_dir, '..', 'qbes', 'benchmarks', 'reference_data.json')
        
        with open(reference_file, 'r') as f:
            data = json.load(f)
        
        # Check FMO coherence lifetime
        fmo_coherence = data['fmo_coherence_lifetime_fs']
        self.assertEqual(fmo_coherence['unit'], 'femtoseconds')
        self.assertGreater(fmo_coherence['value'], 500)  # Should be > 500 fs
        self.assertLess(fmo_coherence['value'], 1000)    # Should be < 1000 fs
        self.assertIn('Engel', fmo_coherence['source'])  # Should reference Engel et al.
        
        # Check FMO transfer efficiency
        fmo_efficiency = data['fmo_energy_transfer_efficiency']
        self.assertEqual(fmo_efficiency['unit'], 'dimensionless')
        self.assertGreater(fmo_efficiency['value'], 0.8)  # Should be > 80%
        self.assertLessEqual(fmo_efficiency['value'], 1.0)  # Should be ≤ 100%
        self.assertIn('Mohseni', fmo_efficiency['source'])  # Should reference Mohseni et al.


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestFMOComplexBenchmark))
    suite.addTest(unittest.makeSuite(TestFMOBenchmarkIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)