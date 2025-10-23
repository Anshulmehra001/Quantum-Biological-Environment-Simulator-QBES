"""
Performance benchmark tests for QBES.
"""

import pytest
import time
import numpy as np
from qbes.quantum_engine import QuantumEngine
from qbes.analysis import ResultsAnalyzer


@pytest.mark.benchmark
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests for QBES components."""
    
    def test_quantum_engine_performance(self, quantum_engine, simple_quantum_subsystem):
        """Benchmark quantum engine performance."""
        # Test Hamiltonian creation performance
        start_time = time.time()
        
        for _ in range(100):
            hamiltonian = quantum_engine.create_two_level_hamiltonian(2.0, 0.1)
        
        hamiltonian_time = time.time() - start_time
        
        # Should create 100 Hamiltonians in reasonable time
        assert hamiltonian_time < 1.0  # Less than 1 second
        
        # Test state initialization performance
        start_time = time.time()
        
        for _ in range(100):
            state = quantum_engine.initialize_state(simple_quantum_subsystem, "ground")
        
        state_init_time = time.time() - start_time
        
        # Should initialize 100 states in reasonable time
        assert state_init_time < 2.0  # Less than 2 seconds
    
    def test_purity_calculation_performance(self, quantum_engine):
        """Benchmark purity calculation performance."""
        # Create various sized density matrices
        sizes = [2, 4, 8, 16]
        times = []
        
        for size in sizes:
            # Create random density matrix
            random_matrix = np.random.random((size, size)) + 1j * np.random.random((size, size))
            # Make it Hermitian
            random_matrix = (random_matrix + random_matrix.conj().T) / 2
            # Normalize trace
            random_matrix = random_matrix / np.trace(random_matrix)
            
            from qbes.core.data_models import DensityMatrix
            state = DensityMatrix(
                matrix=random_matrix,
                basis_labels=[f"state_{i}" for i in range(size)],
                time=0.0
            )
            
            start_time = time.time()
            
            for _ in range(100):
                purity = quantum_engine.calculate_purity(state)
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        # Performance should scale reasonably with matrix size
        # Larger matrices should take more time, but not exponentially
        assert all(t < 5.0 for t in times)  # All should complete in under 5 seconds
    
    def test_coherence_lifetime_calculation_performance(self):
        """Benchmark coherence lifetime calculation performance."""
        analyzer = ResultsAnalyzer()
        
        # Create trajectory with different lengths
        trajectory_lengths = [10, 50, 100, 200]
        times = []
        
        for length in trajectory_lengths:
            # Create mock trajectory
            states = []
            for i in range(length):
                coherence = np.exp(-i * 0.01)
                matrix = np.array([
                    [0.5, coherence * 0.1j],
                    [-coherence * 0.1j, 0.5]
                ], dtype=complex)
                
                from qbes.core.data_models import DensityMatrix
                state = DensityMatrix(
                    matrix=matrix,
                    basis_labels=["ground", "excited"],
                    time=i * 0.01
                )
                states.append(state)
            
            start_time = time.time()
            
            # Run calculation multiple times
            for _ in range(10):
                lifetime = analyzer.calculate_coherence_lifetime(states)
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        # Should complete in reasonable time
        assert all(t < 10.0 for t in times)
    
    def test_state_evolution_performance(self, quantum_engine, simple_density_matrix, 
                                       two_level_hamiltonian, simple_lindblad_operator):
        """Benchmark quantum state evolution performance."""
        # Test evolution with different time steps
        time_steps = [0.001, 0.01, 0.1]
        times = []
        
        for time_step in time_steps:
            start_time = time.time()
            
            current_state = simple_density_matrix
            
            # Evolve for 10 steps
            for _ in range(10):
                current_state = quantum_engine.evolve_state(
                    current_state,
                    time_step,
                    two_level_hamiltonian,
                    [simple_lindblad_operator]
                )
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        # Should complete evolution in reasonable time
        assert all(t < 30.0 for t in times)  # 30 seconds max for 10 evolution steps
    
    def test_statistical_analysis_performance(self):
        """Benchmark statistical analysis performance."""
        analyzer = ResultsAnalyzer()
        
        # Test with different data sizes
        data_sizes = [100, 1000, 10000]
        times = []
        
        for size in data_sizes:
            # Generate random data
            data = np.random.normal(0, 1, size)
            
            start_time = time.time()
            
            # Run various statistical analyses
            for _ in range(10):
                estimates = analyzer.calculate_uncertainty_estimates(data)
                outliers = analyzer.detect_outliers(data, method="iqr")
            
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        # Should complete statistical analysis in reasonable time
        assert all(t < 5.0 for t in times)
    
    @pytest.mark.parametrize("matrix_size", [2, 4, 8])
    def test_matrix_operations_scaling(self, quantum_engine, matrix_size):
        """Test how matrix operations scale with size."""
        # Create random density matrix of given size
        random_matrix = np.random.random((matrix_size, matrix_size)) + 1j * np.random.random((matrix_size, matrix_size))
        random_matrix = (random_matrix + random_matrix.conj().T) / 2
        random_matrix = random_matrix / np.trace(random_matrix)
        
        from qbes.core.data_models import DensityMatrix
        state = DensityMatrix(
            matrix=random_matrix,
            basis_labels=[f"state_{i}" for i in range(matrix_size)],
            time=0.0
        )
        
        start_time = time.time()
        
        # Perform various operations
        for _ in range(50):
            purity = quantum_engine.calculate_purity(state)
            entropy = quantum_engine.calculate_von_neumann_entropy(state)
            validation = quantum_engine.validate_quantum_state(state)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time regardless of size
        # (for these small matrices)
        assert elapsed_time < 10.0
    
    def test_memory_usage_trajectory(self):
        """Test memory usage with long trajectories."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a long trajectory
        states = []
        for i in range(1000):
            matrix = np.array([
                [0.5, 0.1j * np.exp(-i * 0.001)],
                [-0.1j * np.exp(-i * 0.001), 0.5]
            ], dtype=complex)
            
            from qbes.core.data_models import DensityMatrix
            state = DensityMatrix(
                matrix=matrix,
                basis_labels=["ground", "excited"],
                time=i * 0.001
            )
            states.append(state)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100 MB for 1000 states)
        assert memory_increase < 100.0
    
    def test_numerical_stability_performance(self, quantum_engine):
        """Test performance of numerical stability corrections."""
        # Create matrices that need correction
        matrices_to_correct = []
        
        for _ in range(100):
            # Create slightly non-Hermitian matrix
            matrix = np.array([[0.5, 0.1j], [-0.1j + 1e-10, 0.5]], dtype=complex)
            # Make trace slightly off
            matrix = matrix * 1.001
            matrices_to_correct.append(matrix)
        
        start_time = time.time()
        
        # Test the correction process
        for matrix in matrices_to_correct:
            corrected = quantum_engine._ensure_valid_density_matrix(matrix)
        
        elapsed_time = time.time() - start_time
        
        # Should correct 100 matrices quickly
        assert elapsed_time < 1.0