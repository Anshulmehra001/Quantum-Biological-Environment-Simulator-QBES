#!/usr/bin/env python3
"""
Simple test for benchmark system components.
"""

import sys
import os
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qbes.benchmarks import BenchmarkSystem, BenchmarkResult, TwoLevelSystemBenchmark
from qbes.core.data_models import DensityMatrix, Hamiltonian, LindbladOperator

class MockBenchmark(BenchmarkSystem):
    """Mock benchmark that always passes for testing."""
    
    def __init__(self):
        super().__init__("Mock Benchmark", tolerance=1e-6)
    
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        """Simple analytical solution: exponential decay."""
        return np.exp(-time)
    
    def setup_system(self):
        """Mock system setup."""
        # Create simple 2x2 identity matrix
        hamiltonian_matrix = np.eye(2, dtype=complex)
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix,
            basis_labels=["0", "1"],
            time_dependent=False
        )
        
        lindblad_ops = []
        
        # Simple initial state
        initial_matrix = np.eye(2, dtype=complex)
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=["0", "1"],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def extract_observable(self, state: DensityMatrix) -> float:
        """Mock observable extraction."""
        # Return the analytical solution to ensure test passes
        return self.get_analytical_solution(0.1)  # Mock time

def test_benchmark_components():
    """Test individual benchmark components."""
    print("Testing Benchmark Components")
    print("=" * 40)
    
    # Test BenchmarkResult
    print("1. Testing BenchmarkResult...")
    result = BenchmarkResult(
        system_name="Test",
        test_passed=True,
        numerical_result=1.0,
        analytical_result=1.0,
        relative_error=0.0,
        absolute_error=0.0,
        computation_time=0.1,
        tolerance=1e-6
    )
    print(f"   ✅ BenchmarkResult created: {result.system_name}")
    
    # Test TwoLevelSystemBenchmark analytical solution
    print("2. Testing TwoLevelSystemBenchmark analytical solutions...")
    benchmark = TwoLevelSystemBenchmark()
    
    # Test analytical solutions
    sol_0 = benchmark.get_analytical_solution(0.0)
    sol_pi = benchmark.get_analytical_solution(np.pi)
    
    print(f"   Solution at t=0: {sol_0:.6f} (expected: 0.0)")
    print(f"   Solution at t=π: {sol_pi:.6f} (expected: ~1.0)")
    
    if abs(sol_0 - 0.0) < 1e-10 and abs(sol_pi - 1.0) < 1e-3:
        print("   ✅ Analytical solutions correct")
    else:
        print("   ❌ Analytical solutions incorrect")
        return False
    
    # Test system setup
    print("3. Testing system setup...")
    try:
        hamiltonian, lindblad_ops, initial_state = benchmark.setup_system()
        print(f"   Hamiltonian shape: {hamiltonian.matrix.shape}")
        print(f"   Number of Lindblad operators: {len(lindblad_ops)}")
        print(f"   Initial state shape: {initial_state.matrix.shape}")
        print("   ✅ System setup successful")
    except Exception as e:
        print(f"   ❌ System setup failed: {str(e)}")
        return False
    
    # Test observable extraction
    print("4. Testing observable extraction...")
    try:
        # Create test state with known population
        test_matrix = np.array([[0.7, 0], [0, 0.3]], dtype=complex)
        test_state = DensityMatrix(
            matrix=test_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        observable = benchmark.extract_observable(test_state)
        print(f"   Extracted observable: {observable:.6f} (expected: 0.3)")
        
        if abs(observable - 0.3) < 1e-10:
            print("   ✅ Observable extraction correct")
        else:
            print("   ❌ Observable extraction incorrect")
            return False
    except Exception as e:
        print(f"   ❌ Observable extraction failed: {str(e)}")
        return False
    
    # Test mock benchmark
    print("5. Testing mock benchmark...")
    mock_benchmark = MockBenchmark()
    try:
        # This should pass since it's designed to
        result = mock_benchmark.run_benchmark(final_time=0.1, time_step=0.01)
        print(f"   Mock benchmark result: {result.test_passed}")
        print(f"   Relative error: {result.relative_error:.2e}")
        
        if result.test_passed:
            print("   ✅ Mock benchmark passed")
        else:
            print("   ❌ Mock benchmark failed")
            return False
    except Exception as e:
        print(f"   ❌ Mock benchmark crashed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = test_benchmark_components()
        if success:
            print("\n🎉 All benchmark component tests PASSED!")
            sys.exit(0)
        else:
            print("\n💥 Some benchmark component tests FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Benchmark component tests CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)