#!/usr/bin/env python3
"""
Simple test script for benchmark runner functionality.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qbes.benchmarks import BenchmarkRunner, TwoLevelSystemBenchmark

def test_benchmark_runner():
    """Test the benchmark runner with a simple benchmark."""
    print("Testing QBES Benchmark Runner")
    print("=" * 40)
    
    # Create runner and add benchmark
    runner = BenchmarkRunner()
    benchmark = TwoLevelSystemBenchmark(tolerance=1e-3)  # Relaxed tolerance for testing
    runner.add_benchmark(benchmark)
    
    print(f"Added benchmark: {benchmark.name}")
    print(f"Number of benchmarks: {len(runner.benchmarks)}")
    
    # Run benchmarks with short time for quick test
    print("\nRunning benchmark...")
    results = runner.run_all_benchmarks(final_time=0.1, time_step=0.01)
    
    # Check results
    if results:
        result = results[0]
        print(f"\nResults:")
        print(f"  Test Passed: {result.test_passed}")
        print(f"  Numerical Result: {result.numerical_result:.6f}")
        print(f"  Analytical Result: {result.analytical_result:.6f}")
        print(f"  Relative Error: {result.relative_error:.2e}")
        print(f"  Computation Time: {result.computation_time:.3f}s")
        
        if result.error_message:
            print(f"  Error Message: {result.error_message}")
        
        return result.test_passed
    else:
        print("No results returned!")
        return False

if __name__ == "__main__":
    try:
        success = test_benchmark_runner()
        if success:
            print("\n✅ Benchmark runner test PASSED!")
            sys.exit(0)
        else:
            print("\n❌ Benchmark runner test FAILED!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Benchmark runner test CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)