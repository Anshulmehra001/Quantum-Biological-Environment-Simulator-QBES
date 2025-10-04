#!/usr/bin/env python3
"""
QBES Benchmark Runner Script

Simple command-line interface for running QBES benchmark suites.
"""

import argparse
import sys
import os

# Add the current directory to Python path so we can import qbes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from qbes.benchmarks import (
        run_quick_benchmarks,
        run_automated_benchmarks
    )
    _quick_benchmarks_available = True
    _automated_benchmarks_available = True
except ImportError:
    _quick_benchmarks_available = False
    _automated_benchmarks_available = False

try:
    from qbes.benchmarks import run_performance_benchmarks
    _performance_benchmarks_available = True
except ImportError:
    _performance_benchmarks_available = False


def main():
    """Main benchmark runner function."""
    parser = argparse.ArgumentParser(
        description="Run QBES benchmark suites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmarks.py --quick          # Run quick validation benchmarks
  python run_benchmarks.py --performance   # Run performance scaling tests
  python run_benchmarks.py --full          # Run complete benchmark suite
  python run_benchmarks.py --all           # Run all benchmark types
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick benchmark suite for validation'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance scaling benchmarks'
    )
    
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run full automated benchmark suite'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all benchmark types'
    )
    
    parser.add_argument(
        '--max-size', 
        type=int, 
        default=16,
        help='Maximum system size for performance tests (default: 16)'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='benchmark_results',
        help='Directory to store benchmark results (default: benchmark_results)'
    )
    
    args = parser.parse_args()
    
    # If no specific benchmark type is specified, run quick benchmarks
    if not any([args.quick, args.performance, args.full, args.all]):
        args.quick = True
    
    try:
        if args.quick or args.all:
            if _quick_benchmarks_available:
                print("Running Quick Benchmark Suite...")
                print("=" * 50)
                runner = run_quick_benchmarks()
                
                # Print summary
                total_tests = len(runner.results)
                passed_tests = sum(1 for r in runner.results if r.test_passed)
                print(f"\nQuick Benchmarks Summary:")
                print(f"  Total Tests: {total_tests}")
                print(f"  Passed: {passed_tests}")
                print(f"  Failed: {total_tests - passed_tests}")
                print(f"  Success Rate: {(passed_tests/total_tests)*100:.1f}%")
                print()
            else:
                print("Quick benchmarks not available.")
        
        if args.performance or args.all:
            if _performance_benchmarks_available:
                print("Running Performance Benchmarks...")
                print("=" * 50)
                perf_runner = run_performance_benchmarks(max_size=args.max_size)
                print()
            else:
                print("Performance benchmarks not available.")
        
        if args.full or args.all:
            if _automated_benchmarks_available:
                print("Running Full Automated Benchmark Suite...")
                print("=" * 50)
                auto_runner = run_automated_benchmarks(
                    results_dir=args.results_dir,
                    include_performance=True
                )
                print()
            else:
                print("Automated benchmarks not available.")
        
        print("Available benchmarks completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()