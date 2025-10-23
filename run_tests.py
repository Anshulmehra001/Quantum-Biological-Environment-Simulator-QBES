#!/usr/bin/env python3
"""
QBES Test Runner
Comprehensive test execution script for QBES
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED ({elapsed_time:.1f}s)")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} - FAILED ({elapsed_time:.1f}s)")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="QBES Test Runner")
    parser.add_argument(
        "--suite", 
        choices=["unit", "integration", "benchmark", "all", "quick"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--parallel", "-n",
        type=int,
        help="Number of parallel processes (requires pytest-xdist)"
    )
    
    args = parser.parse_args()
    
    print("🧬 QBES Test Runner")
    print("=" * 60)
    print(f"Test Suite: {args.suite}")
    print(f"Coverage: {'Enabled' if args.coverage else 'Disabled'}")
    print(f"Verbose: {'Enabled' if args.verbose else 'Disabled'}")
    
    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ pytest not found. Please install with: pip install pytest")
        return 1
    
    # Build base command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])
    
    if args.coverage:
        base_cmd.extend([
            "--cov=qbes",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Test results
    results = []
    
    if args.suite == "quick":
        # Quick test suite - just unit tests, no slow tests
        cmd = base_cmd + ["tests/unit/", "-m", "not slow"]
        results.append(run_command(" ".join(cmd), "Quick Unit Tests"))
        
    elif args.suite == "unit":
        # Unit tests only
        cmd = base_cmd + ["tests/unit/"]
        results.append(run_command(" ".join(cmd), "Unit Tests"))
        
    elif args.suite == "integration":
        # Integration tests only
        cmd = base_cmd + ["tests/integration/"]
        results.append(run_command(" ".join(cmd), "Integration Tests"))
        
    elif args.suite == "benchmark":
        # Benchmark tests only
        cmd = base_cmd + ["tests/benchmarks/"]
        results.append(run_command(" ".join(cmd), "Benchmark Tests"))
        
    elif args.suite == "all":
        # Run all test suites
        
        # 1. Unit tests (fast)
        cmd = base_cmd + ["tests/unit/", "-m", "not slow"]
        results.append(run_command(" ".join(cmd), "Unit Tests (Fast)"))
        
        # 2. Unit tests (slow)
        cmd = base_cmd + ["tests/unit/", "-m", "slow"]
        results.append(run_command(" ".join(cmd), "Unit Tests (Slow)"))
        
        # 3. Integration tests
        cmd = base_cmd + ["tests/integration/"]
        results.append(run_command(" ".join(cmd), "Integration Tests"))
        
        # 4. Benchmark tests
        cmd = base_cmd + ["tests/benchmarks/"]
        results.append(run_command(" ".join(cmd), "Benchmark Tests"))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, result in enumerate(results):
        status = "✅ PASSED" if result else "❌ FAILED"
        test_names = ["Quick Unit Tests", "Unit Tests (Fast)", "Unit Tests (Slow)", 
                     "Integration Tests", "Benchmark Tests"]
        test_name = test_names[i] if i < len(test_names) else f"Test {i+1}"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} test suites passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())