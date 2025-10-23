"""
Quick verification test for QBES reorganization and enhancements.

This script quickly verifies all new features are properly installed and working.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from qbes.validation import EnhancedValidator, validate_simulation
        print("  ✅ Enhanced validation module")
    except ImportError as e:
        print(f"  ❌ Enhanced validation module: {e}")
        return False
    
    try:
        from qbes.performance import PerformanceProfiler, profile_simulation, quick_profile
        print("  ✅ Performance profiling module")
    except ImportError as e:
        print(f"  ❌ Performance profiling module: {e}")
        return False
    
    try:
        from qbes.benchmarks.literature import LiteratureBenchmarks, validate_against_literature
        print("  ✅ Literature benchmarks module")
    except ImportError as e:
        print(f"  ❌ Literature benchmarks module: {e}")
        return False
    
    try:
        from qbes import (
            EnhancedValidator,
            PerformanceProfiler,
            LiteratureBenchmarks,
            validate_simulation,
            profile_simulation,
            validate_against_literature
        )
        print("  ✅ All features accessible from main qbes package")
    except ImportError as e:
        print(f"  ❌ Main package imports: {e}")
        return False
    
    return True


def test_validation():
    """Test enhanced validation."""
    print("\nTesting enhanced validation...")
    
    try:
        import numpy as np
        from qbes.validation import EnhancedValidator
        from dataclasses import dataclass
        
        @dataclass
        class DensityMatrix:
            matrix: np.ndarray
            time: float = 0.0
        
        validator = EnhancedValidator()
        
        # Valid density matrix
        rho = DensityMatrix(
            matrix=np.array([[0.5, 0], [0, 0.5]], dtype=complex),
            time=0.0
        )
        
        metrics = validator.validate_density_matrix(rho)
        
        if metrics.density_matrix_valid:
            print("  ✅ Validation working correctly")
            return True
        else:
            print("  ❌ Validation failed unexpectedly")
            return False
            
    except Exception as e:
        print(f"  ❌ Validation test error: {e}")
        return False


def test_profiler():
    """Test performance profiler."""
    print("\nTesting performance profiler...")
    
    try:
        from qbes.performance import PerformanceProfiler
        import time
        
        profiler = PerformanceProfiler("Test")
        profiler.start_session("Quick Test")
        
        with profiler.profile_operation("Test Operation"):
            time.sleep(0.01)
        
        profiler.end_session()
        
        if profiler.sessions:
            print("  ✅ Profiler working correctly")
            return True
        else:
            print("  ❌ Profiler failed to record session")
            return False
            
    except Exception as e:
        print(f"  ❌ Profiler test error: {e}")
        return False


def test_benchmarks():
    """Test literature benchmarks."""
    print("\nTesting literature benchmarks...")
    
    try:
        from qbes.benchmarks.literature import LiteratureBenchmarks
        
        benchmarks = LiteratureBenchmarks()
        
        # Check available benchmarks
        available = benchmarks.list_benchmarks()
        
        if len(available) >= 4:  # Should have at least 4 benchmarks
            print(f"  ✅ Literature benchmarks working ({len(available)} benchmarks available)")
            
            # Get one benchmark
            fmo = benchmarks.get_benchmark('fmo_engel_2007')
            if fmo:
                print(f"  ✅ Can retrieve individual benchmarks")
                return True
            else:
                print("  ❌ Failed to retrieve benchmark")
                return False
        else:
            print(f"  ❌ Not enough benchmarks found ({len(available)})")
            return False
            
    except Exception as e:
        print(f"  ❌ Benchmarks test error: {e}")
        return False


def test_file_organization():
    """Test that files are properly organized."""
    print("\nTesting file organization...")
    
    import os
    
    expected_dirs = [
        'tests/debug',
        'scripts/debug',
        'configs/test',
        'qbes/validation',
        'qbes/performance',
        'qbes/benchmarks/literature'
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def test_documentation():
    """Test that documentation exists."""
    print("\nTesting documentation...")
    
    import os
    
    expected_docs = [
        'qbes/validation/README.md',
        'qbes/performance/README.md',
        'qbes/benchmarks/literature/README.md',
        'PROJECT_STRUCTURE_UPDATE.md',
        'PROJECT_COMPLETION_SUMMARY.md'
    ]
    
    all_exist = True
    for doc_path in expected_docs:
        if os.path.isfile(doc_path):
            print(f"  ✅ {doc_path}")
        else:
            print(f"  ❌ {doc_path} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 80)
    print("QBES VERIFICATION TEST")
    print("=" * 80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Validation", test_validation()))
    results.append(("Profiler", test_profiler()))
    results.append(("Benchmarks", test_benchmarks()))
    results.append(("File Organization", test_file_organization()))
    results.append(("Documentation", test_documentation()))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All tests passed! QBES is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python demo_new_features.py")
        print("  2. Read: PROJECT_STRUCTURE_UPDATE.md")
        print("  3. Explore: qbes/*/README.md files")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
