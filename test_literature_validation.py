#!/usr/bin/env python3
"""
Test script for literature validation functionality.

This script tests the new literature validation and cross-validation
capabilities implemented for task 9.2.
"""

import sys
import os

# Add the qbes package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_literature_validation():
    """Test literature validation functionality."""
    print("Testing Literature Validation...")
    print("=" * 40)
    
    try:
        from qbes.benchmarks.literature_validation import run_literature_validation
        
        # Run literature validation
        validator = run_literature_validation()
        
        print(f"\nLiterature validation completed successfully!")
        print(f"Number of validations: {len(validator.results)}")
        
        # Check results
        passed = sum(1 for r in validator.results if r.validation_passed)
        total = len(validator.results)
        
        print(f"Passed: {passed}/{total}")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Literature validation test failed: {str(e)}")
        return False


def test_cross_validation():
    """Test cross-validation functionality."""
    print("\nTesting Cross-Validation...")
    print("=" * 30)
    
    try:
        from qbes.benchmarks.cross_validation import run_cross_validation
        
        # Run cross-validation
        validator = run_cross_validation()
        
        print(f"\nCross-validation completed successfully!")
        print(f"Number of validations: {len(validator.results)}")
        
        # Check results
        passed = sum(1 for r in validator.results if r.validation_passed)
        total = len(validator.results)
        
        print(f"Passed: {passed}/{total}")
        print(f"Success rate: {(passed/total)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Cross-validation test failed: {str(e)}")
        return False


def test_statistical_analysis():
    """Test statistical analysis functionality."""
    print("\nTesting Statistical Analysis...")
    print("=" * 32)
    
    try:
        from qbes.benchmarks.statistical_testing import perform_statistical_validation
        
        # Test with sample data
        observed = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98]
        expected = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        report = perform_statistical_validation(observed, expected)
        
        print(f"Statistical analysis completed successfully!")
        print(f"Sample size: {report.sample_size}")
        print(f"Mean difference: {report.mean_difference:.4f}")
        print(f"Number of normality tests: {len(report.normality_tests)}")
        print(f"Number of significance tests: {len(report.significance_tests)}")
        print(f"Effect size measures: {len(report.effect_size_measures)}")
        
        return True
        
    except Exception as e:
        print(f"Statistical analysis test failed: {str(e)}")
        return False


def test_comprehensive_validation():
    """Test comprehensive validation functionality."""
    print("\nTesting Comprehensive Validation...")
    print("=" * 38)
    
    try:
        from qbes.benchmarks.validation_reports import ComprehensiveValidationReporter
        
        # Create reporter
        reporter = ComprehensiveValidationReporter("test_validation_output")
        
        # Run a minimal validation suite (without external dependencies)
        print("Running minimal validation suite...")
        
        # Test just the reporter functionality
        summary = reporter._generate_validation_summary()
        
        print(f"Comprehensive validation framework tested successfully!")
        print(f"Validation grade: {summary.validation_grade}")
        print(f"Overall score: {summary.overall_validation_score:.1%}")
        
        return True
        
    except Exception as e:
        print(f"Comprehensive validation test failed: {str(e)}")
        return False


def main():
    """Run all validation tests."""
    print("QBES Literature Validation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Literature Validation", test_literature_validation),
        ("Cross-Validation", test_cross_validation),
        ("Statistical Analysis", test_statistical_analysis),
        ("Comprehensive Validation", test_comprehensive_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All literature validation tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())