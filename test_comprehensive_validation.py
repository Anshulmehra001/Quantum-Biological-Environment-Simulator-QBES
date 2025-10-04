#!/usr/bin/env python3
"""
Comprehensive test for the complete validation suite.

This script tests the full validation functionality implemented for task 9.2,
including literature validation, cross-validation, statistical testing,
and comprehensive reporting.
"""

import sys
import os
import tempfile
import shutil

# Add the qbes package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_full_validation_suite():
    """Test the complete validation suite."""
    print("Testing Complete Validation Suite...")
    print("=" * 40)
    
    try:
        from qbes.benchmarks.validation_reports import run_comprehensive_validation
        
        # Create temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Running validation suite in: {temp_dir}")
            
            # Run comprehensive validation
            summary = run_comprehensive_validation(temp_dir)
            
            print(f"\nValidation suite completed!")
            print(f"Overall score: {summary.overall_validation_score:.1%}")
            print(f"Grade: {summary.validation_grade}")
            
            # Check that files were created
            expected_files = [
                "validation_summary.json",
                "comprehensive_validation_report.txt"
            ]
            
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                if os.path.exists(filepath):
                    print(f"✅ Created: {filename}")
                else:
                    print(f"❌ Missing: {filename}")
                    return False
            
            # Read and display part of the comprehensive report
            report_path = os.path.join(temp_dir, "comprehensive_validation_report.txt")
            with open(report_path, 'r', encoding='utf-8') as f:
                report_lines = f.readlines()
            
            print(f"\nReport preview (first 20 lines):")
            print("-" * 40)
            for line in report_lines[:20]:
                print(line.rstrip())
            
            return True
            
    except Exception as e:
        print(f"Comprehensive validation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual validation components."""
    print("\nTesting Individual Components...")
    print("=" * 35)
    
    results = {}
    
    # Test literature validation
    try:
        from qbes.benchmarks.literature_validation import LiteratureValidator
        
        validator = LiteratureValidator()
        validator.add_standard_datasets()
        
        print(f"✅ Literature validation: {len(validator.datasets)} datasets loaded")
        results['literature'] = True
        
    except Exception as e:
        print(f"❌ Literature validation failed: {str(e)}")
        results['literature'] = False
    
    # Test cross-validation
    try:
        from qbes.benchmarks.cross_validation import CrossValidator
        
        validator = CrossValidator()
        validator.add_standard_interfaces()
        availability = validator.check_available_packages()
        
        available_count = sum(1 for avail in availability.values() if avail)
        print(f"✅ Cross-validation: {available_count} packages available")
        results['cross_validation'] = True
        
    except Exception as e:
        print(f"❌ Cross-validation failed: {str(e)}")
        results['cross_validation'] = False
    
    # Test statistical testing
    try:
        from qbes.benchmarks.statistical_testing import StatisticalTester
        
        tester = StatisticalTester()
        
        # Test with sample data
        import numpy as np
        data = np.random.normal(0, 1, 20)
        normality_results = tester.test_normality(data)
        
        print(f"✅ Statistical testing: {len(normality_results)} normality tests")
        results['statistical'] = True
        
    except Exception as e:
        print(f"❌ Statistical testing failed: {str(e)}")
        results['statistical'] = False
    
    # Test benchmark systems
    try:
        from qbes.benchmarks.benchmark_systems import BenchmarkRunner
        
        runner = BenchmarkRunner()
        runner.add_standard_benchmarks()
        
        print(f"✅ Benchmark systems: {len(runner.benchmarks)} benchmarks loaded")
        results['benchmarks'] = True
        
    except Exception as e:
        print(f"❌ Benchmark systems failed: {str(e)}")
        results['benchmarks'] = False
    
    return results


def test_data_export_import():
    """Test data export and import functionality."""
    print("\nTesting Data Export/Import...")
    print("=" * 30)
    
    try:
        from qbes.benchmarks.literature_validation import LiteratureValidator
        
        # Create validator and run validation
        validator = LiteratureValidator()
        validator.add_standard_datasets()
        validator.validate_all_datasets()
        
        # Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, "test_results.json")
            validator.save_validation_results(export_path)
            
            if os.path.exists(export_path):
                print("✅ Export successful")
                
                # Test import
                new_validator = LiteratureValidator()
                new_validator.load_validation_results(export_path)
                
                if len(new_validator.results) == len(validator.results):
                    print("✅ Import successful")
                    return True
                else:
                    print("❌ Import failed: result count mismatch")
                    return False
            else:
                print("❌ Export failed: file not created")
                return False
                
    except Exception as e:
        print(f"❌ Data export/import failed: {str(e)}")
        return False


def test_statistical_robustness():
    """Test statistical analysis with various data patterns."""
    print("\nTesting Statistical Robustness...")
    print("=" * 35)
    
    try:
        from qbes.benchmarks.statistical_testing import perform_statistical_validation
        import numpy as np
        
        test_cases = [
            ("Perfect match", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            ("Small differences", [1.0, 2.0, 3.0], [1.01, 2.02, 2.98]),
            ("Large differences", [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]),
            ("Random data", np.random.normal(5, 1, 10), np.random.normal(5.1, 1, 10))
        ]
        
        for case_name, observed, expected in test_cases:
            try:
                report = perform_statistical_validation(list(observed), list(expected))
                
                print(f"✅ {case_name}: {report.sample_size} samples, "
                      f"{len(report.significance_tests)} tests")
                
            except Exception as e:
                print(f"❌ {case_name} failed: {str(e)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Statistical robustness test failed: {str(e)}")
        return False


def main():
    """Run comprehensive validation tests."""
    print("QBES Comprehensive Validation Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Individual components
    component_results = test_individual_components()
    all_components_pass = all(component_results.values())
    test_results.append(("Individual Components", all_components_pass))
    
    # Test 2: Data export/import
    export_import_pass = test_data_export_import()
    test_results.append(("Data Export/Import", export_import_pass))
    
    # Test 3: Statistical robustness
    statistical_pass = test_statistical_robustness()
    test_results.append(("Statistical Robustness", statistical_pass))
    
    # Test 4: Full validation suite
    full_suite_pass = test_full_validation_suite()
    test_results.append(("Full Validation Suite", full_suite_pass))
    
    # Summary
    print("\n" + "=" * 50)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} test suites passed")
    
    if passed == len(test_results):
        print("\n🎉 All comprehensive validation tests passed!")
        print("✅ Task 9.2 implementation is complete and functional!")
        return 0
    else:
        print("\n⚠️  Some test suites failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())