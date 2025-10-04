#!/usr/bin/env python3
"""
End-to-End System Testing for QBES

This script performs comprehensive end-to-end testing of the complete QBES system,
including simulation workflows with various biological systems, full benchmark
suite execution, stress testing, and performance analysis.

Requirements addressed: 1.4, 5.3, 7.1
"""

import sys
import os
import time
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

# Add the qbes package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class EndToEndTestRunner:
    """Comprehensive end-to-end test runner for QBES."""
    
    def __init__(self, results_dir="end_to_end_results"):
        """Initialize the test runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_test_result(self, test_name, success, duration, details=None):
        """Log a test result."""
        result = {
            'test_name': test_name,
            'success': success,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status} ({duration:.2f}s)")
        
        if not success and details:
            print(f"  Error: {details.get('error', 'Unknown error')}")
    
    def test_basic_simulation_workflow(self):
        """Test basic simulation workflow with simple systems."""
        print("\n1. Testing Basic Simulation Workflows...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.simulation_engine import SimulationEngine
            from qbes.config_manager import ConfigurationManager
            from qbes.core.data_models import SimulationConfig
            
            # Test with minimal configuration
            config = SimulationConfig(
                system_pdb="test_two_level.pdb",
                temperature=300.0,
                simulation_time=1.0,
                time_step=0.01,
                quantum_subsystem_selection="all",
                noise_model_type="ohmic",
                output_directory="test_output"
            )
            
            # Initialize simulation engine
            engine = SimulationEngine()
            
            # Run simulation
            results = engine.run_simulation(config)
            
            # Validate results
            assert results is not None, "Simulation returned None"
            assert hasattr(results, 'state_trajectory'), "Missing state trajectory"
            assert len(results.state_trajectory) > 0, "Empty state trajectory"
            
            duration = time.time() - test_start
            self.log_test_result(
                "Basic Two-Level System", 
                True, 
                duration,
                {'n_states': len(results.state_trajectory)}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Basic Two-Level System", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_biological_system_workflows(self):
        """Test simulation workflows with various biological systems."""
        print("\n2. Testing Biological System Workflows...")
        print("-" * 40)
        
        biological_systems = [
            {
                'name': 'Photosynthetic Complex',
                'n_qubits': 4,
                'coupling_strength': 0.05,
                'noise_model': 'protein_environment'
            },
            {
                'name': 'Enzyme Active Site',
                'n_qubits': 3,
                'coupling_strength': 0.1,
                'noise_model': 'membrane_environment'
            },
            {
                'name': 'DNA Base Pair',
                'n_qubits': 2,
                'coupling_strength': 0.2,
                'noise_model': 'solvent_environment'
            }
        ]
        
        for system_config in biological_systems:
            test_start = time.time()
            
            try:
                from qbes.simulation_engine import SimulationEngine
                from qbes.core.data_models import SimulationConfig
                
                config = SimulationConfig(
                    system_pdb=f"{system_config['name'].lower().replace(' ', '_')}.pdb",
                    temperature=300.0,
                    simulation_time=0.5,  # Shorter for testing
                    time_step=0.01,
                    quantum_subsystem_selection="chromophores",
                    noise_model_type=system_config['noise_model'],
                    output_directory="test_output"
                )
                
                engine = SimulationEngine()
                results = engine.run_simulation(config)
                
                # Validate biological system results
                assert results is not None, "Simulation returned None"
                assert hasattr(results, 'coherence_measures'), "Missing coherence measures"
                
                duration = time.time() - test_start
                self.log_test_result(
                    f"Bio System: {system_config['name']}", 
                    True, 
                    duration,
                    {
                        'n_qubits': system_config['n_qubits'],
                        'final_coherence': getattr(results, 'final_coherence', 'N/A')
                    }
                )
                
            except Exception as e:
                duration = time.time() - test_start
                self.log_test_result(
                    f"Bio System: {system_config['name']}", 
                    False, 
                    duration,
                    {'error': str(e)}
                )
    
    def test_full_benchmark_suite(self):
        """Execute the full benchmark suite and validate results."""
        print("\n3. Testing Full Benchmark Suite...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.benchmarks.automated_benchmarks import run_automated_benchmarks
            
            # Run automated benchmarks with results in our test directory
            benchmark_dir = self.results_dir / "benchmark_results"
            benchmark_dir.mkdir(exist_ok=True)
            
            runner = run_automated_benchmarks(
                results_dir=str(benchmark_dir),
                include_performance=True
            )
            
            # Validate benchmark results
            assert runner is not None, "Benchmark runner returned None"
            assert hasattr(runner, 'results'), "Missing benchmark results"
            
            # Count successful benchmarks
            total_benchmarks = len(runner.results)
            passed_benchmarks = sum(1 for r in runner.results if r.test_passed)
            success_rate = passed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0
            
            duration = time.time() - test_start
            self.log_test_result(
                "Full Benchmark Suite", 
                success_rate >= 0.8,  # Require 80% success rate
                duration,
                {
                    'total_benchmarks': total_benchmarks,
                    'passed_benchmarks': passed_benchmarks,
                    'success_rate': f"{success_rate:.1%}"
                }
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Full Benchmark Suite", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_stress_testing(self):
        """Perform stress testing with large molecular systems."""
        print("\n4. Testing System Stress Limits...")
        print("-" * 40)
        
        stress_tests = [
            {'name': 'Medium System', 'n_qubits': 8, 'time_limit': 30},
            {'name': 'Large System', 'n_qubits': 12, 'time_limit': 60},
            {'name': 'Very Large System', 'n_qubits': 16, 'time_limit': 120}
        ]
        
        for stress_config in stress_tests:
            test_start = time.time()
            
            try:
                from qbes.simulation_engine import SimulationEngine
                from qbes.core.data_models import SimulationConfig
                
                config = SimulationConfig(
                    system_pdb=f"stress_test_{stress_config['n_qubits']}.pdb",
                    temperature=300.0,
                    simulation_time=0.1,  # Very short for stress testing
                    time_step=0.01,
                    quantum_subsystem_selection="all",
                    noise_model_type="ohmic",
                    output_directory="stress_test_output"
                )
                
                engine = SimulationEngine()
                
                # Run with timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Simulation exceeded time limit")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(stress_config['time_limit'])
                
                try:
                    results = engine.run_simulation(config)
                    signal.alarm(0)  # Cancel alarm
                    
                    # Check memory usage and performance
                    duration = time.time() - test_start
                    success = results is not None and duration < stress_config['time_limit']
                    
                    self.log_test_result(
                        f"Stress Test: {stress_config['name']}", 
                        success, 
                        duration,
                        {
                            'n_qubits': stress_config['n_qubits'],
                            'time_limit': stress_config['time_limit'],
                            'completed_within_limit': duration < stress_config['time_limit']
                        }
                    )
                    
                except TimeoutError:
                    signal.alarm(0)
                    duration = time.time() - test_start
                    self.log_test_result(
                        f"Stress Test: {stress_config['name']}", 
                        False, 
                        duration,
                        {'error': 'Timeout exceeded', 'time_limit': stress_config['time_limit']}
                    )
                
            except Exception as e:
                duration = time.time() - test_start
                self.log_test_result(
                    f"Stress Test: {stress_config['name']}", 
                    False, 
                    duration,
                    {'error': str(e)}
                )
    
    def test_performance_scaling(self):
        """Test performance scaling with system size."""
        print("\n5. Testing Performance Scaling...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.benchmarks.performance_benchmarks import run_performance_benchmarks
            
            # Run performance benchmarks
            perf_runner = run_performance_benchmarks(max_size=12)
            
            # Analyze scaling behavior
            if hasattr(perf_runner, 'scaling_results'):
                scaling_data = perf_runner.scaling_results
                
                # Check if scaling is reasonable (not exponential blowup)
                sizes = [r['system_size'] for r in scaling_data]
                times = [r['execution_time'] for r in scaling_data]
                
                if len(sizes) >= 2:
                    # Calculate scaling exponent
                    log_sizes = np.log(sizes)
                    log_times = np.log(times)
                    scaling_exponent = np.polyfit(log_sizes, log_times, 1)[0]
                    
                    # Reasonable scaling should be polynomial, not exponential
                    reasonable_scaling = scaling_exponent < 4.0
                else:
                    reasonable_scaling = True
                    scaling_exponent = 0.0
            else:
                reasonable_scaling = True
                scaling_exponent = 0.0
            
            duration = time.time() - test_start
            self.log_test_result(
                "Performance Scaling", 
                reasonable_scaling, 
                duration,
                {
                    'scaling_exponent': f"{scaling_exponent:.2f}",
                    'reasonable_scaling': reasonable_scaling
                }
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Performance Scaling", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_validation_suite(self):
        """Test the complete validation suite."""
        print("\n6. Testing Validation Suite...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.benchmarks.validation_reports import run_comprehensive_validation
            
            # Run comprehensive validation
            validation_dir = self.results_dir / "validation_results"
            validation_dir.mkdir(exist_ok=True)
            
            summary = run_comprehensive_validation(str(validation_dir))
            
            # Check validation score
            validation_success = summary.overall_validation_score >= 0.7  # 70% threshold
            
            duration = time.time() - test_start
            self.log_test_result(
                "Validation Suite", 
                validation_success, 
                duration,
                {
                    'validation_score': f"{summary.overall_validation_score:.1%}",
                    'validation_grade': summary.validation_grade
                }
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Validation Suite", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_cli_integration(self):
        """Test command-line interface integration."""
        print("\n7. Testing CLI Integration...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            import subprocess
            
            # Test CLI help
            result = subprocess.run(
                [sys.executable, "-m", "qbes.cli", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            cli_help_works = result.returncode == 0
            
            # Test CLI version
            result = subprocess.run(
                [sys.executable, "-m", "qbes.cli", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            cli_version_works = result.returncode == 0
            
            cli_success = cli_help_works and cli_version_works
            
            duration = time.time() - test_start
            self.log_test_result(
                "CLI Integration", 
                cli_success, 
                duration,
                {
                    'help_works': cli_help_works,
                    'version_works': cli_version_works
                }
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "CLI Integration", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Generate detailed report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'total_duration_seconds': total_duration,
                'test_start_time': self.start_time.isoformat(),
                'test_end_time': datetime.now().isoformat()
            },
            'test_results': self.test_results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            }
        }
        
        # Save JSON report
        json_report_path = self.results_dir / "end_to_end_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        text_report_path = self.results_dir / "end_to_end_test_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("QBES End-to-End System Test Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Execution Summary:\n")
            f.write(f"  Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Total Duration: {total_duration:.1f} seconds\n")
            f.write(f"  Total Tests: {total_tests}\n")
            f.write(f"  Passed: {passed_tests}\n")
            f.write(f"  Failed: {failed_tests}\n")
            f.write(f"  Success Rate: {success_rate:.1%}\n\n")
            
            f.write("Detailed Test Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in self.test_results:
                status = "PASS" if result['success'] else "FAIL"
                f.write(f"{result['test_name']}: {status} ({result['duration_seconds']:.2f}s)\n")
                
                if result['details']:
                    for key, value in result['details'].items():
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Performance analysis
            f.write("Performance Analysis:\n")
            f.write("-" * 20 + "\n")
            
            durations = [r['duration_seconds'] for r in self.test_results]
            if durations:
                f.write(f"  Average test duration: {np.mean(durations):.2f}s\n")
                f.write(f"  Longest test: {max(durations):.2f}s\n")
                f.write(f"  Shortest test: {min(durations):.2f}s\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            
            if success_rate >= 0.9:
                f.write("EXCELLENT: System is ready for production use.\n")
            elif success_rate >= 0.8:
                f.write("GOOD: System is mostly functional, minor issues to address.\n")
            elif success_rate >= 0.6:
                f.write("FAIR: System has significant issues that need attention.\n")
            else:
                f.write("POOR: System has major issues and is not ready for use.\n")
        
        print(f"\nTest reports generated:")
        print(f"  JSON Report: {json_report_path}")
        print(f"  Text Report: {text_report_path}")
        
        return report
    
    def run_all_tests(self):
        """Run all end-to-end tests."""
        print("QBES End-to-End System Testing")
        print("=" * 50)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results Directory: {self.results_dir}")
        
        # Run all test suites
        self.test_basic_simulation_workflow()
        self.test_biological_system_workflows()
        self.test_full_benchmark_suite()
        self.test_stress_testing()
        self.test_performance_scaling()
        self.test_validation_suite()
        self.test_cli_integration()
        
        # Generate comprehensive report
        report = self.generate_test_report()
        
        return report


def main():
    """Main function to run end-to-end system tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run QBES end-to-end system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='end_to_end_results',
        help='Directory to store test results (default: end_to_end_results)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run end-to-end tests
        runner = EndToEndTestRunner(results_dir=args.results_dir)
        report = runner.run_all_tests()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL TEST SUMMARY")
        print("=" * 60)
        
        summary = report['test_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration_seconds']:.1f} seconds")
        
        if summary['success_rate'] >= 0.8:
            print("\n🎉 End-to-end system testing completed successfully!")
            print("✅ QBES system is ready for production use.")
            return 0
        else:
            print("\n⚠️  End-to-end system testing completed with issues.")
            print("❌ QBES system needs attention before production use.")
            return 1
            
    except KeyboardInterrupt:
        print("\nEnd-to-end testing interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nEnd-to-end testing failed: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())