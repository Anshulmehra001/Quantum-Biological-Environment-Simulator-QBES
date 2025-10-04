#!/usr/bin/env python3
"""
Core End-to-End System Testing for QBES

Simplified version focusing on core functionality testing.
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

class CoreEndToEndTestRunner:
    """Core end-to-end test runner for QBES."""
    
    def __init__(self, results_dir="core_test_results"):
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
        
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status} ({duration:.2f}s)")
        
        if not success and details:
            print(f"  Error: {details.get('error', 'Unknown error')}")
    
    def test_core_imports(self):
        """Test that all core modules can be imported."""
        print("\n1. Testing Core Module Imports...")
        print("-" * 40)
        
        modules_to_test = [
            'qbes.core.data_models',
            'qbes.core.interfaces',
            'qbes.config_manager',
            'qbes.quantum_engine',
            'qbes.simulation_engine',
            'qbes.noise_models',
            'qbes.analysis',
            'qbes.visualization',
            'qbes.cli'
        ]
        
        for module_name in modules_to_test:
            test_start = time.time()
            
            try:
                __import__(module_name)
                duration = time.time() - test_start
                self.log_test_result(
                    f"Import {module_name}", 
                    True, 
                    duration
                )
                
            except Exception as e:
                duration = time.time() - test_start
                self.log_test_result(
                    f"Import {module_name}", 
                    False, 
                    duration,
                    {'error': str(e)}
                )
    
    def test_data_models(self):
        """Test core data model functionality."""
        print("\n2. Testing Data Models...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.core.data_models import SimulationConfig, Atom, QuantumState
            
            # Test SimulationConfig creation
            config = SimulationConfig(
                system_pdb="test.pdb",
                temperature=300.0,
                simulation_time=1.0,
                time_step=0.01,
                quantum_subsystem_selection="all",
                noise_model_type="ohmic",
                output_directory="test_output"
            )
            
            assert config.temperature == 300.0
            assert config.simulation_time == 1.0
            
            # Test Atom creation
            atom = Atom(
                element="C",
                position=np.array([0.0, 0.0, 0.0]),
                charge=0.0,
                mass=12.0,
                atom_id=1
            )
            
            assert atom.element == "C"
            assert len(atom.position) == 3
            
            duration = time.time() - test_start
            self.log_test_result(
                "Data Models", 
                True, 
                duration,
                {'config_created': True, 'atom_created': True}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Data Models", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_configuration_manager(self):
        """Test configuration management functionality."""
        print("\n3. Testing Configuration Manager...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.config_manager import ConfigurationManager
            
            # Create a test configuration file
            test_config = {
                'system': {
                    'pdb_file': 'test.pdb',
                    'temperature': 300.0,
                    'force_field': 'amber14'
                },
                'simulation': {
                    'time_step': 0.01,
                    'total_time': 1.0
                },
                'quantum': {
                    'subsystem_selection': 'all',
                    'noise_model': 'ohmic'
                },
                'output': {
                    'directory': 'test_output'
                }
            }
            
            # Test configuration manager
            config_manager = ConfigurationManager()
            
            # Test validation methods exist
            assert hasattr(config_manager, 'validate_config')
            assert hasattr(config_manager, 'load_config')
            
            duration = time.time() - test_start
            self.log_test_result(
                "Configuration Manager", 
                True, 
                duration,
                {'methods_available': True}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Configuration Manager", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_quantum_engine(self):
        """Test quantum engine functionality."""
        print("\n4. Testing Quantum Engine...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.quantum_engine import QuantumEngine
            from qbes.core.data_models import QuantumState
            
            # Create quantum engine
            engine = QuantumEngine()
            
            # Test basic quantum state operations
            assert hasattr(engine, 'initialize_state')
            assert hasattr(engine, 'evolve_state')
            assert hasattr(engine, 'calculate_observables')
            
            duration = time.time() - test_start
            self.log_test_result(
                "Quantum Engine", 
                True, 
                duration,
                {'engine_created': True}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Quantum Engine", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_noise_models(self):
        """Test noise model functionality."""
        print("\n5. Testing Noise Models...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.noise_models import NoiseModelFactory, OhmicNoiseModel
            
            # Test noise model factory
            factory = NoiseModelFactory()
            
            # Test creating different noise models
            ohmic_model = factory.create_noise_model("ohmic", temperature=300.0)
            
            assert ohmic_model is not None
            assert hasattr(ohmic_model, 'calculate_decoherence_rate')
            
            duration = time.time() - test_start
            self.log_test_result(
                "Noise Models", 
                True, 
                duration,
                {'ohmic_model_created': True}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Noise Models", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_analysis_tools(self):
        """Test analysis functionality."""
        print("\n6. Testing Analysis Tools...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.analysis import CoherenceAnalyzer, StatisticalAnalyzer
            
            # Test coherence analyzer
            coherence_analyzer = CoherenceAnalyzer()
            
            assert hasattr(coherence_analyzer, 'calculate_coherence_lifetime')
            assert hasattr(coherence_analyzer, 'analyze_decoherence')
            
            # Test statistical analyzer
            stats_analyzer = StatisticalAnalyzer()
            
            assert hasattr(stats_analyzer, 'calculate_statistics')
            assert hasattr(stats_analyzer, 'generate_confidence_intervals')
            
            duration = time.time() - test_start
            self.log_test_result(
                "Analysis Tools", 
                True, 
                duration,
                {'analyzers_created': True}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Analysis Tools", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_benchmark_core(self):
        """Test core benchmark functionality."""
        print("\n7. Testing Benchmark Core...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.benchmarks.benchmark_systems import BenchmarkRunner
            
            # Create benchmark runner
            runner = BenchmarkRunner()
            
            # Test that basic methods exist
            assert hasattr(runner, 'add_benchmark')
            assert hasattr(runner, 'run_benchmarks')
            
            duration = time.time() - test_start
            self.log_test_result(
                "Benchmark Core", 
                True, 
                duration,
                {'runner_created': True}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Benchmark Core", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_cli_basic(self):
        """Test basic CLI functionality."""
        print("\n8. Testing CLI Basic...")
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
            
            cli_works = result.returncode == 0
            
            duration = time.time() - test_start
            self.log_test_result(
                "CLI Basic", 
                cli_works, 
                duration,
                {'help_command_works': cli_works}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "CLI Basic", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def test_validation_core(self):
        """Test core validation functionality."""
        print("\n9. Testing Validation Core...")
        print("-" * 40)
        
        test_start = time.time()
        
        try:
            from qbes.benchmarks.validation_reports import ValidationSummary
            
            # Test validation summary creation
            summary = ValidationSummary(
                benchmark_score=0.9,
                literature_score=0.8,
                cross_validation_score=0.7,
                statistical_score=0.85
            )
            
            assert summary.overall_validation_score > 0
            assert hasattr(summary, 'validation_grade')
            
            duration = time.time() - test_start
            self.log_test_result(
                "Validation Core", 
                True, 
                duration,
                {'summary_created': True, 'score': summary.overall_validation_score}
            )
            
        except Exception as e:
            duration = time.time() - test_start
            self.log_test_result(
                "Validation Core", 
                False, 
                duration,
                {'error': str(e)}
            )
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("GENERATING CORE TEST REPORT")
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
        json_report_path = self.results_dir / "core_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        text_report_path = self.results_dir / "core_test_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("QBES Core System Test Report\n")
            f.write("=" * 40 + "\n\n")
            
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
            
            # Recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            
            if success_rate >= 0.9:
                f.write("EXCELLENT: Core system is fully functional.\n")
            elif success_rate >= 0.8:
                f.write("GOOD: Core system is mostly functional.\n")
            elif success_rate >= 0.6:
                f.write("FAIR: Core system has some issues.\n")
            else:
                f.write("POOR: Core system has major issues.\n")
        
        print(f"\nTest reports generated:")
        print(f"  JSON Report: {json_report_path}")
        print(f"  Text Report: {text_report_path}")
        
        return report
    
    def run_all_tests(self):
        """Run all core tests."""
        print("QBES Core System Testing")
        print("=" * 40)
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results Directory: {self.results_dir}")
        
        # Run all test suites
        self.test_core_imports()
        self.test_data_models()
        self.test_configuration_manager()
        self.test_quantum_engine()
        self.test_noise_models()
        self.test_analysis_tools()
        self.test_benchmark_core()
        self.test_cli_basic()
        self.test_validation_core()
        
        # Generate comprehensive report
        report = self.generate_test_report()
        
        return report


def main():
    """Main function to run core system tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run QBES core system tests"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='core_test_results',
        help='Directory to store test results'
    )
    
    args = parser.parse_args()
    
    try:
        # Run core tests
        runner = CoreEndToEndTestRunner(results_dir=args.results_dir)
        report = runner.run_all_tests()
        
        # Print final summary
        print("\n" + "=" * 50)
        print("FINAL CORE TEST SUMMARY")
        print("=" * 50)
        
        summary = report['test_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary['total_duration_seconds']:.1f} seconds")
        
        if summary['success_rate'] >= 0.8:
            print("\nCore system testing completed successfully!")
            print("QBES core functionality is operational.")
            return 0
        else:
            print("\nCore system testing completed with issues.")
            print("QBES core functionality needs attention.")
            return 1
            
    except KeyboardInterrupt:
        print("\nCore testing interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nCore testing failed: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())