#!/usr/bin/env python3
"""
Comprehensive Test Report Generator for QBES End-to-End Testing

This script generates a comprehensive report of all end-to-end testing
performed for task 12.1, including working functionality, identified issues,
and recommendations.
"""

import sys
import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Add the qbes package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ComprehensiveTestReporter:
    """Generates comprehensive test reports for QBES."""
    
    def __init__(self, results_dir="comprehensive_test_results"):
        """Initialize the test reporter."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.test_summary = {
            'working_functionality': [],
            'identified_issues': [],
            'performance_metrics': {},
            'validation_results': {},
            'recommendations': []
        }
        
    def test_core_functionality(self):
        """Test and document core functionality that works."""
        print("Testing Core Functionality...")
        print("-" * 40)
        
        working_features = []
        
        # Test 1: Module imports
        try:
            import qbes.core.data_models
            import qbes.core.interfaces
            import qbes.config_manager
            import qbes.quantum_engine
            import qbes.simulation_engine
            import qbes.noise_models
            import qbes.analysis
            import qbes.visualization
            import qbes.cli
            
            working_features.append({
                'feature': 'Core Module Imports',
                'status': 'WORKING',
                'description': 'All core QBES modules can be imported successfully'
            })
            print("✅ Core module imports: WORKING")
            
        except Exception as e:
            working_features.append({
                'feature': 'Core Module Imports',
                'status': 'PARTIAL',
                'description': f'Some import issues: {str(e)}'
            })
            print(f"⚠️  Core module imports: PARTIAL - {str(e)}")
        
        # Test 2: Data models
        try:
            from qbes.core.data_models import SimulationConfig, Atom
            import numpy as np
            
            config = SimulationConfig(
                system_pdb="test.pdb",
                temperature=300.0,
                simulation_time=1.0,
                time_step=0.01,
                quantum_subsystem_selection="all",
                noise_model_type="ohmic",
                output_directory="test_output"
            )
            
            atom = Atom(
                element="C",
                position=np.array([0.0, 0.0, 0.0]),
                charge=0.0,
                mass=12.0,
                atom_id=1
            )
            
            working_features.append({
                'feature': 'Data Models',
                'status': 'WORKING',
                'description': 'SimulationConfig and Atom classes work correctly'
            })
            print("✅ Data models: WORKING")
            
        except Exception as e:
            working_features.append({
                'feature': 'Data Models',
                'status': 'ISSUES',
                'description': f'Data model issues: {str(e)}'
            })
            print(f"❌ Data models: ISSUES - {str(e)}")
        
        # Test 3: Benchmark system
        try:
            from qbes.benchmarks.benchmark_systems import run_quick_benchmarks
            
            runner = run_quick_benchmarks()
            
            working_features.append({
                'feature': 'Benchmark System',
                'status': 'WORKING',
                'description': f'Quick benchmarks run successfully with {len(runner.results)} tests'
            })
            print(f"✅ Benchmark system: WORKING ({len(runner.results)} tests)")
            
        except Exception as e:
            working_features.append({
                'feature': 'Benchmark System',
                'status': 'ISSUES',
                'description': f'Benchmark issues: {str(e)}'
            })
            print(f"❌ Benchmark system: ISSUES - {str(e)}")
        
        # Test 4: Validation suite
        try:
            from qbes.benchmarks.validation_reports import run_comprehensive_validation
            import tempfile
            
            with tempfile.TemporaryDirectory() as temp_dir:
                summary = run_comprehensive_validation(temp_dir)
                
                working_features.append({
                    'feature': 'Validation Suite',
                    'status': 'WORKING',
                    'description': f'Comprehensive validation works with {summary.overall_validation_score:.1%} score'
                })
                print(f"✅ Validation suite: WORKING ({summary.overall_validation_score:.1%} score)")
                
                self.test_summary['validation_results'] = {
                    'overall_score': summary.overall_validation_score,
                    'grade': summary.validation_grade
                }
            
        except Exception as e:
            working_features.append({
                'feature': 'Validation Suite',
                'status': 'ISSUES',
                'description': f'Validation issues: {str(e)}'
            })
            print(f"❌ Validation suite: ISSUES - {str(e)}")
        
        # Test 5: CLI functionality
        try:
            result = subprocess.run(
                [sys.executable, "-m", "qbes.cli", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                working_features.append({
                    'feature': 'Command Line Interface',
                    'status': 'WORKING',
                    'description': 'CLI help command works correctly'
                })
                print("✅ CLI functionality: WORKING")
            else:
                working_features.append({
                    'feature': 'Command Line Interface',
                    'status': 'ISSUES',
                    'description': f'CLI returned error code {result.returncode}'
                })
                print(f"❌ CLI functionality: ISSUES - error code {result.returncode}")
            
        except Exception as e:
            working_features.append({
                'feature': 'Command Line Interface',
                'status': 'ISSUES',
                'description': f'CLI issues: {str(e)}'
            })
            print(f"❌ CLI functionality: ISSUES - {str(e)}")
        
        self.test_summary['working_functionality'] = working_features
        return working_features
    
    def identify_system_issues(self):
        """Identify and document system issues."""
        print("\nIdentifying System Issues...")
        print("-" * 40)
        
        issues = []
        
        # Issue 1: Missing dependencies
        missing_deps = []
        try:
            import openmm
        except ImportError:
            missing_deps.append('OpenMM')
        
        try:
            import mdtraj
        except ImportError:
            missing_deps.append('MDTraj')
        
        if missing_deps:
            issues.append({
                'issue': 'Missing Optional Dependencies',
                'severity': 'MEDIUM',
                'description': f'Optional dependencies not available: {", ".join(missing_deps)}',
                'impact': 'MD functionality is limited',
                'recommendation': 'Install missing dependencies for full functionality'
            })
            print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        
        # Issue 2: API inconsistencies
        api_issues = []
        try:
            from qbes.quantum_engine import QuantumEngine
            engine = QuantumEngine()
            
            # Check for expected methods
            if not hasattr(engine, 'initialize_state'):
                api_issues.append('QuantumEngine.initialize_state method missing')
            if not hasattr(engine, 'evolve_state'):
                api_issues.append('QuantumEngine.evolve_state method missing')
                
        except Exception as e:
            api_issues.append(f'QuantumEngine instantiation failed: {str(e)}')
        
        if api_issues:
            issues.append({
                'issue': 'API Inconsistencies',
                'severity': 'HIGH',
                'description': 'Some expected API methods are missing or inconsistent',
                'impact': 'End-to-end simulation workflows cannot complete',
                'recommendation': 'Review and standardize API interfaces',
                'details': api_issues
            })
            print(f"❌ API issues found: {len(api_issues)} problems")
        
        # Issue 3: Configuration compatibility
        try:
            from qbes.simulation_engine import SimulationEngine
            from qbes.core.data_models import SimulationConfig
            
            config = SimulationConfig(
                system_pdb="test_two_level.pdb",
                temperature=300.0,
                simulation_time=0.1,
                time_step=0.01,
                quantum_subsystem_selection="all",
                noise_model_type="ohmic",
                output_directory="test_output"
            )
            
            engine = SimulationEngine()
            # Try to run simulation - this will likely fail but we can catch the specific error
            try:
                results = engine.run_simulation(config)
            except Exception as sim_error:
                issues.append({
                    'issue': 'Simulation Engine Integration',
                    'severity': 'HIGH',
                    'description': 'Simulation engine cannot complete full workflows',
                    'impact': 'End-to-end simulations fail',
                    'recommendation': 'Debug simulation engine integration',
                    'error_details': str(sim_error)
                })
                print(f"❌ Simulation integration issue: {str(sim_error)}")
                
        except Exception as e:
            issues.append({
                'issue': 'Configuration System',
                'severity': 'MEDIUM',
                'description': f'Configuration system has issues: {str(e)}',
                'impact': 'Cannot properly configure simulations',
                'recommendation': 'Review configuration management system'
            })
            print(f"⚠️  Configuration issue: {str(e)}")
        
        self.test_summary['identified_issues'] = issues
        return issues
    
    def assess_performance_characteristics(self):
        """Assess system performance characteristics."""
        print("\nAssessing Performance Characteristics...")
        print("-" * 40)
        
        performance_metrics = {}
        
        # Test import performance
        import time
        
        start_time = time.time()
        try:
            import qbes
            import_time = time.time() - start_time
            performance_metrics['import_time'] = import_time
            print(f"✅ Import performance: {import_time:.3f}s")
        except Exception as e:
            performance_metrics['import_time'] = None
            print(f"❌ Import performance test failed: {str(e)}")
        
        # Test basic operations performance
        start_time = time.time()
        try:
            from qbes.core.data_models import SimulationConfig
            
            # Create multiple configurations to test performance
            configs = []
            for i in range(100):
                config = SimulationConfig(
                    system_pdb=f"test_{i}.pdb",
                    temperature=300.0,
                    simulation_time=1.0,
                    time_step=0.01,
                    quantum_subsystem_selection="all",
                    noise_model_type="ohmic",
                    output_directory=f"output_{i}"
                )
                configs.append(config)
            
            config_creation_time = time.time() - start_time
            performance_metrics['config_creation_time'] = config_creation_time
            performance_metrics['configs_per_second'] = 100 / config_creation_time
            print(f"✅ Configuration creation: {config_creation_time:.3f}s ({100/config_creation_time:.1f} configs/s)")
            
        except Exception as e:
            performance_metrics['config_creation_time'] = None
            print(f"❌ Configuration performance test failed: {str(e)}")
        
        # Test memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            performance_metrics['memory_usage_mb'] = memory_info.rss / (1024 * 1024)
            print(f"✅ Memory usage: {performance_metrics['memory_usage_mb']:.1f} MB")
        except Exception as e:
            performance_metrics['memory_usage_mb'] = None
            print(f"❌ Memory usage test failed: {str(e)}")
        
        self.test_summary['performance_metrics'] = performance_metrics
        return performance_metrics
    
    def generate_recommendations(self):
        """Generate recommendations based on test results."""
        print("\nGenerating Recommendations...")
        print("-" * 40)
        
        recommendations = []
        
        # Analyze working functionality
        working_count = sum(1 for f in self.test_summary['working_functionality'] 
                          if f['status'] == 'WORKING')
        total_features = len(self.test_summary['working_functionality'])
        
        if working_count / total_features >= 0.8:
            recommendations.append({
                'category': 'Overall Assessment',
                'priority': 'HIGH',
                'recommendation': 'System is largely functional and ready for production use with minor fixes',
                'rationale': f'{working_count}/{total_features} core features are working'
            })
        elif working_count / total_features >= 0.6:
            recommendations.append({
                'category': 'Overall Assessment',
                'priority': 'MEDIUM',
                'recommendation': 'System needs moderate improvements before production use',
                'rationale': f'{working_count}/{total_features} core features are working'
            })
        else:
            recommendations.append({
                'category': 'Overall Assessment',
                'priority': 'HIGH',
                'recommendation': 'System needs significant improvements before production use',
                'rationale': f'Only {working_count}/{total_features} core features are working'
            })
        
        # Analyze issues
        high_severity_issues = sum(1 for i in self.test_summary['identified_issues'] 
                                 if i['severity'] == 'HIGH')
        
        if high_severity_issues > 0:
            recommendations.append({
                'category': 'Critical Issues',
                'priority': 'HIGH',
                'recommendation': f'Address {high_severity_issues} high-severity issues immediately',
                'rationale': 'High-severity issues prevent core functionality'
            })
        
        # Dependency recommendations
        missing_deps = [i for i in self.test_summary['identified_issues'] 
                       if i['issue'] == 'Missing Optional Dependencies']
        if missing_deps:
            recommendations.append({
                'category': 'Dependencies',
                'priority': 'MEDIUM',
                'recommendation': 'Install optional dependencies for full functionality',
                'rationale': 'Missing dependencies limit MD and trajectory analysis capabilities'
            })
        
        # Performance recommendations
        if self.test_summary['performance_metrics'].get('import_time', 0) > 5.0:
            recommendations.append({
                'category': 'Performance',
                'priority': 'LOW',
                'recommendation': 'Optimize import performance',
                'rationale': 'Import time is slower than optimal'
            })
        
        # Validation recommendations
        if self.test_summary['validation_results'].get('overall_score', 0) < 0.8:
            recommendations.append({
                'category': 'Validation',
                'priority': 'MEDIUM',
                'recommendation': 'Improve validation scores through better algorithm accuracy',
                'rationale': 'Validation score below 80% threshold'
            })
        
        self.test_summary['recommendations'] = recommendations
        
        for rec in recommendations:
            priority_symbol = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
            print(f"{priority_symbol} {rec['category']}: {rec['recommendation']}")
        
        return recommendations
    
    def generate_comprehensive_report(self):
        """Generate the final comprehensive report."""
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE END-TO-END TEST REPORT")
        print("=" * 80)
        
        # Add metadata
        self.test_summary['metadata'] = {
            'test_date': datetime.now().isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'qbes_version': '1.0.0',
            'test_type': 'End-to-End System Testing',
            'task_reference': '12.1 Perform end-to-end system testing'
        }
        
        # Save JSON report
        json_report_path = self.results_dir / "comprehensive_end_to_end_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.test_summary, f, indent=2)
        
        # Generate human-readable report
        text_report_path = self.results_dir / "comprehensive_end_to_end_report.txt"
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("QBES COMPREHENSIVE END-TO-END TEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"QBES Version: 1.0.0\n")
            f.write(f"Task Reference: 12.1 Perform end-to-end system testing\n")
            f.write(f"Python Version: {sys.version.split()[0]}\n")
            f.write(f"Platform: {sys.platform}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            working_count = sum(1 for f in self.test_summary['working_functionality'] 
                              if f['status'] == 'WORKING')
            total_features = len(self.test_summary['working_functionality'])
            high_issues = sum(1 for i in self.test_summary['identified_issues'] 
                            if i['severity'] == 'HIGH')
            
            f.write(f"Core Functionality: {working_count}/{total_features} features working\n")
            f.write(f"Critical Issues: {high_issues} high-severity issues identified\n")
            
            if 'overall_score' in self.test_summary['validation_results']:
                f.write(f"Validation Score: {self.test_summary['validation_results']['overall_score']:.1%}\n")
            
            f.write("\n")
            
            # Working Functionality
            f.write("WORKING FUNCTIONALITY\n")
            f.write("-" * 25 + "\n")
            
            for feature in self.test_summary['working_functionality']:
                status_symbol = "✅" if feature['status'] == 'WORKING' else "⚠️" if feature['status'] == 'PARTIAL' else "❌"
                f.write(f"{status_symbol} {feature['feature']}: {feature['status']}\n")
                f.write(f"   {feature['description']}\n\n")
            
            # Identified Issues
            f.write("IDENTIFIED ISSUES\n")
            f.write("-" * 20 + "\n")
            
            for issue in self.test_summary['identified_issues']:
                severity_symbol = "🔴" if issue['severity'] == 'HIGH' else "🟡" if issue['severity'] == 'MEDIUM' else "🟢"
                f.write(f"{severity_symbol} {issue['issue']} ({issue['severity']} SEVERITY)\n")
                f.write(f"   Description: {issue['description']}\n")
                f.write(f"   Impact: {issue['impact']}\n")
                f.write(f"   Recommendation: {issue['recommendation']}\n")
                if 'details' in issue:
                    f.write(f"   Details: {', '.join(issue['details'])}\n")
                f.write("\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            
            for metric, value in self.test_summary['performance_metrics'].items():
                if value is not None:
                    if 'time' in metric:
                        f.write(f"{metric}: {value:.3f}s\n")
                    elif 'mb' in metric.lower():
                        f.write(f"{metric}: {value:.1f} MB\n")
                    else:
                        f.write(f"{metric}: {value}\n")
                else:
                    f.write(f"{metric}: Not available\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            for rec in self.test_summary['recommendations']:
                priority_symbol = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
                f.write(f"{priority_symbol} {rec['category']} ({rec['priority']} PRIORITY)\n")
                f.write(f"   Recommendation: {rec['recommendation']}\n")
                f.write(f"   Rationale: {rec['rationale']}\n\n")
            
            # Conclusion
            f.write("CONCLUSION\n")
            f.write("-" * 10 + "\n")
            
            if working_count / total_features >= 0.8 and high_issues == 0:
                f.write("READY FOR PRODUCTION: System demonstrates strong functionality with minimal issues.\n")
            elif working_count / total_features >= 0.6:
                f.write("NEEDS MINOR IMPROVEMENTS: System is largely functional but requires attention to identified issues.\n")
            else:
                f.write("NEEDS MAJOR IMPROVEMENTS: System requires significant work before production readiness.\n")
        
        print(f"\nComprehensive reports generated:")
        print(f"  JSON Report: {json_report_path}")
        print(f"  Text Report: {text_report_path}")
        
        return self.test_summary
    
    def run_comprehensive_testing(self):
        """Run all comprehensive testing and generate report."""
        print("QBES COMPREHENSIVE END-TO-END TESTING")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results Directory: {self.results_dir}")
        
        # Run all tests
        self.test_core_functionality()
        self.identify_system_issues()
        self.assess_performance_characteristics()
        self.generate_recommendations()
        
        # Generate final report
        report = self.generate_comprehensive_report()
        
        return report


def main():
    """Main function to run comprehensive testing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive end-to-end test report for QBES"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='comprehensive_test_results',
        help='Directory to store test results'
    )
    
    args = parser.parse_args()
    
    try:
        # Run comprehensive testing
        reporter = ComprehensiveTestReporter(results_dir=args.results_dir)
        report = reporter.run_comprehensive_testing()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        working_count = sum(1 for f in report['working_functionality'] 
                          if f['status'] == 'WORKING')
        total_features = len(report['working_functionality'])
        high_issues = sum(1 for i in report['identified_issues'] 
                        if i['severity'] == 'HIGH')
        
        print(f"Working Features: {working_count}/{total_features}")
        print(f"Critical Issues: {high_issues}")
        
        if 'overall_score' in report['validation_results']:
            print(f"Validation Score: {report['validation_results']['overall_score']:.1%}")
        
        success_rate = working_count / total_features if total_features > 0 else 0
        
        if success_rate >= 0.8 and high_issues == 0:
            print("\n🎉 Comprehensive testing shows system is ready for production!")
            return 0
        elif success_rate >= 0.6:
            print("\n⚠️  Comprehensive testing shows system needs minor improvements.")
            return 0
        else:
            print("\n❌ Comprehensive testing shows system needs major improvements.")
            return 1
            
    except KeyboardInterrupt:
        print("\nComprehensive testing interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nComprehensive testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())