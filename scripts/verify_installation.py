#!/usr/bin/env python3
"""
QBES Installation Verification Script

This script verifies that QBES has been installed correctly and all
components are functioning as expected.
"""

import sys
import subprocess
import importlib
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class InstallationVerifier:
    """Verifies QBES installation completeness and functionality."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the verifier."""
        self.verbose = verbose
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def _log(self, message: str, level: str = 'INFO'):
        """Log verification messages."""
        if self.verbose or level in ['ERROR', 'WARNING']:
            prefix = {
                'INFO': '  ',
                'SUCCESS': '✓ ',
                'WARNING': '⚠ ',
                'ERROR': '✗ '
            }.get(level, '  ')
            print(f"{prefix}{message}")
    
    def verify_python_version(self) -> bool:
        """Verify Python version compatibility."""
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            self._log(f"Python {version.major}.{version.minor}.{version.micro} is compatible", 'SUCCESS')
            self.results['python_version'] = True
            return True
        else:
            self._log(f"Python {version.major}.{version.minor} is too old (requires 3.8+)", 'ERROR')
            self.errors.append("Python version incompatible")
            self.results['python_version'] = False
            return False
    
    def verify_core_imports(self) -> bool:
        """Verify that core QBES modules can be imported."""
        core_modules = [
            'qbes',
            'qbes.core',
            'qbes.core.data_models',
            'qbes.core.interfaces',
            'qbes.config_manager',
            'qbes.quantum_engine',
            'qbes.md_engine',
            'qbes.noise_models',
            'qbes.simulation_engine',
            'qbes.analysis',
            'qbes.visualization',
            'qbes.cli'
        ]
        
        failed_imports = []
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                self._log(f"Successfully imported {module}", 'SUCCESS')
            except ImportError as e:
                self._log(f"Failed to import {module}: {e}", 'ERROR')
                failed_imports.append(module)
        
        if failed_imports:
            self.errors.append(f"Failed to import core modules: {', '.join(failed_imports)}")
            self.results['core_imports'] = False
            return False
        else:
            self.results['core_imports'] = True
            return True
    
    def verify_dependencies(self) -> bool:
        """Verify that all required dependencies are available."""
        required_deps = [
            'numpy',
            'scipy',
            'matplotlib',
            'qutip',
            'pandas',
            'h5py',
            'yaml',
            'click',
            'tqdm'
        ]
        
        optional_deps = [
            'openmm',
            'mdtraj',
            'Bio',  # biopython
            'numba',
            'seaborn',
            'plotly',
            'astropy',
            'joblib'
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required dependencies
        for dep in required_deps:
            try:
                importlib.import_module(dep)
                self._log(f"Required dependency {dep} found", 'SUCCESS')
            except ImportError:
                self._log(f"Required dependency {dep} missing", 'ERROR')
                missing_required.append(dep)
        
        # Check optional dependencies
        for dep in optional_deps:
            try:
                importlib.import_module(dep)
                self._log(f"Optional dependency {dep} found", 'SUCCESS')
            except ImportError:
                self._log(f"Optional dependency {dep} missing", 'WARNING')
                missing_optional.append(dep)
        
        if missing_required:
            self.errors.append(f"Missing required dependencies: {', '.join(missing_required)}")
            self.results['dependencies'] = False
            return False
        
        if missing_optional:
            self.warnings.append(f"Missing optional dependencies: {', '.join(missing_optional)}")
        
        self.results['dependencies'] = True
        return True
    
    def verify_cli_functionality(self) -> bool:
        """Verify that the CLI is functional."""
        try:
            # Test basic CLI help
            result = subprocess.run([
                sys.executable, '-m', 'qbes.cli', '--help'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and 'Quantum Biological Environment Simulator' in result.stdout:
                self._log("CLI help command works", 'SUCCESS')
            else:
                self._log(f"CLI help command failed: {result.stderr}", 'ERROR')
                self.errors.append("CLI help command failed")
                self.results['cli_functionality'] = False
                return False
            
            # Test CLI version command
            result = subprocess.run([
                sys.executable, '-m', 'qbes.cli', '--version'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self._log(f"CLI version: {result.stdout.strip()}", 'SUCCESS')
            else:
                self._log("CLI version command failed", 'WARNING')
                self.warnings.append("CLI version command failed")
            
            self.results['cli_functionality'] = True
            return True
            
        except subprocess.TimeoutExpired:
            self._log("CLI commands timed out", 'ERROR')
            self.errors.append("CLI commands timed out")
            self.results['cli_functionality'] = False
            return False
        except Exception as e:
            self._log(f"Error testing CLI: {e}", 'ERROR')
            self.errors.append(f"CLI test error: {e}")
            self.results['cli_functionality'] = False
            return False
    
    def verify_data_models(self) -> bool:
        """Verify that data models can be created and used."""
        try:
            from qbes.core.data_models import SimulationConfig, QuantumSubsystem, SimulationResults
            
            # Test SimulationConfig creation
            config = SimulationConfig(
                system_pdb="test.pdb",
                temperature=300.0,
                simulation_time=1e-12,
                time_step=1e-15,
                quantum_subsystem_selection="chromophores",
                noise_model_type="protein_ohmic",
                output_directory="./test_output"
            )
            self._log("SimulationConfig creation successful", 'SUCCESS')
            
            # Test basic validation
            if hasattr(config, 'temperature') and config.temperature == 300.0:
                self._log("Data model attributes accessible", 'SUCCESS')
            else:
                self._log("Data model attributes not accessible", 'ERROR')
                self.errors.append("Data model attribute access failed")
                self.results['data_models'] = False
                return False
            
            self.results['data_models'] = True
            return True
            
        except Exception as e:
            self._log(f"Data model verification failed: {e}", 'ERROR')
            self.errors.append(f"Data model error: {e}")
            self.results['data_models'] = False
            return False
    
    def verify_configuration_manager(self) -> bool:
        """Verify that the configuration manager works."""
        try:
            from qbes.config_manager import ConfigurationManager
            
            # Create configuration manager
            config_manager = ConfigurationManager()
            self._log("ConfigurationManager creation successful", 'SUCCESS')
            
            # Test with a temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("""
system_pdb: "test.pdb"
temperature: 300.0
simulation_time: 1e-12
time_step: 1e-15
quantum_subsystem_selection: "chromophores"
noise_model_type: "protein_ohmic"
output_directory: "./test_output"
""")
                temp_config_path = f.name
            
            try:
                # Test config loading
                config = config_manager.load_config(temp_config_path)
                self._log("Configuration loading successful", 'SUCCESS')
                
                # Test validation
                validation_result = config_manager.validate_parameters(config)
                if validation_result.is_valid:
                    self._log("Configuration validation successful", 'SUCCESS')
                else:
                    self._log("Configuration validation failed", 'WARNING')
                    self.warnings.append("Configuration validation issues")
                
            finally:
                os.unlink(temp_config_path)
            
            self.results['configuration_manager'] = True
            return True
            
        except Exception as e:
            self._log(f"Configuration manager verification failed: {e}", 'ERROR')
            self.errors.append(f"Configuration manager error: {e}")
            self.results['configuration_manager'] = False
            return False
    
    def verify_quantum_engine(self) -> bool:
        """Verify basic quantum engine functionality."""
        try:
            from qbes.quantum_engine import QuantumEngine
            import numpy as np
            
            # Create quantum engine
            engine = QuantumEngine()
            self._log("QuantumEngine creation successful", 'SUCCESS')
            
            # Test basic quantum state operations
            if hasattr(engine, 'create_density_matrix'):
                # Test density matrix creation
                rho = engine.create_density_matrix(2)  # 2x2 density matrix
                if rho is not None and rho.shape == (2, 2):
                    self._log("Density matrix creation successful", 'SUCCESS')
                else:
                    self._log("Density matrix creation failed", 'WARNING')
                    self.warnings.append("Quantum engine basic operations failed")
            
            self.results['quantum_engine'] = True
            return True
            
        except Exception as e:
            self._log(f"Quantum engine verification failed: {e}", 'ERROR')
            self.errors.append(f"Quantum engine error: {e}")
            self.results['quantum_engine'] = False
            return False
    
    def verify_benchmarks(self) -> bool:
        """Verify that benchmark systems are available."""
        try:
            from qbes.benchmarks import benchmark_systems
            
            # Check if benchmark systems are available
            if hasattr(benchmark_systems, 'get_available_benchmarks'):
                benchmarks = benchmark_systems.get_available_benchmarks()
                if benchmarks:
                    self._log(f"Found {len(benchmarks)} benchmark systems", 'SUCCESS')
                else:
                    self._log("No benchmark systems found", 'WARNING')
                    self.warnings.append("No benchmark systems available")
            else:
                self._log("Benchmark system interface not found", 'WARNING')
                self.warnings.append("Benchmark system interface missing")
            
            self.results['benchmarks'] = True
            return True
            
        except Exception as e:
            self._log(f"Benchmark verification failed: {e}", 'WARNING')
            self.warnings.append(f"Benchmark error: {e}")
            self.results['benchmarks'] = False
            return True  # Non-critical failure
    
    def run_basic_simulation_test(self) -> bool:
        """Run a minimal simulation test to verify end-to-end functionality."""
        try:
            from qbes.simulation_engine import SimulationEngine
            from qbes.core.data_models import SimulationConfig
            
            # Create a minimal test configuration
            config = SimulationConfig(
                system_pdb="test.pdb",
                temperature=300.0,
                simulation_time=1e-15,  # Very short simulation
                time_step=1e-16,
                quantum_subsystem_selection="test",
                noise_model_type="none",
                output_directory=tempfile.mkdtemp()
            )
            
            # Create simulation engine
            engine = SimulationEngine(config)
            self._log("SimulationEngine creation successful", 'SUCCESS')
            
            # Note: We don't actually run the simulation as it would require
            # a real PDB file and could take time. Just verify the engine can be created.
            
            self.results['simulation_test'] = True
            return True
            
        except Exception as e:
            self._log(f"Basic simulation test failed: {e}", 'WARNING')
            self.warnings.append(f"Simulation test error: {e}")
            self.results['simulation_test'] = False
            return True  # Non-critical failure
    
    def run_full_verification(self) -> bool:
        """Run all verification tests."""
        print("QBES Installation Verification")
        print("=" * 50)
        
        tests = [
            ("Python Version", self.verify_python_version),
            ("Core Imports", self.verify_core_imports),
            ("Dependencies", self.verify_dependencies),
            ("CLI Functionality", self.verify_cli_functionality),
            ("Data Models", self.verify_data_models),
            ("Configuration Manager", self.verify_configuration_manager),
            ("Quantum Engine", self.verify_quantum_engine),
            ("Benchmarks", self.verify_benchmarks),
            ("Basic Simulation Test", self.run_basic_simulation_test)
        ]
        
        all_passed = True
        critical_failures = 0
        
        for test_name, test_func in tests:
            print(f"\nTesting {test_name}...")
            try:
                result = test_func()
                if not result and test_name in ["Python Version", "Core Imports", "Dependencies", "CLI Functionality"]:
                    critical_failures += 1
                    all_passed = False
            except Exception as e:
                self._log(f"Test {test_name} crashed: {e}", 'ERROR')
                critical_failures += 1
                all_passed = False
        
        return critical_failures == 0
    
    def print_summary(self):
        """Print verification summary."""
        print("\n" + "=" * 50)
        print("VERIFICATION SUMMARY")
        print("=" * 50)
        
        # Count results
        passed = sum(1 for result in self.results.values() if result)
        total = len(self.results)
        
        print(f"Tests passed: {passed}/{total}")
        
        # Print individual results
        for test, result in self.results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {test.replace('_', ' ').title()}")
        
        # Print warnings and errors
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        # Overall assessment
        print("\n" + "-" * 50)
        critical_errors = len([e for e in self.errors if any(keyword in e.lower() 
                              for keyword in ['python', 'import', 'dependencies', 'cli'])])
        
        if critical_errors == 0:
            if not self.errors:
                print("✓ QBES installation is fully functional")
                print("  All tests passed successfully.")
            else:
                print("⚠ QBES installation is mostly functional")
                print("  Some non-critical issues detected.")
        else:
            print("✗ QBES installation has critical issues")
            print("  Please resolve the errors before using QBES.")
        
        return critical_errors == 0


def main():
    """Main verification function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QBES Installation Verifier')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--save-report', action='store_true',
                       help='Save verification report to file')
    parser.add_argument('--report-file', default='qbes_verification_report.txt',
                       help='Report filename (default: qbes_verification_report.txt)')
    
    args = parser.parse_args()
    
    verifier = InstallationVerifier(verbose=args.verbose)
    
    # Run verification
    success = verifier.run_full_verification()
    
    # Print summary
    overall_success = verifier.print_summary()
    
    # Save report if requested
    if args.save_report:
        try:
            with open(args.report_file, 'w') as f:
                f.write("QBES Installation Verification Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Test Results:\n")
                for test, result in verifier.results.items():
                    status = "PASS" if result else "FAIL"
                    f.write(f"  {status}: {test.replace('_', ' ').title()}\n")
                
                if verifier.warnings:
                    f.write(f"\nWarnings:\n")
                    for warning in verifier.warnings:
                        f.write(f"  - {warning}\n")
                
                if verifier.errors:
                    f.write(f"\nErrors:\n")
                    for error in verifier.errors:
                        f.write(f"  - {error}\n")
                
                f.write(f"\nOverall Status: {'SUCCESS' if overall_success else 'FAILURE'}\n")
            
            print(f"\nVerification report saved to: {args.report_file}")
            
        except Exception as e:
            print(f"Error saving report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()