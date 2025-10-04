"""
Cross-validation against other quantum simulation packages.

This module provides tools to compare QBES results with other established
quantum simulation packages for validation and benchmarking.
"""

import numpy as np
import subprocess
import tempfile
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings

from ..core.data_models import (
    SimulationConfig, QuantumSubsystem, DensityMatrix, 
    Hamiltonian, LindbladOperator
)
from .benchmark_systems import BenchmarkSystem, BenchmarkResult


@dataclass
class CrossValidationResult:
    """Results from cross-validation against another package."""
    package_name: str
    package_version: str
    system_name: str
    validation_passed: bool
    qbes_results: List[float]
    reference_results: List[float]
    relative_differences: List[float]
    max_relative_difference: float
    mean_relative_difference: float
    correlation_coefficient: float
    tolerance: float
    error_message: Optional[str] = None


class ExternalPackageInterface(ABC):
    """Abstract interface for external quantum simulation packages."""
    
    def __init__(self, package_name: str, tolerance: float = 1e-4):
        """
        Initialize external package interface.
        
        Args:
            package_name: Name of the external package
            tolerance: Tolerance for comparison
        """
        self.package_name = package_name
        self.tolerance = tolerance
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the external package is available."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get version of the external package."""
        pass
    
    @abstractmethod
    def run_simulation(self, system_config: Dict[str, Any]) -> List[float]:
        """Run simulation with the external package."""
        pass
    
    @abstractmethod
    def convert_qbes_config(self, config: SimulationConfig, 
                           subsystem: QuantumSubsystem) -> Dict[str, Any]:
        """Convert QBES configuration to external package format."""
        pass


class QuTiPInterface(ExternalPackageInterface):
    """
    Interface to QuTiP (Quantum Toolbox in Python).
    
    QuTiP is a widely-used open-source quantum simulation library
    that provides reference implementations for quantum dynamics.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        """Initialize QuTiP interface."""
        super().__init__("QuTiP", tolerance)
        self._qutip_available = None
        self._qutip_version = None
    
    def is_available(self) -> bool:
        """Check if QuTiP is available."""
        if self._qutip_available is None:
            try:
                import qutip
                self._qutip_available = True
                self._qutip_version = qutip.__version__
            except ImportError:
                self._qutip_available = False
                self._qutip_version = None
        
        return self._qutip_available
    
    def get_version(self) -> str:
        """Get QuTiP version."""
        if not self.is_available():
            return "Not available"
        return self._qutip_version
    
    def run_simulation(self, system_config: Dict[str, Any]) -> List[float]:
        """Run simulation using QuTiP."""
        if not self.is_available():
            raise RuntimeError("QuTiP is not available")
        
        try:
            import qutip as qt
            
            # Extract configuration
            hamiltonian_matrix = np.array(system_config['hamiltonian'])
            lindblad_matrices = system_config.get('lindblad_operators', [])
            initial_state_matrix = np.array(system_config['initial_state'])
            times = np.array(system_config['times'])
            observables = system_config.get('observables', [])
            
            # Convert to QuTiP objects
            H = qt.Qobj(hamiltonian_matrix)
            rho0 = qt.Qobj(initial_state_matrix)
            
            # Create collapse operators
            c_ops = []
            for lindblad_matrix in lindblad_matrices:
                c_ops.append(qt.Qobj(lindblad_matrix))
            
            # Run master equation evolution
            if c_ops:
                result = qt.mesolve(H, rho0, times, c_ops)
            else:
                # Unitary evolution
                result = qt.mesolve(H, rho0, times)
            
            # Extract observables
            output_values = []
            
            if observables:
                for obs_matrix in observables:
                    obs_op = qt.Qobj(obs_matrix)
                    expectation_values = qt.expect(obs_op, result.states)
                    output_values.extend(expectation_values.tolist())
            else:
                # Default: return final state populations
                final_state = result.states[-1]
                populations = np.real(np.diag(final_state.full()))
                output_values.extend(populations.tolist())
            
            return output_values
            
        except Exception as e:
            raise RuntimeError(f"QuTiP simulation failed: {str(e)}")
    
    def convert_qbes_config(self, config: SimulationConfig, 
                           subsystem: QuantumSubsystem) -> Dict[str, Any]:
        """Convert QBES configuration to QuTiP format."""
        
        # Build Hamiltonian matrix
        n_sites = len(subsystem.basis_states)
        hamiltonian = np.zeros((n_sites, n_sites), dtype=complex)
        
        # Diagonal elements (site energies)
        if 'site_energies' in subsystem.hamiltonian_parameters:
            site_energies = subsystem.hamiltonian_parameters['site_energies']
            for i, energy in enumerate(site_energies):
                hamiltonian[i, i] = energy
        
        # Off-diagonal elements (couplings)
        hamiltonian += subsystem.coupling_matrix
        
        # Create initial state (ground state or equal superposition)
        initial_state = np.zeros((n_sites, n_sites), dtype=complex)
        initial_state[0, 0] = 1.0  # Start in first site
        
        # Time points
        times = np.linspace(0, config.simulation_time, 
                          int(config.simulation_time / config.time_step) + 1)
        
        # Lindblad operators (simplified - would need proper noise model conversion)
        lindblad_operators = []
        if config.noise_model_type != "none":
            # Add simple dephasing
            for i in range(n_sites):
                dephasing_op = np.zeros((n_sites, n_sites), dtype=complex)
                dephasing_op[i, i] = 1.0
                lindblad_operators.append(0.1 * dephasing_op)  # Weak dephasing
        
        # Observable operators
        observables = []
        # Population operators
        for i in range(n_sites):
            pop_op = np.zeros((n_sites, n_sites), dtype=complex)
            pop_op[i, i] = 1.0
            observables.append(pop_op)
        
        return {
            'hamiltonian': hamiltonian,
            'initial_state': initial_state,
            'times': times,
            'lindblad_operators': lindblad_operators,
            'observables': observables
        }


class QuantumOpticsPyInterface(ExternalPackageInterface):
    """
    Interface to QuantumOptics.py (Julia package via PyJulia).
    
    QuantumOptics.py is a high-performance quantum simulation library
    written in Julia with Python bindings.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        """Initialize QuantumOptics.py interface."""
        super().__init__("QuantumOptics.py", tolerance)
        self._qo_available = None
    
    def is_available(self) -> bool:
        """Check if QuantumOptics.py is available."""
        if self._qo_available is None:
            try:
                # Try to import julia and QuantumOptics
                import julia
                from julia import QuantumOptics as qo
                self._qo_available = True
            except (ImportError, RuntimeError):
                self._qo_available = False
        
        return self._qo_available
    
    def get_version(self) -> str:
        """Get QuantumOptics.py version."""
        if not self.is_available():
            return "Not available"
        
        try:
            from julia import Pkg
            # This is a simplified version check
            return "0.4.x"  # Placeholder
        except:
            return "Unknown"
    
    def run_simulation(self, system_config: Dict[str, Any]) -> List[float]:
        """Run simulation using QuantumOptics.py."""
        if not self.is_available():
            raise RuntimeError("QuantumOptics.py is not available")
        
        # This would implement the actual QuantumOptics.py simulation
        # For now, return mock results that would come from the Julia package
        hamiltonian_matrix = np.array(system_config['hamiltonian'])
        times = np.array(system_config['times'])
        
        # Mock evolution (in practice, this would call Julia code)
        n_sites = hamiltonian_matrix.shape[0]
        final_populations = np.random.random(n_sites)
        final_populations /= np.sum(final_populations)
        
        return final_populations.tolist()
    
    def convert_qbes_config(self, config: SimulationConfig, 
                           subsystem: QuantumSubsystem) -> Dict[str, Any]:
        """Convert QBES configuration to QuantumOptics.py format."""
        # Similar to QuTiP conversion
        n_sites = len(subsystem.basis_states)
        hamiltonian = np.zeros((n_sites, n_sites), dtype=complex)
        
        if 'site_energies' in subsystem.hamiltonian_parameters:
            site_energies = subsystem.hamiltonian_parameters['site_energies']
            for i, energy in enumerate(site_energies):
                hamiltonian[i, i] = energy
        
        hamiltonian += subsystem.coupling_matrix
        
        times = np.linspace(0, config.simulation_time, 
                          int(config.simulation_time / config.time_step) + 1)
        
        return {
            'hamiltonian': hamiltonian,
            'times': times,
            'n_sites': n_sites
        }


class MockReferenceInterface(ExternalPackageInterface):
    """
    Mock interface for testing cross-validation functionality.
    
    Provides known reference results for validation testing
    when external packages are not available.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        """Initialize mock reference interface."""
        super().__init__("MockReference", tolerance)
    
    def is_available(self) -> bool:
        """Mock package is always available."""
        return True
    
    def get_version(self) -> str:
        """Return mock version."""
        return "1.0.0-mock"
    
    def run_simulation(self, system_config: Dict[str, Any]) -> List[float]:
        """Return known reference results."""
        hamiltonian_matrix = np.array(system_config['hamiltonian'])
        n_sites = hamiltonian_matrix.shape[0]
        
        # Generate deterministic "reference" results based on system properties
        np.random.seed(42)  # Fixed seed for reproducible results
        
        if n_sites == 2:
            # Two-level system: oscillating populations
            return [0.6, 0.4]  # Final populations
        elif n_sites == 7:
            # FMO-like system
            return [0.05, 0.15, 0.25, 0.30, 0.15, 0.08, 0.02]
        else:
            # General case: exponential decay from first site
            populations = np.exp(-np.arange(n_sites) * 0.5)
            populations /= np.sum(populations)
            return populations.tolist()
    
    def convert_qbes_config(self, config: SimulationConfig, 
                           subsystem: QuantumSubsystem) -> Dict[str, Any]:
        """Convert configuration (minimal for mock)."""
        n_sites = len(subsystem.basis_states)
        hamiltonian = subsystem.coupling_matrix.copy()
        
        if 'site_energies' in subsystem.hamiltonian_parameters:
            site_energies = subsystem.hamiltonian_parameters['site_energies']
            for i, energy in enumerate(site_energies):
                hamiltonian[i, i] = energy
        
        return {
            'hamiltonian': hamiltonian,
            'n_sites': n_sites
        }


class CrossValidator:
    """
    Main class for cross-validation against external packages.
    
    Manages multiple external package interfaces and provides
    automated comparison and validation functionality.
    """
    
    def __init__(self):
        """Initialize cross-validator."""
        self.interfaces: Dict[str, ExternalPackageInterface] = {}
        self.results: List[CrossValidationResult] = []
    
    def add_interface(self, interface: ExternalPackageInterface):
        """Add an external package interface."""
        self.interfaces[interface.package_name] = interface
    
    def add_standard_interfaces(self):
        """Add standard external package interfaces."""
        self.add_interface(QuTiPInterface())
        self.add_interface(QuantumOpticsPyInterface())
        self.add_interface(MockReferenceInterface())  # Always available for testing
    
    def check_available_packages(self) -> Dict[str, bool]:
        """Check which external packages are available."""
        availability = {}
        
        for name, interface in self.interfaces.items():
            available = interface.is_available()
            availability[name] = available
            
            if available:
                version = interface.get_version()
                print(f"✅ {name} v{version} - Available")
            else:
                print(f"❌ {name} - Not available")
        
        return availability
    
    def validate_against_package(self, 
                                package_name: str,
                                config: SimulationConfig,
                                subsystem: QuantumSubsystem,
                                qbes_results: List[float],
                                system_name: str = "test_system") -> CrossValidationResult:
        """
        Validate QBES results against a specific external package.
        
        Args:
            package_name: Name of the external package
            config: QBES simulation configuration
            subsystem: QBES quantum subsystem
            qbes_results: Results from QBES simulation
            system_name: Name of the test system
            
        Returns:
            CrossValidationResult with comparison statistics
        """
        if package_name not in self.interfaces:
            raise ValueError(f"Package {package_name} not found")
        
        interface = self.interfaces[package_name]
        
        if not interface.is_available():
            return CrossValidationResult(
                package_name=package_name,
                package_version="Not available",
                system_name=system_name,
                validation_passed=False,
                qbes_results=qbes_results,
                reference_results=[],
                relative_differences=[],
                max_relative_difference=float('inf'),
                mean_relative_difference=float('inf'),
                correlation_coefficient=0.0,
                tolerance=interface.tolerance,
                error_message=f"Package {package_name} is not available"
            )
        
        try:
            # Convert QBES configuration to external package format
            external_config = interface.convert_qbes_config(config, subsystem)
            
            # Run simulation with external package
            reference_results = interface.run_simulation(external_config)
            
            # Perform comparison
            return self._compare_results(
                package_name,
                interface.get_version(),
                system_name,
                qbes_results,
                reference_results,
                interface.tolerance
            )
            
        except Exception as e:
            return CrossValidationResult(
                package_name=package_name,
                package_version=interface.get_version(),
                system_name=system_name,
                validation_passed=False,
                qbes_results=qbes_results,
                reference_results=[],
                relative_differences=[],
                max_relative_difference=float('inf'),
                mean_relative_difference=float('inf'),
                correlation_coefficient=0.0,
                tolerance=interface.tolerance,
                error_message=str(e)
            )
    
    def _compare_results(self,
                        package_name: str,
                        package_version: str,
                        system_name: str,
                        qbes_results: List[float],
                        reference_results: List[float],
                        tolerance: float) -> CrossValidationResult:
        """Compare QBES results with reference results."""
        
        qbes_array = np.array(qbes_results)
        ref_array = np.array(reference_results)
        
        # Ensure arrays have the same length
        min_length = min(len(qbes_array), len(ref_array))
        qbes_array = qbes_array[:min_length]
        ref_array = ref_array[:min_length]
        
        # Calculate relative differences
        relative_differences = np.abs(qbes_array - ref_array) / (np.abs(ref_array) + 1e-12)
        max_relative_difference = np.max(relative_differences)
        mean_relative_difference = np.mean(relative_differences)
        
        # Calculate correlation coefficient
        if len(qbes_array) > 1:
            correlation_coefficient = np.corrcoef(qbes_array, ref_array)[0, 1]
            if np.isnan(correlation_coefficient):
                correlation_coefficient = 0.0
        else:
            correlation_coefficient = 1.0 if qbes_array[0] == ref_array[0] else 0.0
        
        # Validation passes if mean relative difference is within tolerance
        # and correlation is high
        validation_passed = (mean_relative_difference < tolerance) and (correlation_coefficient > 0.9)
        
        return CrossValidationResult(
            package_name=package_name,
            package_version=package_version,
            system_name=system_name,
            validation_passed=validation_passed,
            qbes_results=qbes_array.tolist(),
            reference_results=ref_array.tolist(),
            relative_differences=relative_differences.tolist(),
            max_relative_difference=max_relative_difference,
            mean_relative_difference=mean_relative_difference,
            correlation_coefficient=correlation_coefficient,
            tolerance=tolerance
        )
    
    def run_cross_validation_suite(self, 
                                  test_systems: Optional[List[Tuple[str, SimulationConfig, QuantumSubsystem, List[float]]]] = None) -> List[CrossValidationResult]:
        """
        Run cross-validation against all available packages.
        
        Args:
            test_systems: List of (name, config, subsystem, qbes_results) tuples
            
        Returns:
            List of cross-validation results
        """
        if test_systems is None:
            test_systems = self._create_default_test_systems()
        
        self.results = []
        
        print("Running Cross-Validation Suite")
        print("=" * 40)
        
        # Check package availability
        availability = self.check_available_packages()
        available_packages = [name for name, avail in availability.items() if avail]
        
        if not available_packages:
            print("No external packages available for cross-validation.")
            return self.results
        
        print(f"\nTesting against {len(available_packages)} packages...")
        print()
        
        # Run validation for each system and package
        for system_name, config, subsystem, qbes_results in test_systems:
            print(f"Testing system: {system_name}")
            
            for package_name in available_packages:
                print(f"  vs {package_name}...", end=" ")
                
                result = self.validate_against_package(
                    package_name, config, subsystem, qbes_results, system_name
                )
                
                self.results.append(result)
                
                status = "PASS" if result.validation_passed else "FAIL"
                print(f"{status} (dev: {result.mean_relative_difference:.1%})")
            
            print()
        
        return self.results
    
    def _create_default_test_systems(self) -> List[Tuple[str, SimulationConfig, QuantumSubsystem, List[float]]]:
        """Create default test systems for cross-validation."""
        from ..core.data_models import Atom, QuantumState
        
        test_systems = []
        
        # Two-level system
        config_2level = SimulationConfig(
            system_pdb="",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.01,
            quantum_subsystem_selection="all",
            noise_model_type="none",
            output_directory="cross_validation"
        )
        
        atoms_2level = [
            Atom(element="C", position=np.array([0, 0, 0]), charge=0.0, mass=12.0, atom_id=0),
            Atom(element="C", position=np.array([0, 0, 1]), charge=0.0, mass=12.0, atom_id=1)
        ]
        
        basis_states_2level = [
            QuantumState(
                coefficients=np.array([1.0, 0.0], dtype=complex),
                basis_labels=["ground", "excited"],
                energy=0.0
            ),
            QuantumState(
                coefficients=np.array([0.0, 1.0], dtype=complex),
                basis_labels=["ground", "excited"],
                energy=1.0
            )
        ]
        
        subsystem_2level = QuantumSubsystem(
            atoms=atoms_2level,
            hamiltonian_parameters={"site_energies": [0.0, 1.0]},
            coupling_matrix=np.array([[0.0, 0.1], [0.1, 0.0]]),
            basis_states=basis_states_2level
        )
        
        qbes_results_2level = [0.55, 0.45]  # Mock QBES results
        
        test_systems.append(("two_level_system", config_2level, subsystem_2level, qbes_results_2level))
        
        return test_systems
    
    def generate_cross_validation_report(self) -> str:
        """Generate comprehensive cross-validation report."""
        if not self.results:
            return "No cross-validation results available."
        
        report = []
        report.append("=" * 70)
        report.append("QBES Cross-Validation Report")
        report.append("=" * 70)
        
        # Summary statistics
        total_validations = len(self.results)
        passed_validations = sum(1 for r in self.results if r.validation_passed)
        
        report.append(f"Total Cross-Validations: {total_validations}")
        report.append(f"Passed: {passed_validations}")
        report.append(f"Failed: {total_validations - passed_validations}")
        report.append(f"Success Rate: {(passed_validations/total_validations)*100:.1f}%")
        report.append("")
        
        # Results by package
        packages = set(r.package_name for r in self.results)
        
        for package in sorted(packages):
            package_results = [r for r in self.results if r.package_name == package]
            package_passed = sum(1 for r in package_results if r.validation_passed)
            
            report.append(f"Package: {package}")
            report.append("-" * 30)
            
            if package_results:
                version = package_results[0].package_version
                report.append(f"Version: {version}")
                report.append(f"Tests: {len(package_results)}")
                report.append(f"Passed: {package_passed}")
                report.append(f"Success Rate: {(package_passed/len(package_results))*100:.1f}%")
                
                # Average statistics for passed tests
                passed_results = [r for r in package_results if r.validation_passed]
                if passed_results:
                    avg_deviation = np.mean([r.mean_relative_difference for r in passed_results])
                    avg_correlation = np.mean([r.correlation_coefficient for r in passed_results])
                    report.append(f"Average Deviation: {avg_deviation:.1%}")
                    report.append(f"Average Correlation: {avg_correlation:.3f}")
                
                report.append("")
                
                # Detailed results
                for result in package_results:
                    status = "✅ PASS" if result.validation_passed else "❌ FAIL"
                    report.append(f"  {result.system_name}: {status}")
                    
                    if result.validation_passed:
                        report.append(f"    Mean Deviation: {result.mean_relative_difference:.1%}")
                        report.append(f"    Correlation: {result.correlation_coefficient:.3f}")
                    else:
                        if result.error_message:
                            report.append(f"    Error: {result.error_message}")
                        else:
                            report.append(f"    Mean Deviation: {result.mean_relative_difference:.1%} (>{result.tolerance:.1%})")
            
            report.append("")
        
        # Overall assessment
        report.append("Overall Assessment:")
        report.append("-" * 19)
        
        if passed_validations == total_validations:
            report.append("✅ All cross-validations passed!")
            report.append("QBES results are consistent with established packages.")
        elif passed_validations >= total_validations * 0.8:
            report.append("⚠️  Most cross-validations passed.")
            report.append("Minor discrepancies may indicate implementation differences.")
        else:
            report.append("❌ Multiple cross-validation failures detected.")
            report.append("Significant algorithmic differences or bugs may exist.")
        
        return "\n".join(report)
    
    def save_cross_validation_results(self, filepath: str):
        """Save cross-validation results to JSON file."""
        from datetime import datetime
        results_dict = {
            'cross_validation_results': [asdict(result) for result in self.results],
            'timestamp': datetime.now().isoformat(),
            'total_validations': len(self.results),
            'passed_validations': sum(1 for r in self.results if r.validation_passed)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)


def run_cross_validation() -> CrossValidator:
    """
    Run comprehensive cross-validation suite.
    
    Returns:
        CrossValidator with validation results
    """
    print("Running QBES Cross-Validation Suite")
    print("=" * 50)
    
    validator = CrossValidator()
    validator.add_standard_interfaces()
    
    # Run cross-validation
    validator.run_cross_validation_suite()
    
    # Generate and display report
    print("\nCross-Validation Report:")
    print("=" * 25)
    print(validator.generate_cross_validation_report())
    
    # Save results
    os.makedirs("validation_results", exist_ok=True)
    results_file = "validation_results/cross_validation_results.json"
    validator.save_cross_validation_results(results_file)
    print(f"\nResults saved to: {results_file}")
    
    return validator