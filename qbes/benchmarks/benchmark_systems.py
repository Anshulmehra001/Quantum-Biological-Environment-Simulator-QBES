"""
Benchmark test systems with known analytical solutions.

This module implements various quantum systems with analytical solutions
that can be used to validate the QBES simulation engine.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import time

from ..core.data_models import (
    SimulationConfig, QuantumSubsystem, DensityMatrix, 
    Hamiltonian, LindbladOperator, Atom, QuantumState,
    ValidationResult, CoherenceMetrics
)
from ..quantum_engine import QuantumEngine


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    system_name: str
    test_passed: bool
    numerical_result: float
    analytical_result: float
    relative_error: float
    absolute_error: float
    computation_time: float
    tolerance: float
    error_message: Optional[str] = None


class BenchmarkSystem(ABC):
    """Abstract base class for benchmark test systems."""
    
    def __init__(self, name: str, tolerance: float = 1e-6):
        """
        Initialize benchmark system.
        
        Args:
            name: Name of the benchmark system
            tolerance: Numerical tolerance for comparison
        """
        self.name = name
        self.tolerance = tolerance
        self.quantum_engine = QuantumEngine()
    
    @abstractmethod
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        """Get analytical solution at given time."""
        pass
    
    @abstractmethod
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up the quantum system for simulation."""
        pass
    
    @abstractmethod
    def extract_observable(self, state: DensityMatrix) -> float:
        """Extract the observable to compare with analytical solution."""
        pass
    
    def run_benchmark(self, final_time: float, time_step: float, **kwargs) -> BenchmarkResult:
        """
        Run benchmark test and compare with analytical solution.
        
        Args:
            final_time: Final simulation time
            time_step: Time step for evolution
            **kwargs: Additional parameters for analytical solution
            
        Returns:
            BenchmarkResult with comparison results
        """
        start_time = time.time()
        
        try:
            # Set up system
            hamiltonian, lindblad_ops, initial_state = self.setup_system()
            
            # Evolve system
            current_state = initial_state
            current_time = 0.0
            
            while current_time < final_time:
                current_state = self.quantum_engine.evolve_state(
                    current_state, time_step, hamiltonian, lindblad_ops
                )
                current_time += time_step
            
            # Extract observable
            numerical_result = self.extract_observable(current_state)
            
            # Get analytical solution
            analytical_result = self.get_analytical_solution(final_time, **kwargs)
            
            # Calculate errors
            absolute_error = abs(numerical_result - analytical_result)
            relative_error = absolute_error / abs(analytical_result) if analytical_result != 0 else absolute_error
            
            # Check if test passed
            test_passed = relative_error < self.tolerance
            
            computation_time = time.time() - start_time
            
            return BenchmarkResult(
                system_name=self.name,
                test_passed=test_passed,
                numerical_result=numerical_result,
                analytical_result=analytical_result,
                relative_error=relative_error,
                absolute_error=absolute_error,
                computation_time=computation_time,
                tolerance=self.tolerance
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            return BenchmarkResult(
                system_name=self.name,
                test_passed=False,
                numerical_result=0.0,
                analytical_result=0.0,
                relative_error=float('inf'),
                absolute_error=float('inf'),
                computation_time=computation_time,
                tolerance=self.tolerance,
                error_message=str(e)
            )


class TwoLevelSystemBenchmark(BenchmarkSystem):
    """
    Two-level system with known analytical solution.
    
    This benchmark tests basic quantum evolution without decoherence.
    The analytical solution for Rabi oscillations is well-known.
    """
    
    def __init__(self, energy_gap: float = 1.0, tolerance: float = 1e-6):
        """
        Initialize two-level system benchmark.
        
        Args:
            energy_gap: Energy difference between levels (in units of ħ)
            tolerance: Numerical tolerance for comparison
        """
        super().__init__("Two-Level System", tolerance)
        self.energy_gap = energy_gap
    
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        """
        Analytical solution for excited state population in Rabi oscillations.
        
        For a two-level system starting in ground state with Hamiltonian:
        H = (ω/2) * σz + (Ω/2) * σx
        
        The excited state population is: P_e(t) = sin²(Ωt/2)
        For simplicity, we use Ω = energy_gap.
        """
        return np.sin(self.energy_gap * time / 2) ** 2
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up two-level quantum system."""
        # Hamiltonian matrix: H = (ω/2) * σz + (Ω/2) * σx
        # Using Ω = energy_gap for driving field
        hamiltonian_matrix = 0.5 * np.array([
            [self.energy_gap, self.energy_gap],
            [self.energy_gap, -self.energy_gap]
        ], dtype=complex)
        
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix,
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
        
        # No Lindblad operators for isolated system
        lindblad_ops = []
        
        # Initial state: ground state |0⟩⟨0|
        initial_matrix = np.array([
            [1.0, 0.0],
            [0.0, 0.0]
        ], dtype=complex)
        
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def extract_observable(self, state: DensityMatrix) -> float:
        """Extract excited state population."""
        return float(np.real(state.matrix[1, 1]))


class HarmonicOscillatorBenchmark(BenchmarkSystem):
    """
    Harmonic oscillator with coherent state evolution.
    
    Tests quantum evolution of coherent states, which have known
    analytical solutions for position and momentum expectation values.
    """
    
    def __init__(self, frequency: float = 1.0, n_levels: int = 10, tolerance: float = 1e-5):
        """
        Initialize harmonic oscillator benchmark.
        
        Args:
            frequency: Oscillator frequency
            n_levels: Number of energy levels to include
            tolerance: Numerical tolerance
        """
        super().__init__("Harmonic Oscillator", tolerance)
        self.frequency = frequency
        self.n_levels = n_levels
    
    def get_analytical_solution(self, time: float, alpha: complex = 1.0, **kwargs) -> float:
        """
        Analytical solution for position expectation value of coherent state.
        
        For coherent state |α⟩, position expectation value is:
        ⟨x⟩(t) = √2 * Re(α * exp(-iωt))
        """
        alpha_t = alpha * np.exp(-1j * self.frequency * time)
        return np.sqrt(2) * np.real(alpha_t)
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up harmonic oscillator system."""
        # Create Hamiltonian matrix
        hamiltonian_matrix = np.zeros((self.n_levels, self.n_levels), dtype=complex)
        for n in range(self.n_levels):
            hamiltonian_matrix[n, n] = self.frequency * (n + 0.5)
        
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix,
            basis_labels=[f"n={n}" for n in range(self.n_levels)],
            time_dependent=False
        )
        
        # No Lindblad operators for isolated system
        lindblad_ops = []
        
        # Initial coherent state |α⟩ with α = 1
        alpha = 1.0
        initial_matrix = self._create_coherent_state_density_matrix(alpha)
        
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=[f"n={n}" for n in range(self.n_levels)],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def _create_coherent_state_density_matrix(self, alpha: complex) -> np.ndarray:
        """Create density matrix for coherent state |α⟩."""
        # Coherent state coefficients: c_n = exp(-|α|²/2) * α^n / √(n!)
        coeffs = np.zeros(self.n_levels, dtype=complex)
        norm_factor = np.exp(-abs(alpha)**2 / 2)
        
        for n in range(self.n_levels):
            coeffs[n] = norm_factor * (alpha**n) / np.sqrt(np.math.factorial(n))
        
        # Create density matrix |ψ⟩⟨ψ|
        return np.outer(coeffs, np.conj(coeffs))
    
    def extract_observable(self, state: DensityMatrix) -> float:
        """Extract position expectation value."""
        # Position operator in number basis: x = √(ħ/2mω) * (a† + a)
        # For simplicity, use units where √(ħ/2mω) = 1/√2
        position_matrix = np.zeros((self.n_levels, self.n_levels), dtype=complex)
        
        for n in range(self.n_levels - 1):
            # a|n⟩ = √n|n-1⟩, a†|n⟩ = √(n+1)|n+1⟩
            position_matrix[n, n+1] = np.sqrt(n + 1) / np.sqrt(2)  # a†
            position_matrix[n+1, n] = np.sqrt(n + 1) / np.sqrt(2)  # a
        
        # ⟨x⟩ = Tr(ρ * x)
        expectation = np.trace(state.matrix @ position_matrix)
        return float(np.real(expectation))


class DampedTwoLevelSystemBenchmark(BenchmarkSystem):
    """
    Two-level system with spontaneous emission.
    
    Tests Lindblad evolution with known analytical solution
    for population decay.
    """
    
    def __init__(self, energy_gap: float = 1.0, decay_rate: float = 0.1, tolerance: float = 1e-5):
        """
        Initialize damped two-level system.
        
        Args:
            energy_gap: Energy difference between levels
            decay_rate: Spontaneous emission rate
            tolerance: Numerical tolerance
        """
        super().__init__("Damped Two-Level System", tolerance)
        self.energy_gap = energy_gap
        self.decay_rate = decay_rate
    
    def get_analytical_solution(self, time: float, initial_population: float = 1.0, **kwargs) -> float:
        """
        Analytical solution for excited state population decay.
        
        For spontaneous emission: P_e(t) = P_e(0) * exp(-γt)
        """
        return initial_population * np.exp(-self.decay_rate * time)
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up damped two-level system."""
        # Hamiltonian: H = (ω/2) * σz
        hamiltonian_matrix = 0.5 * self.energy_gap * np.array([
            [1.0, 0.0],
            [0.0, -1.0]
        ], dtype=complex)
        
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix,
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
        
        # Lindblad operator for spontaneous emission: L = √γ * σ-
        lindblad_matrix = np.sqrt(self.decay_rate) * np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=complex)
        
        lindblad_op = LindbladOperator(
            matrix=lindblad_matrix,
            coupling_strength=np.sqrt(self.decay_rate),
            description="spontaneous_emission"
        )
        
        lindblad_ops = [lindblad_op]
        
        # Initial state: excited state |1⟩⟨1|
        initial_matrix = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ], dtype=complex)
        
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def extract_observable(self, state: DensityMatrix) -> float:
        """Extract excited state population."""
        return float(np.real(state.matrix[1, 1]))


class PhotosyntheticComplexBenchmark(BenchmarkSystem):
    """
    Simple photosynthetic complex model (dimer).
    
    Tests biological system with known decoherence behavior.
    Based on simplified models from literature.
    """
    
    def __init__(self, site_energy_1: float = 12000, site_energy_2: float = 12200,
                 coupling: float = 100, reorganization_energy: float = 35,
                 temperature: float = 300, tolerance: float = 1e-4):
        """
        Initialize photosynthetic complex benchmark.
        
        Args:
            site_energy_1: Energy of first chromophore (cm⁻¹)
            site_energy_2: Energy of second chromophore (cm⁻¹)
            coupling: Inter-site coupling (cm⁻¹)
            reorganization_energy: Reorganization energy (cm⁻¹)
            temperature: Temperature (K)
            tolerance: Numerical tolerance
        """
        super().__init__("Photosynthetic Complex", tolerance)
        self.site_energy_1 = site_energy_1
        self.site_energy_2 = site_energy_2
        self.coupling = coupling
        self.reorganization_energy = reorganization_energy
        self.temperature = temperature
    
    def get_analytical_solution(self, time: float, initial_coherence: float = 0.5, **kwargs) -> float:
        """
        Analytical solution for coherence decay in dimer.
        
        Approximate solution: |ρ₁₂(t)| ≈ |ρ₁₂(0)| * exp(-γ_deph * t)
        where γ_deph depends on reorganization energy and temperature.
        """
        # Simplified dephasing rate (Redfield theory approximation)
        kT = 0.695 * self.temperature  # kT in cm⁻¹ (using k_B = 0.695 cm⁻¹/K)
        gamma_deph = 2 * self.reorganization_energy**2 / (kT**2) * self.coupling**2 / (kT**3)
        
        return initial_coherence * np.exp(-gamma_deph * time)
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up photosynthetic dimer system."""
        # Hamiltonian matrix for dimer
        hamiltonian_matrix = np.array([
            [self.site_energy_1, self.coupling],
            [self.coupling, self.site_energy_2]
        ], dtype=complex)
        
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix,
            basis_labels=["site_1", "site_2"],
            time_dependent=False
        )
        
        # Lindblad operators for dephasing
        kT = 0.695 * self.temperature
        dephasing_rate = self.reorganization_energy**2 / kT**2
        
        # Dephasing operators: σz for each site
        lindblad_1 = LindbladOperator(
            matrix=np.sqrt(dephasing_rate) * np.array([[1, 0], [0, -1]], dtype=complex),
            coupling_strength=np.sqrt(dephasing_rate),
            description="site_1_dephasing"
        )
        
        lindblad_2 = LindbladOperator(
            matrix=np.sqrt(dephasing_rate) * np.array([[-1, 0], [0, 1]], dtype=complex),
            coupling_strength=np.sqrt(dephasing_rate),
            description="site_2_dephasing"
        )
        
        lindblad_ops = [lindblad_1, lindblad_2]
        
        # Initial state with coherence: (|1⟩ + |2⟩)/√2
        initial_matrix = 0.5 * np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ], dtype=complex)
        
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=["site_1", "site_2"],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def extract_observable(self, state: DensityMatrix) -> float:
        """Extract coherence magnitude |ρ₁₂|."""
        return float(abs(state.matrix[0, 1]))


class BenchmarkRunner:
    """
    Runner for executing benchmark test suites.
    
    Manages multiple benchmark systems and provides
    automated execution and reporting.
    """
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.benchmarks: List[BenchmarkSystem] = []
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, benchmark: BenchmarkSystem):
        """Add a benchmark system to the runner."""
        self.benchmarks.append(benchmark)
    
    def add_standard_benchmarks(self):
        """Add the standard suite of benchmark tests."""
        self.add_benchmark(TwoLevelSystemBenchmark())
        self.add_benchmark(HarmonicOscillatorBenchmark())
        self.add_benchmark(DampedTwoLevelSystemBenchmark())
        self.add_benchmark(PhotosyntheticComplexBenchmark())
    
    def run_all_benchmarks(self, final_time: float = 1.0, time_step: float = 0.01) -> List[BenchmarkResult]:
        """
        Run all benchmark tests.
        
        Args:
            final_time: Final simulation time for all tests
            time_step: Time step for evolution
            
        Returns:
            List of benchmark results
        """
        self.results = []
        
        for benchmark in self.benchmarks:
            print(f"Running benchmark: {benchmark.name}")
            try:
                result = benchmark.run_benchmark(final_time, time_step)
                self.results.append(result)
                
                status = "PASSED" if result.test_passed else "FAILED"
                print(f"  Status: {status}")
                print(f"  Relative Error: {result.relative_error:.2e}")
                print(f"  Computation Time: {result.computation_time:.3f}s")
                
                if result.error_message:
                    print(f"  Error: {result.error_message}")
                
            except Exception as e:
                print(f"  FAILED with exception: {str(e)}")
                # Create failed result
                failed_result = BenchmarkResult(
                    system_name=benchmark.name,
                    test_passed=False,
                    numerical_result=0.0,
                    analytical_result=0.0,
                    relative_error=float('inf'),
                    absolute_error=float('inf'),
                    computation_time=0.0,
                    tolerance=benchmark.tolerance,
                    error_message=str(e)
                )
                self.results.append(failed_result)
            
            print()
        
        return self.results
    
    def run_benchmarks(self, final_time: float = 1.0, time_step: float = 0.01) -> List[BenchmarkResult]:
        """
        Alias for run_all_benchmarks method for backward compatibility.
        
        Args:
            final_time: Final simulation time for all tests
            time_step: Time step for evolution
            
        Returns:
            List of benchmark results
        """
        return self.run_all_benchmarks(final_time, time_step)
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.test_passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100
        
        report = []
        report.append("=" * 60)
        report.append("QBES Benchmark Test Report")
        report.append("=" * 60)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {failed_tests}")
        report.append(f"Success Rate: {success_rate:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("Detailed Results:")
        report.append("-" * 40)
        
        for result in self.results:
            status = "PASS" if result.test_passed else "FAIL"
            report.append(f"{result.system_name}: {status}")
            report.append(f"  Numerical: {result.numerical_result:.6e}")
            report.append(f"  Analytical: {result.analytical_result:.6e}")
            report.append(f"  Relative Error: {result.relative_error:.2e}")
            report.append(f"  Computation Time: {result.computation_time:.3f}s")
            
            if result.error_message:
                report.append(f"  Error: {result.error_message}")
            
            report.append("")
        
        # Performance summary
        total_time = sum(r.computation_time for r in self.results)
        avg_time = total_time / total_tests
        
        report.append("Performance Summary:")
        report.append("-" * 20)
        report.append(f"Total Computation Time: {total_time:.3f}s")
        report.append(f"Average Time per Test: {avg_time:.3f}s")
        
        return "\n".join(report)
    
    def get_failed_tests(self) -> List[BenchmarkResult]:
        """Get list of failed benchmark tests."""
        return [r for r in self.results if not r.test_passed]
    
    def performance_scaling_test(self, system_sizes: List[int] = None) -> Dict[int, float]:
        """
        Test computational scaling with system size.
        
        Args:
            system_sizes: List of system sizes to test
            
        Returns:
            Dictionary mapping system size to computation time
        """
        if system_sizes is None:
            system_sizes = [2, 4, 8, 16]
        
        scaling_results = {}
        
        for size in system_sizes:
            print(f"Testing system size: {size}")
            
            # Create harmonic oscillator with specified number of levels
            benchmark = HarmonicOscillatorBenchmark(n_levels=size)
            
            # Run short benchmark
            start_time = time.time()
            result = benchmark.run_benchmark(final_time=0.1, time_step=0.01)
            computation_time = time.time() - start_time
            
            scaling_results[size] = computation_time
            print(f"  Time: {computation_time:.3f}s")
        
        return scaling_results


def run_quick_benchmarks() -> BenchmarkRunner:
    """
    Run a quick benchmark suite for validation.
    
    Returns:
        BenchmarkRunner with results
    """
    print("Running QBES Quick Benchmark Suite")
    print("=" * 40)
    
    runner = BenchmarkRunner()
    runner.add_standard_benchmarks()
    
    # Run with shorter time for quick validation
    runner.run_all_benchmarks(final_time=0.5, time_step=0.01)
    
    print("\nBenchmark Summary:")
    print(runner.generate_report())
    
    return runner