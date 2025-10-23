"""
Analytical benchmark systems with known solutions.

This module implements quantum systems with exact analytical solutions
for validating the QBES simulation engine accuracy.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time
import math

from ..core.data_models import (
    SimulationConfig, QuantumSubsystem, DensityMatrix, 
    Hamiltonian, LindbladOperator
)
from ..quantum_engine import QuantumEngine


@dataclass
class AnalyticalBenchmarkResult:
    """Results from an analytical benchmark test."""
    system_name: str
    test_passed: bool
    numerical_result: float
    analytical_result: float
    relative_error: float
    absolute_error: float
    computation_time: float
    tolerance: float
    error_message: Optional[str] = None


class AnalyticalBenchmarkSystem(ABC):
    """Abstract base class for analytical benchmark test systems."""
    
    def __init__(self, name: str, tolerance: float = 1e-6):
        """
        Initialize analytical benchmark system.
        
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
    
    def run_benchmark(self, final_time: float, time_step: float, **kwargs) -> AnalyticalBenchmarkResult:
        """
        Run benchmark test and compare with analytical solution.
        
        Args:
            final_time: Final simulation time
            time_step: Time step for evolution
            **kwargs: Additional parameters for analytical solution
            
        Returns:
            AnalyticalBenchmarkResult with comparison results
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
            
            return AnalyticalBenchmarkResult(
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
            return AnalyticalBenchmarkResult(
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


class TwoLevelSystemBenchmark(AnalyticalBenchmarkSystem):
    """
    Two-level system with known analytical solution for Rabi oscillations.
    
    This benchmark tests basic quantum evolution without decoherence.
    The analytical solution for Rabi oscillations is well-known and provides
    a fundamental test of the quantum evolution algorithms.
    """
    
    def __init__(self, energy_gap: float = 1.0, rabi_frequency: float = 1.0, tolerance: float = 1e-6):
        """
        Initialize two-level system benchmark.
        
        Args:
            energy_gap: Energy difference between levels (in units of ħ)
            rabi_frequency: Rabi frequency for driving field
            tolerance: Numerical tolerance for comparison
        """
        super().__init__("Two-Level System Rabi Oscillations", tolerance)
        self.energy_gap = energy_gap
        self.rabi_frequency = rabi_frequency
    
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        """
        Analytical solution for excited state population in Rabi oscillations.
        
        For a two-level system starting in ground state with Hamiltonian:
        H = (ω/2) * σz + (Ω/2) * σx
        
        The excited state population is: P_e(t) = sin²(Ωt/2)
        where Ω is the Rabi frequency.
        """
        return np.sin(self.rabi_frequency * time / 2) ** 2
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up two-level quantum system with Rabi driving."""
        # Hamiltonian matrix: H = (ω/2) * σz + (Ω/2) * σx
        # σz = [[1, 0], [0, -1]], σx = [[0, 1], [1, 0]]
        hamiltonian_matrix = 0.5 * np.array([
            [self.energy_gap, self.rabi_frequency],
            [self.rabi_frequency, -self.energy_gap]
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


class HarmonicOscillatorBenchmark(AnalyticalBenchmarkSystem):
    """
    Harmonic oscillator with coherent state evolution.
    
    Tests quantum evolution of coherent states, which have known
    analytical solutions for position and momentum expectation values.
    This provides a test of multi-level quantum systems.
    """
    
    def __init__(self, frequency: float = 1.0, n_levels: int = 10, 
                 initial_displacement: float = 2.0, tolerance: float = 1e-5):
        """
        Initialize harmonic oscillator benchmark.
        
        Args:
            frequency: Oscillator frequency
            n_levels: Number of energy levels to include
            initial_displacement: Initial coherent state displacement
            tolerance: Numerical tolerance
        """
        super().__init__("Harmonic Oscillator Coherent State", tolerance)
        self.frequency = frequency
        self.n_levels = n_levels
        self.initial_displacement = initial_displacement
    
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        """
        Analytical solution for position expectation value of coherent state.
        
        For coherent state |α⟩, position expectation value is:
        ⟨x⟩(t) = √2 * Re(α * exp(-iωt))
        
        where α is the initial displacement and ω is the frequency.
        """
        alpha = self.initial_displacement
        alpha_t = alpha * np.exp(-1j * self.frequency * time)
        return np.sqrt(2) * np.real(alpha_t)
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up harmonic oscillator system."""
        # Create Hamiltonian matrix: H = ħω(a†a + 1/2)
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
        
        # Initial coherent state |α⟩ with α = initial_displacement
        initial_matrix = self._create_coherent_state_density_matrix(self.initial_displacement)
        
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=[f"n={n}" for n in range(self.n_levels)],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def _create_coherent_state_density_matrix(self, alpha: complex) -> np.ndarray:
        """
        Create density matrix for coherent state |α⟩.
        
        Coherent state coefficients: c_n = exp(-|α|²/2) * α^n / √(n!)
        For finite truncation, we need to renormalize.
        """
        coeffs = np.zeros(self.n_levels, dtype=complex)
        norm_factor = np.exp(-abs(alpha)**2 / 2)
        
        for n in range(self.n_levels):
            coeffs[n] = norm_factor * (alpha**n) / np.sqrt(math.factorial(n))
        
        # Renormalize for finite truncation
        norm = np.sqrt(np.sum(np.abs(coeffs)**2))
        coeffs = coeffs / norm
        
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


class DampedTwoLevelSystemBenchmark(AnalyticalBenchmarkSystem):
    """
    Two-level system with spontaneous emission (Lindblad evolution).
    
    Tests Lindblad evolution with known analytical solution
    for population decay. This validates the implementation
    of open quantum system dynamics.
    """
    
    def __init__(self, energy_gap: float = 1.0, decay_rate: float = 0.1, 
                 initial_population: float = 1.0, tolerance: float = 1e-5):
        """
        Initialize damped two-level system.
        
        Args:
            energy_gap: Energy difference between levels
            decay_rate: Spontaneous emission rate (γ)
            initial_population: Initial excited state population
            tolerance: Numerical tolerance
        """
        super().__init__("Damped Two-Level System", tolerance)
        self.energy_gap = energy_gap
        self.decay_rate = decay_rate
        self.initial_population = initial_population
    
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        """
        Analytical solution for excited state population decay.
        
        For spontaneous emission with Lindblad operator L = √γ * σ-:
        P_e(t) = P_e(0) * exp(-γt)
        
        where γ is the decay rate.
        """
        return self.initial_population * np.exp(-self.decay_rate * time)
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """Set up damped two-level system."""
        # Hamiltonian: H = (ω/2) * σz (no driving field)
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
        # σ- = |0⟩⟨1| = [[0, 1], [0, 0]]
        lindblad_matrix = np.sqrt(self.decay_rate) * np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ], dtype=complex)
        
        lindblad_op = LindbladOperator(
            operator=lindblad_matrix,
            coupling_strength=np.sqrt(self.decay_rate),
            operator_type="spontaneous_emission"
        )
        
        lindblad_ops = [lindblad_op]
        
        # Initial state: mixed state with specified excited population
        # ρ(0) = P_e(0) * |1⟩⟨1| + (1 - P_e(0)) * |0⟩⟨0|
        initial_matrix = np.array([
            [1.0 - self.initial_population, 0.0],
            [0.0, self.initial_population]
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


def create_analytical_benchmark_suite() -> List[AnalyticalBenchmarkSystem]:
    """
    Create a standard suite of analytical benchmark systems.
    
    Returns:
        List of analytical benchmark systems for validation
    """
    benchmarks = [
        TwoLevelSystemBenchmark(
            energy_gap=1.0,
            rabi_frequency=1.0,
            tolerance=1e-6
        ),
        HarmonicOscillatorBenchmark(
            frequency=1.0,
            n_levels=10,
            initial_displacement=2.0,
            tolerance=1e-5
        ),
        DampedTwoLevelSystemBenchmark(
            energy_gap=1.0,
            decay_rate=0.1,
            initial_population=1.0,
            tolerance=1e-5
        )
    ]
    
    return benchmarks


def run_analytical_benchmarks(final_time: float = 1.0, time_step: float = 0.01) -> List[AnalyticalBenchmarkResult]:
    """
    Run all analytical benchmark tests.
    
    Args:
        final_time: Final simulation time for all tests
        time_step: Time step for evolution
        
    Returns:
        List of analytical benchmark results
    """
    benchmarks = create_analytical_benchmark_suite()
    results = []
    
    print("Running Analytical Benchmark Suite")
    print("=" * 50)
    
    for benchmark in benchmarks:
        print(f"\nRunning: {benchmark.name}")
        print("-" * 30)
        
        try:
            result = benchmark.run_benchmark(final_time, time_step)
            results.append(result)
            
            status = "PASSED" if result.test_passed else "FAILED"
            print(f"Status: {status}")
            print(f"Numerical Result: {result.numerical_result:.8f}")
            print(f"Analytical Result: {result.analytical_result:.8f}")
            print(f"Relative Error: {result.relative_error:.2e}")
            print(f"Computation Time: {result.computation_time:.3f}s")
            
            if result.error_message:
                print(f"Error: {result.error_message}")
                
        except Exception as e:
            print(f"FAILED with exception: {str(e)}")
            # Create failed result
            failed_result = AnalyticalBenchmarkResult(
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
            results.append(failed_result)
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.test_passed)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print("ANALYTICAL BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return results