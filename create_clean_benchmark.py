#!/usr/bin/env python3

# Create a clean benchmark file

content = '''"""
Benchmark test systems with known analytical solutions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time

from ..core.data_models import (
    DensityMatrix, Hamiltonian, LindbladOperator
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
        self.name = name
        self.tolerance = tolerance
        self.quantum_engine = QuantumEngine()
    
    @abstractmethod
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        pass
    
    @abstractmethod
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        pass
    
    @abstractmethod
    def extract_observable(self, state: DensityMatrix) -> float:
        pass
    
    def run_benchmark(self, final_time: float, time_step: float, **kwargs) -> BenchmarkResult:
        """Run benchmark test and compare with analytical solution."""
        start_time = time.time()
        
        try:
            hamiltonian, lindblad_ops, initial_state = self.setup_system()
            
            current_state = initial_state
            current_time = 0.0
            
            while current_time < final_time:
                current_state = self.quantum_engine.evolve_state(
                    current_state, time_step, hamiltonian, lindblad_ops
                )
                current_time += time_step
            
            numerical_result = self.extract_observable(current_state)
            analytical_result = self.get_analytical_solution(final_time, **kwargs)
            
            absolute_error = abs(numerical_result - analytical_result)
            relative_error = absolute_error / abs(analytical_result) if analytical_result != 0 else absolute_error
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
    """Benchmark for isolated two-level system."""
    
    def __init__(self, energy_gap: float = 1.0, tolerance: float = 1e-6):
        super().__init__("Two-Level System", tolerance)
        self.energy_gap = energy_gap
    
    def get_analytical_solution(self, time: float, **kwargs) -> float:
        omega = self.energy_gap
        return np.sin(omega * time / 2) ** 2
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        hamiltonian_matrix = (self.energy_gap / 2) * np.array([[0, 1], [1, 0]], dtype=complex)
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix,
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
        
        lindblad_ops = []
        
        initial_matrix = np.array([[1, 0], [0, 0]], dtype=complex)
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def extract_observable(self, state: DensityMatrix) -> float:
        return float(np.real(state.matrix[1, 1]))


class BenchmarkRunner:
    """Automated benchmark execution system."""
    
    def __init__(self):
        self.benchmarks = []
        self.results = []
    
    def add_benchmark(self, benchmark: BenchmarkSystem):
        self.benchmarks.append(benchmark)
    
    def add_standard_benchmarks(self):
        self.add_benchmark(TwoLevelSystemBenchmark())
    
    def run_all_benchmarks(self, final_time: float = 1.0, time_step: float = 0.01):
        """Run all benchmark tests."""
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


def run_quick_benchmarks():
    """Run standard benchmark suite and return runner with results."""
    runner = BenchmarkRunner()
    runner.add_standard_benchmarks()
    runner.run_all_benchmarks()
    return runner
'''

# Write the file
with open('qbes/benchmarks/benchmark_systems.py', 'w') as f:
    f.write(content)

print("Clean benchmark file created successfully")

# Verify
with open('qbes/benchmarks/benchmark_systems.py', 'r') as f:
    content_check = f.read()
    print(f"File size: {len(content_check)} characters")
    print(f"Lines: {len(content_check.splitlines())}")