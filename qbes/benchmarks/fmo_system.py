"""
Fenna-Matthews-Olson (FMO) complex benchmark with literature validation.

This module implements the FMO complex benchmark system for validating QBES
against published experimental and theoretical results. The FMO complex is
a well-studied photosynthetic antenna complex with known coherence dynamics.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time
import json
import os

from ..core.data_models import (
    SimulationConfig, QuantumSubsystem, DensityMatrix, 
    Hamiltonian, LindbladOperator
)
from ..quantum_engine import QuantumEngine
from .analytical_systems import AnalyticalBenchmarkResult


@dataclass
class FMOBenchmarkResult:
    """Results from FMO complex benchmark test."""
    system_name: str
    test_passed: bool
    computed_coherence_lifetime: float
    reference_coherence_lifetime: float
    computed_transfer_efficiency: float
    reference_transfer_efficiency: float
    coherence_error: float
    efficiency_error: float
    computation_time: float
    tolerance: float
    temperature: float
    error_message: Optional[str] = None


class FMOComplexBenchmark:
    """
    Fenna-Matthews-Olson complex benchmark system.
    
    The FMO complex is a photosynthetic antenna complex found in green sulfur bacteria.
    It consists of 7 bacteriochlorophyll-a (BChl-a) molecules arranged in a specific
    geometry that facilitates efficient energy transfer from the antenna to the
    reaction center.
    
    This benchmark validates QBES against published experimental results for:
    1. Coherence lifetime at different temperatures
    2. Energy transfer efficiency
    3. Population dynamics
    
    References:
    - Engel et al. Nature 447, 782-786 (2007) - Coherence measurements
    - Mohseni et al. J. Chem. Phys. 129, 174106 (2008) - Efficiency studies
    - Adolphs & Renger, Biophys. J. 91, 2778-2797 (2006) - Structure and energetics
    """
    
    def __init__(self, temperature: float = 77.0, tolerance: float = 0.15):
        """
        Initialize FMO complex benchmark.
        
        Args:
            temperature: Temperature in Kelvin (77K for low-temp experiments)
            tolerance: Relative tolerance for comparison with literature (15% default)
        """
        self.name = "FMO Complex Benchmark"
        self.temperature = temperature
        self.tolerance = tolerance
        self.quantum_engine = QuantumEngine()
        
        # FMO system parameters from literature
        self.num_sites = 7  # 7 BChl-a molecules
        self.reorganization_energy = 35.0  # cm^-1, typical for BChl-a
        
        # Load reference data
        self._load_reference_data()
        
        # FMO Hamiltonian from Adolphs & Renger (2006)
        # Site energies in cm^-1 (converted to eV for calculations)
        self.site_energies = np.array([
            12410, 12530, 12210, 12320, 12480, 12630, 12440
        ]) * 1.24e-4  # Convert cm^-1 to eV
        
        # Electronic coupling matrix (cm^-1, converted to eV)
        self.coupling_matrix = np.array([
            [0,    -87.7, 5.5,   -5.9,  6.7,   -13.7, -9.9],
            [-87.7, 0,    30.8,  8.2,   0.7,   11.8,  4.3],
            [5.5,   30.8, 0,     -53.5, -2.2,  -9.6,  6.0],
            [-5.9,  8.2,  -53.5, 0,     -70.7, -17.0, -63.3],
            [6.7,   0.7,  -2.2,  -70.7, 0,     81.1,  -1.3],
            [-13.7, 11.8, -9.6,  -17.0, 81.1,  0,     39.7],
            [-9.9,  4.3,  6.0,   -63.3, -1.3,  39.7,  0]
        ]) * 1.24e-4  # Convert cm^-1 to eV
    
    def _load_reference_data(self):
        """Load reference data from JSON file."""
        try:
            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            reference_file = os.path.join(current_dir, "reference_data.json")
            
            with open(reference_file, 'r') as f:
                data = json.load(f)
            
            # Extract FMO-specific reference values
            self.ref_coherence_lifetime = data["fmo_coherence_lifetime_fs"]["value"]
            self.ref_transfer_efficiency = data["fmo_energy_transfer_efficiency"]["value"]
            
        except (FileNotFoundError, KeyError) as e:
            # Use default values if reference file not found
            self.ref_coherence_lifetime = 660.0  # fs, from Engel et al.
            self.ref_transfer_efficiency = 0.95   # from Mohseni et al.
    
    def setup_system(self) -> Tuple[Hamiltonian, List[LindbladOperator], DensityMatrix]:
        """
        Set up the FMO quantum system.
        
        Returns:
            Tuple of (Hamiltonian, Lindblad operators, initial state)
        """
        # Create system Hamiltonian
        hamiltonian_matrix = np.diag(self.site_energies) + self.coupling_matrix
        
        hamiltonian = Hamiltonian(
            matrix=hamiltonian_matrix.astype(complex),
            basis_labels=[f"BChl_{i+1}" for i in range(self.num_sites)],
            time_dependent=False
        )
        
        # Create Lindblad operators for environmental decoherence
        lindblad_ops = self._create_decoherence_operators()
        
        # Initial state: excitation on site 1 (antenna side)
        # This represents initial excitation from light harvesting antenna
        initial_matrix = np.zeros((self.num_sites, self.num_sites), dtype=complex)
        initial_matrix[0, 0] = 1.0  # |1⟩⟨1| - excitation on BChl_1
        
        initial_state = DensityMatrix(
            matrix=initial_matrix,
            basis_labels=[f"BChl_{i+1}" for i in range(self.num_sites)],
            time=0.0
        )
        
        return hamiltonian, lindblad_ops, initial_state
    
    def _create_decoherence_operators(self) -> List[LindbladOperator]:
        """
        Create Lindblad operators for environmental decoherence.
        
        Models both pure dephasing and population relaxation based on
        the spectral density of the protein environment.
        """
        lindblad_ops = []
        
        # Calculate decoherence rates based on temperature and reorganization energy
        # Using simple Ohmic spectral density model
        kT = 8.617e-5 * self.temperature  # eV (Boltzmann constant * temperature)
        
        # Pure dephasing rate (approximate)
        dephasing_rate = self.reorganization_energy * 1.24e-4 / (2 * kT) if kT > 0 else 0.1
        dephasing_rate = min(dephasing_rate, 1.0)  # Cap at reasonable value
        
        # Population relaxation rate (smaller than dephasing)
        relaxation_rate = dephasing_rate * 0.1
        
        # Pure dephasing operators: σz for each site
        for i in range(self.num_sites):
            dephasing_op = np.zeros((self.num_sites, self.num_sites), dtype=complex)
            dephasing_op[i, i] = 1.0
            
            lindblad_ops.append(LindbladOperator(
                operator=np.sqrt(dephasing_rate) * dephasing_op,
                coupling_strength=np.sqrt(dephasing_rate),
                operator_type="pure_dephasing"
            ))
        
        # Population relaxation operators: σ- for each site
        for i in range(self.num_sites):
            for j in range(i):  # Only downward transitions
                relaxation_op = np.zeros((self.num_sites, self.num_sites), dtype=complex)
                relaxation_op[j, i] = 1.0  # |j⟩⟨i| with j < i (lower energy)
                
                # Rate depends on energy gap
                energy_gap = abs(self.site_energies[i] - self.site_energies[j])
                rate = relaxation_rate * np.exp(-energy_gap / (2 * kT)) if kT > 0 else relaxation_rate
                
                lindblad_ops.append(LindbladOperator(
                    operator=np.sqrt(rate) * relaxation_op,
                    coupling_strength=np.sqrt(rate),
                    operator_type="population_relaxation"
                ))
        
        return lindblad_ops
    
    def calculate_coherence_lifetime(self, states: List[DensityMatrix], 
                                   times: List[float]) -> float:
        """
        Calculate coherence lifetime from time evolution data.
        
        Coherence lifetime is extracted from the decay of off-diagonal
        elements of the density matrix, which represent quantum coherences.
        
        Args:
            states: List of density matrices at different times
            times: Corresponding time points
            
        Returns:
            Coherence lifetime in femtoseconds
        """
        if len(states) < 2:
            return 0.0
        
        # Calculate coherence measure: sum of absolute values of off-diagonal elements
        coherences = []
        for state in states:
            # Sum |ρ_ij| for i ≠ j
            coherence = 0.0
            for i in range(self.num_sites):
                for j in range(i+1, self.num_sites):
                    coherence += abs(state.matrix[i, j]) + abs(state.matrix[j, i])
            coherences.append(coherence)
        
        # Fit exponential decay: C(t) = C(0) * exp(-t/τ)
        if coherences[0] == 0:
            return 0.0
        
        # Find time when coherence drops to 1/e of initial value
        target_coherence = coherences[0] / np.e
        
        for i, coherence in enumerate(coherences):
            if coherence <= target_coherence:
                if i == 0:
                    return times[0] if times[0] > 0 else 1.0
                # Linear interpolation between points
                t1, t2 = times[i-1], times[i]
                c1, c2 = coherences[i-1], coherences[i]
                
                if c1 == c2:
                    return t1
                
                # Interpolate to find exact crossing time
                lifetime = t1 + (t2 - t1) * (target_coherence - c1) / (c2 - c1)
                return lifetime  # Already in femtoseconds
        
        # If coherence doesn't decay enough, estimate from slope
        if len(times) >= 2 and coherences[0] > 0:
            # Use first two points to estimate decay rate
            dt = times[1] - times[0]
            dc = coherences[1] - coherences[0]
            if dc < 0:  # Decaying
                decay_rate = -dc / (coherences[0] * dt)
                lifetime = 1.0 / decay_rate if decay_rate > 0 else 1000.0
                return lifetime  # Already in femtoseconds
        
        return 1000.0  # Default long lifetime in fs
    
    def calculate_transfer_efficiency(self, final_state: DensityMatrix) -> float:
        """
        Calculate energy transfer efficiency to the reaction center.
        
        Efficiency is measured as the population that reaches sites 3 and 6,
        which are closest to the reaction center based on the FMO structure.
        
        Args:
            final_state: Final density matrix after evolution
            
        Returns:
            Transfer efficiency (0 to 1)
        """
        # Sites 3 and 6 (indices 2 and 5) are closest to reaction center
        rc_sites = [2, 5]  # BChl_3 and BChl_6
        
        efficiency = 0.0
        for site in rc_sites:
            efficiency += np.real(final_state.matrix[site, site])
        
        return efficiency
    
    def run_benchmark(self, simulation_time: float = 1000.0, 
                     time_step: float = 1.0) -> FMOBenchmarkResult:
        """
        Run FMO complex benchmark test.
        
        Args:
            simulation_time: Total simulation time in femtoseconds
            time_step: Time step for evolution in femtoseconds
            
        Returns:
            FMOBenchmarkResult with comparison to literature values
        """
        start_time = time.time()
        
        try:
            # Set up system
            hamiltonian, lindblad_ops, initial_state = self.setup_system()
            
            # Convert time units (fs to atomic units for simulation)
            dt_au = time_step * 4.134e-2  # fs to atomic time units
            total_time_au = simulation_time * 4.134e-2
            
            # Evolve system and collect states
            current_state = initial_state
            current_time = 0.0
            states = [current_state]
            times = [0.0]
            
            while current_time < total_time_au:
                current_state = self.quantum_engine.evolve_state(
                    current_state, dt_au, hamiltonian, lindblad_ops
                )
                current_time += dt_au
                states.append(current_state)
                times.append(current_time / 4.134e-2)  # Convert back to fs
            
            # Calculate observables
            computed_coherence_lifetime = self.calculate_coherence_lifetime(states, times)
            computed_transfer_efficiency = self.calculate_transfer_efficiency(states[-1])
            
            # Calculate errors
            coherence_error = abs(computed_coherence_lifetime - self.ref_coherence_lifetime) / self.ref_coherence_lifetime
            efficiency_error = abs(computed_transfer_efficiency - self.ref_transfer_efficiency) / self.ref_transfer_efficiency
            
            # Check if test passed (both metrics within tolerance)
            test_passed = (coherence_error < self.tolerance and 
                          efficiency_error < self.tolerance)
            
            computation_time = time.time() - start_time
            
            return FMOBenchmarkResult(
                system_name=self.name,
                test_passed=test_passed,
                computed_coherence_lifetime=computed_coherence_lifetime,
                reference_coherence_lifetime=self.ref_coherence_lifetime,
                computed_transfer_efficiency=computed_transfer_efficiency,
                reference_transfer_efficiency=self.ref_transfer_efficiency,
                coherence_error=coherence_error,
                efficiency_error=efficiency_error,
                computation_time=computation_time,
                tolerance=self.tolerance,
                temperature=self.temperature
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            return FMOBenchmarkResult(
                system_name=self.name,
                test_passed=False,
                computed_coherence_lifetime=0.0,
                reference_coherence_lifetime=self.ref_coherence_lifetime,
                computed_transfer_efficiency=0.0,
                reference_transfer_efficiency=self.ref_transfer_efficiency,
                coherence_error=float('inf'),
                efficiency_error=float('inf'),
                computation_time=computation_time,
                tolerance=self.tolerance,
                temperature=self.temperature,
                error_message=str(e)
            )


def create_fmo_benchmark_suite() -> List[FMOComplexBenchmark]:
    """
    Create a suite of FMO benchmarks at different temperatures.
    
    Returns:
        List of FMO benchmark systems for validation
    """
    benchmarks = [
        FMOComplexBenchmark(temperature=77.0, tolerance=0.15),   # Low temperature
        FMOComplexBenchmark(temperature=300.0, tolerance=0.20),  # Room temperature
    ]
    
    return benchmarks


def run_fmo_benchmarks(simulation_time: float = 1000.0, 
                      time_step: float = 1.0) -> List[FMOBenchmarkResult]:
    """
    Run all FMO benchmark tests.
    
    Args:
        simulation_time: Total simulation time in femtoseconds
        time_step: Time step for evolution in femtoseconds
        
    Returns:
        List of FMO benchmark results
    """
    benchmarks = create_fmo_benchmark_suite()
    results = []
    
    print("Running FMO Complex Benchmark Suite")
    print("=" * 50)
    
    for benchmark in benchmarks:
        print(f"\nRunning: {benchmark.name} at {benchmark.temperature}K")
        print("-" * 40)
        
        try:
            result = benchmark.run_benchmark(simulation_time, time_step)
            results.append(result)
            
            status = "PASSED" if result.test_passed else "FAILED"
            print(f"Status: {status}")
            print(f"Temperature: {result.temperature}K")
            print(f"Coherence Lifetime: {result.computed_coherence_lifetime:.1f} fs "
                  f"(ref: {result.reference_coherence_lifetime:.1f} fs)")
            print(f"Transfer Efficiency: {result.computed_transfer_efficiency:.3f} "
                  f"(ref: {result.reference_transfer_efficiency:.3f})")
            print(f"Coherence Error: {result.coherence_error:.1%}")
            print(f"Efficiency Error: {result.efficiency_error:.1%}")
            print(f"Computation Time: {result.computation_time:.3f}s")
            
            if result.error_message:
                print(f"Error: {result.error_message}")
                
        except Exception as e:
            print(f"FAILED with exception: {str(e)}")
            # Create failed result
            failed_result = FMOBenchmarkResult(
                system_name=benchmark.name,
                test_passed=False,
                computed_coherence_lifetime=0.0,
                reference_coherence_lifetime=benchmark.ref_coherence_lifetime,
                computed_transfer_efficiency=0.0,
                reference_transfer_efficiency=benchmark.ref_transfer_efficiency,
                coherence_error=float('inf'),
                efficiency_error=float('inf'),
                computation_time=0.0,
                tolerance=benchmark.tolerance,
                temperature=benchmark.temperature,
                error_message=str(e)
            )
            results.append(failed_result)
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.test_passed)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print("FMO BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return results