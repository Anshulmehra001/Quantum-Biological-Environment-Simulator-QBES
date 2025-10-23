"""
Literature benchmarks for QBES validation.

This module provides comparison with published quantum biology results,
including FMO complex, photosystem II, and other well-studied systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

try:
    from ...core.data_models import SimulationResults
except ImportError:
    SimulationResults = None

try:
    from ...validation.validator import ValidationResult
except ImportError:
    # Fallback validation result
    from dataclasses import dataclass, field
    
    @dataclass
    class ValidationResult:
        is_valid: bool = True
        errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        
        def add_error(self, error: str):
            self.errors.append(error)
        
        def add_warning(self, warning: str):
            self.warnings.append(warning)


@dataclass
class LiteratureBenchmark:
    """Literature benchmark data."""
    name: str
    system: str
    reference: str
    year: int
    parameters: Dict
    results: Dict[str, np.ndarray]
    uncertainty: Optional[Dict[str, float]] = None
    conditions: Optional[Dict] = None


class LiteratureBenchmarks:
    """
    Collection of literature benchmarks for validation.
    
    Includes published results from:
    - FMO complex (Engel et al., 2007)
    - Photosystem II reaction center
    - DNA base pairs (quantum tunneling)
    - Enzyme active sites
    """
    
    def __init__(self):
        """Initialize literature benchmarks."""
        self.logger = logging.getLogger(__name__)
        self.benchmarks = self._load_benchmarks()
    
    def _load_benchmarks(self) -> Dict[str, LiteratureBenchmark]:
        """Load all literature benchmarks."""
        benchmarks = {}
        
        # FMO Complex from Engel et al. Nature 2007
        benchmarks['fmo_engel_2007'] = self._fmo_engel_2007()
        
        # FMO Complex from Ishizaki & Fleming PNAS 2009
        benchmarks['fmo_ishizaki_2009'] = self._fmo_ishizaki_2009()
        
        # Photosystem II from Romero et al. Nature Physics 2014
        benchmarks['psii_romero_2014'] = self._psii_romero_2014()
        
        # Two-level system (analytical)
        benchmarks['tls_analytical'] = self._two_level_analytical()
        
        return benchmarks
    
    def _fmo_engel_2007(self) -> LiteratureBenchmark:
        """
        FMO complex benchmark from Engel et al., Nature 446, 782-786 (2007).
        
        Famous paper demonstrating long-lived quantum coherence in 
        photosynthetic light harvesting at physiological temperatures.
        """
        # Site energies (cm^-1) from spectroscopy
        site_energies = np.array([
            12410, 12530, 12210, 12320, 12480, 12630, 12440
        ])
        
        # Coupling matrix (cm^-1) - symmetric 7x7
        # From Adolphs & Renger (2006) fit to spectroscopy
        couplings = np.array([
            [0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
            [-87.7, 0, 30.8, 8.2, 0.7, 11.8, 4.3],
            [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6.0],
            [-5.9, 8.2, -53.5, 0, -70.7, -17.0, -63.3],
            [6.7, 0.7, -2.2, -70.7, 0, 81.1, -1.3],
            [-13.7, 11.8, -9.6, -17.0, 81.1, 0, 39.7],
            [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 0]
        ])
        
        # Observed coherence lifetime ~660 fs at 77K
        coherence_lifetime = 660e-15  # seconds
        
        # Energy transfer time to reaction center ~1 ps
        transfer_time = 1e-12  # seconds
        
        # Temperature
        temperature = 77.0  # Kelvin
        
        return LiteratureBenchmark(
            name="FMO Complex - Engel et al. 2007",
            system="fmo_complex",
            reference="Nature 446, 782-786 (2007)",
            year=2007,
            parameters={
                'site_energies': site_energies,
                'couplings': couplings,
                'temperature': temperature,
                'n_sites': 7
            },
            results={
                'coherence_lifetime': np.array([coherence_lifetime]),
                'transfer_time': np.array([transfer_time]),
                'quantum_efficiency': np.array([0.95])  # Near unity
            },
            uncertainty={
                'coherence_lifetime': 60e-15,  # ±60 fs
                'transfer_time': 0.1e-12,
                'quantum_efficiency': 0.05
            },
            conditions={
                'temperature': temperature,
                'wavelength': '825nm',
                'excitation': 'two_photon_echo'
            }
        )
    
    def _fmo_ishizaki_2009(self) -> LiteratureBenchmark:
        """
        FMO complex benchmark from Ishizaki & Fleming, PNAS 106, 17255 (2009).
        
        Theoretical study with optimized spectral density parameters.
        """
        # Same site energies as Engel
        site_energies = np.array([
            12410, 12530, 12210, 12320, 12480, 12630, 12440
        ])
        
        couplings = np.array([
            [0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
            [-87.7, 0, 30.8, 8.2, 0.7, 11.8, 4.3],
            [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6.0],
            [-5.9, 8.2, -53.5, 0, -70.7, -17.0, -63.3],
            [6.7, 0.7, -2.2, -70.7, 0, 81.1, -1.3],
            [-13.7, 11.8, -9.6, -17.0, 81.1, 0, 39.7],
            [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 0]
        ])
        
        # Spectral density parameters (optimized)
        lambda_reorg = 35.0  # cm^-1 reorganization energy
        omega_cutoff = 166.0  # cm^-1 cutoff frequency
        
        return LiteratureBenchmark(
            name="FMO Complex - Ishizaki & Fleming 2009",
            system="fmo_complex",
            reference="PNAS 106, 17255-17260 (2009)",
            year=2009,
            parameters={
                'site_energies': site_energies,
                'couplings': couplings,
                'lambda': lambda_reorg,
                'omega_c': omega_cutoff,
                'temperature': 300.0
            },
            results={
                'population_transfer': np.array([0.85]),
                'transfer_efficiency': np.array([0.99]),
                'transfer_time': np.array([1.2e-12])
            },
            conditions={
                'temperature': 300.0,
                'spectral_density': 'ohmic_exponential'
            }
        )
    
    def _psii_romero_2014(self) -> LiteratureBenchmark:
        """
        Photosystem II from Romero et al., Nature Physics 10, 676 (2014).
        
        Long-lived quantum coherence at room temperature.
        """
        # Simplified model parameters
        site_energies = np.array([15000, 15100, 14900, 15050])  # cm^-1
        
        couplings = np.array([
            [0, 100, 20, 10],
            [100, 0, 80, 15],
            [20, 80, 0, 90],
            [10, 15, 90, 0]
        ])
        
        # Coherence lifetime ~400 fs at 294K
        coherence_lifetime = 400e-15
        
        return LiteratureBenchmark(
            name="PSII Reaction Center - Romero et al. 2014",
            system="photosystem_ii",
            reference="Nature Physics 10, 676-682 (2014)",
            year=2014,
            parameters={
                'site_energies': site_energies,
                'couplings': couplings,
                'temperature': 294.0,
                'n_sites': 4
            },
            results={
                'coherence_lifetime': np.array([coherence_lifetime]),
                'quantum_yield': np.array([0.85])
            },
            uncertainty={
                'coherence_lifetime': 50e-15
            },
            conditions={
                'temperature': 294.0,
                'technique': '2d_electronic_spectroscopy'
            }
        )
    
    def _two_level_analytical(self) -> LiteratureBenchmark:
        """
        Analytical two-level system for validation.
        
        Rabi oscillations with dephasing.
        """
        # Simple two-level system
        omega = 1.0e13  # rad/s (Rabi frequency)
        gamma = 1.0e12  # 1/s (dephasing rate)
        
        # Analytical solution for Rabi oscillations with damping
        time_points = np.linspace(0, 10e-12, 1000)
        
        # Population in excited state: P(t) = sin²(Ωt) * exp(-γt)
        population = np.sin(omega * time_points)**2 * np.exp(-gamma * time_points)
        
        # Coherence decays as: C(t) = C₀ * exp(-γt/2)
        coherence = np.exp(-gamma * time_points / 2)
        
        return LiteratureBenchmark(
            name="Two-Level System - Analytical",
            system="two_level_system",
            reference="Textbook (Sakurai QM)",
            year=1994,
            parameters={
                'rabi_frequency': omega,
                'dephasing_rate': gamma,
                'temperature': 0.0
            },
            results={
                'time': time_points,
                'population': population,
                'coherence': coherence
            },
            conditions={
                'initial_state': 'ground',
                'driving': 'resonant'
            }
        )
    
    def compare_with_literature(self,
                               simulation_results,
                               benchmark_name: str) -> Dict[str, float]:
        """
        Compare simulation results with literature benchmark.
        
        Args:
            simulation_results: Results from QBES simulation
            benchmark_name: Name of literature benchmark to compare with
            
        Returns:
            Dictionary of comparison metrics and errors
        """
        if benchmark_name not in self.benchmarks:
            self.logger.error(f"Benchmark '{benchmark_name}' not found")
            return {}
        
        benchmark = self.benchmarks[benchmark_name]
        comparison = {}
        
        # Compare coherence lifetime
        if 'coherence_lifetime' in benchmark.results:
            lit_lifetime = benchmark.results['coherence_lifetime'][0]
            
            # Extract from simulation
            if hasattr(simulation_results, 'coherence_metrics'):
                sim_lifetime = getattr(simulation_results.coherence_metrics, 
                                      'coherence_lifetime', None)
                if sim_lifetime:
                    relative_error = abs(sim_lifetime - lit_lifetime) / lit_lifetime
                    comparison['coherence_lifetime_error'] = relative_error
                    comparison['coherence_lifetime_match'] = relative_error < 0.3  # 30% tolerance
        
        # Compare transfer time
        if 'transfer_time' in benchmark.results:
            lit_transfer = benchmark.results['transfer_time'][0]
            
            if hasattr(simulation_results, 'transfer_time'):
                sim_transfer = simulation_results.transfer_time
                relative_error = abs(sim_transfer - lit_transfer) / lit_transfer
                comparison['transfer_time_error'] = relative_error
                comparison['transfer_time_match'] = relative_error < 0.3
        
        # Compare quantum efficiency
        if 'quantum_efficiency' in benchmark.results:
            lit_efficiency = benchmark.results['quantum_efficiency'][0]
            
            if hasattr(simulation_results, 'quantum_efficiency'):
                sim_efficiency = simulation_results.quantum_efficiency
                absolute_error = abs(sim_efficiency - lit_efficiency)
                comparison['efficiency_error'] = absolute_error
                comparison['efficiency_match'] = absolute_error < 0.1
        
        return comparison
    
    def generate_benchmark_report(self, 
                                 comparisons: Dict[str, Dict[str, float]]) -> str:
        """
        Generate literature comparison report.
        
        Args:
            comparisons: Dictionary of benchmark comparisons
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("LITERATURE BENCHMARK COMPARISON")
        report.append("="*70)
        report.append("")
        
        for benchmark_name, metrics in comparisons.items():
            if benchmark_name in self.benchmarks:
                benchmark = self.benchmarks[benchmark_name]
                report.append(f"Benchmark: {benchmark.name}")
                report.append(f"Reference: {benchmark.reference}")
                report.append("")
                
                for metric, value in metrics.items():
                    if isinstance(value, bool):
                        status = "✅ PASS" if value else "❌ FAIL"
                        report.append(f"  {metric}: {status}")
                    else:
                        report.append(f"  {metric}: {value:.4f}")
                
                report.append("")
        
        report.append("="*70)
        return "\n".join(report)
    
    def get_benchmark(self, name: str) -> Optional[LiteratureBenchmark]:
        """Get specific benchmark by name."""
        return self.benchmarks.get(name)
    
    def list_benchmarks(self) -> List[str]:
        """List all available benchmarks."""
        return list(self.benchmarks.keys())


# Convenience function
def validate_against_literature(results, 
                               benchmark: str = 'fmo_engel_2007') -> ValidationResult:
    """
    Validate simulation against literature benchmark.
    
    Args:
        results: Simulation results
        benchmark: Name of literature benchmark
        
    Returns:
        ValidationResult with literature comparison
    """
    lit_bench = LiteratureBenchmarks()
    comparison = lit_bench.compare_with_literature(results, benchmark)
    
    validation = ValidationResult(is_valid=True)
    
    for metric, value in comparison.items():
        if metric.endswith('_match'):
            if not value:
                validation.is_valid = False
                validation.add_error(f"Failed literature comparison: {metric}")
        elif metric.endswith('_error'):
            if value > 0.5:  # 50% error threshold
                validation.add_warning(f"Large error in {metric}: {value:.2%}")
    
    return validation
