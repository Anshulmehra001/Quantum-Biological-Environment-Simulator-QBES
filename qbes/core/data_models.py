"""
Core data models and structures for QBES.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class SimulationConfig:
    """Configuration parameters for a quantum biological simulation."""
    system_pdb: str
    temperature: float
    simulation_time: float
    time_step: float
    quantum_subsystem_selection: str
    noise_model_type: str
    output_directory: str
    force_field: str = "amber14"
    solvent_model: str = "tip3p"
    ionic_strength: float = 0.15
    # Enhanced debugging parameters
    debug_level: str = "INFO"
    save_snapshot_interval: int = 0  # 0 = disabled, >0 = save every N steps
    enable_sanity_checks: bool = True
    dry_run_mode: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.simulation_time <= 0:
            raise ValueError("Simulation time must be positive")
        if self.time_step <= 0:
            raise ValueError("Time step must be positive")
        if self.debug_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError("debug_level must be one of: DEBUG, INFO, WARNING, ERROR")
        if self.save_snapshot_interval < 0:
            raise ValueError("save_snapshot_interval must be non-negative")


@dataclass
class Atom:
    """Represents an atom in the molecular system."""
    element: str
    position: np.ndarray
    charge: float
    mass: float
    atom_id: int
    residue_id: int = 0
    residue_name: str = ""
    
    def __post_init__(self):
        """Validate atom data."""
        if len(self.position) != 3:
            raise ValueError("Position must be a 3D vector")


@dataclass
class QuantumState:
    """Represents a quantum state in the system."""
    coefficients: np.ndarray
    basis_labels: List[str]
    energy: Optional[float] = None
    
    def __post_init__(self):
        """Validate quantum state."""
        if len(self.coefficients) != len(self.basis_labels):
            raise ValueError("Coefficients and basis labels must have same length")
        
        # Check normalization
        norm = np.linalg.norm(self.coefficients)
        if not np.isclose(norm, 1.0, rtol=1e-10):
            raise ValueError(f"State is not normalized: norm = {norm}")


@dataclass
class DensityMatrix:
    """Represents a quantum density matrix."""
    matrix: np.ndarray
    basis_labels: List[str]
    time: float = 0.0
    
    def __post_init__(self):
        """Validate density matrix properties."""
        n = len(self.basis_labels)
        if self.matrix.shape != (n, n):
            raise ValueError(f"Matrix shape {self.matrix.shape} doesn't match basis size {n}")
        
        # Check Hermiticity
        if not np.allclose(self.matrix, self.matrix.conj().T):
            raise ValueError("Density matrix is not Hermitian")
        
        # Check trace = 1
        trace = np.trace(self.matrix)
        if not np.isclose(trace, 1.0, rtol=1e-10):
            raise ValueError(f"Density matrix trace is not 1: trace = {trace}")
        
        # Check positive semidefinite
        eigenvals = np.linalg.eigvals(self.matrix)
        if np.any(eigenvals < -1e-10):
            raise ValueError("Density matrix has negative eigenvalues")


@dataclass
class QuantumSubsystem:
    """Represents the quantum mechanical subsystem within the biological environment."""
    atoms: List[Atom]
    hamiltonian_parameters: Dict[str, float]
    coupling_matrix: np.ndarray
    basis_states: List[QuantumState]
    subsystem_id: str = ""
    
    def __post_init__(self):
        """Validate quantum subsystem."""
        n_states = len(self.basis_states)
        if self.coupling_matrix.shape != (n_states, n_states):
            raise ValueError("Coupling matrix size doesn't match number of basis states")


@dataclass
class MolecularSystem:
    """Represents the complete molecular system from PDB."""
    atoms: List[Atom]
    bonds: List[tuple]
    residues: Dict[int, str]
    system_name: str = ""
    total_charge: float = 0.0
    
    def get_quantum_subsystem_atoms(self, selection_criteria: str) -> List[Atom]:
        """Extract atoms for quantum subsystem based on selection criteria."""
        # Placeholder implementation - would be expanded based on selection logic
        return self.atoms[:10]  # Return first 10 atoms as example


@dataclass
class ValidationResult:
    """Results of parameter or simulation validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)


@dataclass
class CoherenceMetrics:
    """Quantum coherence measures for analysis."""
    coherence_lifetime: float
    quantum_discord: float
    entanglement_measure: float
    purity: float
    von_neumann_entropy: float
    
    def __post_init__(self):
        """Validate coherence metrics."""
        if self.coherence_lifetime < 0:
            raise ValueError("Coherence lifetime cannot be negative")
        if not 0 <= self.purity <= 1:
            raise ValueError("Purity must be between 0 and 1")


@dataclass
class StatisticalSummary:
    """Statistical analysis of simulation results."""
    mean_values: Dict[str, float]
    std_deviations: Dict[str, float]
    confidence_intervals: Dict[str, tuple]
    sample_size: int
    
    def __post_init__(self):
        """Validate statistical summary."""
        if self.sample_size <= 0:
            raise ValueError("Sample size must be positive")


@dataclass
class SimulationResults:
    """Complete results from a quantum biological simulation."""
    state_trajectory: List[DensityMatrix]
    coherence_measures: Dict[str, List[float]]
    energy_trajectory: List[float]
    decoherence_rates: Dict[str, float]
    statistical_summary: StatisticalSummary
    simulation_config: SimulationConfig
    computation_time: float = 0.0
    
    def __post_init__(self):
        """Validate simulation results."""
        if len(self.state_trajectory) == 0:
            raise ValueError("State trajectory cannot be empty")
        if len(self.energy_trajectory) != len(self.state_trajectory):
            raise ValueError("Energy trajectory length must match state trajectory")
    
    def format_summary_table(self, save_to_file: Optional[str] = None) -> str:
        """
        Format results as terminal-friendly table showing key results.
        
        Args:
            save_to_file: Optional path to save the summary table as text file
            
        Returns:
            str: Formatted summary table as string
        """
        import math
        
        # Calculate key metrics
        final_state = self.state_trajectory[-1] if self.state_trajectory else None
        final_energy = self.energy_trajectory[-1] if self.energy_trajectory else None
        
        # Calculate purity from final state
        purity = None
        if final_state:
            eigenvals = np.abs(np.diag(final_state.matrix))
            purity = np.sum(eigenvals**2)
        
        # Estimate coherence lifetime
        coherence_lifetime = None
        if 'purity' in self.coherence_measures and len(self.coherence_measures['purity']) > 1:
            purity_data = self.coherence_measures['purity']
            coherence_lifetime = self._estimate_coherence_lifetime(purity_data)
        
        # Get decoherence rate
        total_decoherence_rate = None
        if self.decoherence_rates:
            total_decoherence_rate = self.decoherence_rates.get('total', 
                                   sum(self.decoherence_rates.values()))
        
        # Energy conservation check
        energy_conservation_error = None
        if len(self.energy_trajectory) > 1:
            initial_energy = self.energy_trajectory[0]
            final_energy_val = self.energy_trajectory[-1]
            energy_conservation_error = abs(final_energy_val - initial_energy) / abs(initial_energy)
        
        # Build summary table
        table_lines = []
        table_lines.append("=" * 70)
        table_lines.append("QUANTUM BIOLOGICAL SIMULATION RESULTS SUMMARY")
        table_lines.append("=" * 70)
        table_lines.append("")
        
        # System information
        table_lines.append("SYSTEM CONFIGURATION")
        table_lines.append("-" * 20)
        table_lines.append(f"  PDB File                : {self.simulation_config.system_pdb}")
        table_lines.append(f"  Temperature             : {self.simulation_config.temperature:.1f} K")
        table_lines.append(f"  Force Field             : {self.simulation_config.force_field}")
        table_lines.append(f"  Quantum Selection       : {self.simulation_config.quantum_subsystem_selection}")
        table_lines.append(f"  Noise Model             : {self.simulation_config.noise_model_type}")
        table_lines.append("")
        
        # Simulation parameters
        table_lines.append("SIMULATION PARAMETERS")
        table_lines.append("-" * 21)
        table_lines.append(f"  Total Time              : {self.simulation_config.simulation_time:.2e} s")
        table_lines.append(f"  Time Step               : {self.simulation_config.time_step:.2e} s")
        table_lines.append(f"  Total Steps             : {len(self.state_trajectory):,}")
        table_lines.append(f"  Computation Time        : {self.computation_time:.2f} s")
        table_lines.append("")
        
        # Key results
        table_lines.append("KEY RESULTS")
        table_lines.append("-" * 11)
        
        if final_energy is not None:
            table_lines.append(f"  Final Energy            : {final_energy:.6f} a.u.")
        else:
            table_lines.append(f"  Final Energy            : N/A")
        
        if purity is not None:
            table_lines.append(f"  Final Purity            : {purity:.4f}")
        else:
            table_lines.append(f"  Final Purity            : N/A")
        
        if coherence_lifetime is not None:
            table_lines.append(f"  Coherence Lifetime      : {coherence_lifetime:.2e} s")
        else:
            table_lines.append(f"  Coherence Lifetime      : N/A")
        
        if total_decoherence_rate is not None:
            table_lines.append(f"  Decoherence Rate        : {total_decoherence_rate:.2e} s⁻¹")
        else:
            table_lines.append(f"  Decoherence Rate        : N/A")
        
        if energy_conservation_error is not None:
            table_lines.append(f"  Energy Conservation Err : {energy_conservation_error:.2e}")
        else:
            table_lines.append(f"  Energy Conservation Err : N/A")
        
        table_lines.append("")
        
        # Statistical summary if available
        if self.statistical_summary and self.statistical_summary.mean_values:
            table_lines.append("STATISTICAL SUMMARY")
            table_lines.append("-" * 19)
            
            for metric, mean_val in self.statistical_summary.mean_values.items():
                std_val = self.statistical_summary.std_deviations.get(metric, 0.0)
                table_lines.append(f"  {metric:<20} : {mean_val:.4e} ± {std_val:.4e}")
            
            table_lines.append(f"  Sample Size             : {self.statistical_summary.sample_size}")
            table_lines.append("")
        
        # Output information
        table_lines.append("OUTPUT")
        table_lines.append("-" * 6)
        table_lines.append(f"  Results Directory       : {self.simulation_config.output_directory}")
        table_lines.append("")
        table_lines.append("=" * 70)
        
        # Join all lines
        summary_text = "\n".join(table_lines)
        
        # Save to file if requested
        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write(summary_text)
            except Exception as e:
                print(f"Warning: Could not save summary to {save_to_file}: {e}")
        
        return summary_text
    
    def _estimate_coherence_lifetime(self, purity_data: List[float]) -> Optional[float]:
        """Estimate coherence lifetime from purity decay."""
        try:
            if len(purity_data) < 2:
                return None
            
            # Convert to numpy array
            purity_array = np.array(purity_data)
            
            # Find 1/e decay point
            initial_purity = purity_array[0]
            target_purity = initial_purity / np.e
            
            # Find first point below target
            below_target = np.where(purity_array <= target_purity)[0]
            
            if len(below_target) > 0:
                # Estimate based on time step and position
                time_step = self.simulation_config.time_step
                lifetime_steps = below_target[0]
                return lifetime_steps * time_step
            
            return None
            
        except Exception:
            return None


# Additional helper classes for specific components

@dataclass
class LindbladOperator:
    """Represents a Lindblad operator for open quantum system evolution."""
    operator: np.ndarray
    coupling_strength: float
    operator_type: str = "dephasing"
    
    def __post_init__(self):
        """Validate Lindblad operator."""
        if self.coupling_strength < 0:
            raise ValueError("Coupling strength cannot be negative")


@dataclass
class Hamiltonian:
    """Represents the system Hamiltonian."""
    matrix: np.ndarray
    basis_labels: List[str]
    time_dependent: bool = False
    
    def __post_init__(self):
        """Validate Hamiltonian."""
        n = len(self.basis_labels)
        if self.matrix.shape != (n, n):
            raise ValueError("Hamiltonian matrix size doesn't match basis")
        
        # Check Hermiticity
        if not np.allclose(self.matrix, self.matrix.conj().T):
            raise ValueError("Hamiltonian is not Hermitian")


@dataclass
class SpectralDensity:
    """Represents environmental spectral density function."""
    frequencies: np.ndarray
    spectral_values: np.ndarray
    temperature: float
    spectral_type: str = "ohmic"
    
    def __post_init__(self):
        """Validate spectral density."""
        if len(self.frequencies) != len(self.spectral_values):
            raise ValueError("Frequency and spectral value arrays must have same length")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")