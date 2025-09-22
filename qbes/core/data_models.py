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
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.simulation_time <= 0:
            raise ValueError("Simulation time must be positive")
        if self.time_step <= 0:
            raise ValueError("Time step must be positive")


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