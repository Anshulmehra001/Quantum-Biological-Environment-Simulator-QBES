# API Reference

Complete reference for the QBES Python API.

## Table of Contents

- [Core Modules](#core-modules)
- [Configuration Management](#configuration-management)
- [Simulation Engine](#simulation-engine)
- [Quantum Mechanics](#quantum-mechanics)
- [Molecular Dynamics](#molecular-dynamics)
- [Noise Models](#noise-models)
- [Analysis Tools](#analysis-tools)
- [Visualization](#visualization)
- [Utilities](#utilities)

## Core Modules

### qbes.core.data_models

Core data structures used throughout QBES.

#### SimulationConfig

```python
@dataclass
class SimulationConfig:
    """Main configuration for QBES simulations."""
    
    # System definition
    pdb_file: str
    force_field: str
    temperature: float
    
    # Simulation parameters
    simulation_time: float
    time_step: float
    integration_method: str = "runge_kutta"
    
    # Quantum subsystem
    quantum_selection: str
    max_quantum_atoms: int = 200
    
    # Noise model
    noise_model_type: str
    coupling_strength: float
    
    # Output settings
    output_directory: str
    save_trajectory: bool = True
    
    def validate(self) -> ValidationResult:
        """Validate configuration parameters."""
        pass
```

#### QuantumSubsystem

```python
@dataclass
class QuantumSubsystem:
    """Represents the quantum mechanical subsystem."""
    
    atoms: List[Atom]
    hamiltonian_parameters: Dict[str, float]
    coupling_matrix: np.ndarray
    basis_states: List[QuantumState]
    
    def get_hamiltonian(self) -> np.ndarray:
        """Get system Hamiltonian matrix."""
        pass
    
    def get_dimension(self) -> int:
        """Get Hilbert space dimension."""
        return len(self.basis_states)
```

#### SimulationResults

```python
@dataclass
class SimulationResults:
    """Container for simulation results."""
    
    # Time evolution data
    time_points: np.ndarray
    state_trajectory: List[DensityMatrix]
    energy_trajectory: np.ndarray
    
    # Analysis results
    coherence_measures: Dict[str, np.ndarray]
    decoherence_rates: Dict[str, float]
    transfer_efficiency: float
    
    # Statistical data
    statistical_summary: StatisticalSummary
    uncertainty_estimates: Dict[str, float]
    
    def save(self, output_directory: str) -> None:
        """Save results to files."""
        pass
    
    @classmethod
    def load(cls, output_directory: str) -> 'SimulationResults':
        """Load results from files."""
        pass
```

### qbes.core.interfaces

Abstract base classes defining QBES interfaces.

#### SimulationModule

```python
from abc import ABC, abstractmethod

class SimulationModule(ABC):
    """Base class for all simulation modules."""
    
    @abstractmethod
    def initialize(self, config: SimulationConfig) -> InitializationResult:
        """Initialize the module with configuration."""
        pass
    
    @abstractmethod
    def step(self, current_state: SystemState, time_step: float) -> SystemState:
        """Perform one simulation step."""
        pass
    
    @abstractmethod
    def finalize(self, results: SimulationResults) -> SimulationResults:
        """Finalize and process results."""
        pass
```

#### NoiseModel

```python
class NoiseModel(ABC):
    """Base class for environmental noise models."""
    
    @abstractmethod
    def generate_lindblad_operators(self, system: QuantumSubsystem) -> List[LindbladOperator]:
        """Generate Lindblad operators for this noise model."""
        pass
    
    @abstractmethod
    def calculate_decoherence_rates(self, temperature: float) -> Dict[str, float]:
        """Calculate temperature-dependent decoherence rates."""
        pass
    
    @abstractmethod
    def get_spectral_density(self, frequency: float) -> float:
        """Get spectral density at given frequency."""
        pass
```

## Configuration Management

### qbes.config_manager.ConfigurationManager

Main class for handling configuration files and parameter validation.

```python
class ConfigurationManager:
    """Manages QBES configuration files and validation."""
    
    def __init__(self):
        self.validators = {}
        self._register_default_validators()
    
    def load_config(self, config_path: str) -> SimulationConfig:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            SimulationConfig object
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        pass
    
    def validate_parameters(self, config: SimulationConfig) -> ValidationResult:
        """Validate all configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        pass
    
    def generate_default_config(self, output_path: str) -> bool:
        """Generate default configuration template.
        
        Args:
            output_path: Where to save the template
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def parse_pdb(self, pdb_path: str) -> MolecularSystem:
        """Parse PDB file and extract molecular system.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            MolecularSystem object
            
        Raises:
            PDBParsingError: If PDB file is invalid
        """
        pass
    
    def identify_quantum_subsystem(self, 
                                 system: MolecularSystem, 
                                 selection: str) -> QuantumSubsystem:
        """Identify quantum subsystem from molecular system.
        
        Args:
            system: Full molecular system
            selection: Selection criteria (e.g., "resname CHL")
            
        Returns:
            QuantumSubsystem object
        """
        pass
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    """Result of parameter validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, message: str) -> None:
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
```

## Simulation Engine

### qbes.simulation_engine.SimulationEngine

Main orchestrator for QBES simulations.

```python
class SimulationEngine:
    """Main simulation engine coordinating all components."""
    
    def __init__(self):
        self.config = None
        self.quantum_engine = None
        self.md_engine = None
        self.noise_model = None
        self.is_running = False
        self.progress = 0.0
    
    def initialize_simulation(self, config: SimulationConfig) -> InitializationResult:
        """Initialize simulation with given configuration.
        
        Args:
            config: Simulation configuration
            
        Returns:
            InitializationResult indicating success/failure
        """
        pass
    
    def run_simulation(self) -> SimulationResults:
        """Run the complete simulation.
        
        Returns:
            SimulationResults object with all results
            
        Raises:
            SimulationError: If simulation fails
        """
        pass
    
    def get_progress(self) -> float:
        """Get current simulation progress (0-100)."""
        return self.progress
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save simulation checkpoint.
        
        Args:
            checkpoint_path: Where to save checkpoint
            
        Returns:
            True if successful
        """
        pass
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load simulation from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successful
        """
        pass
    
    def stop_simulation(self) -> None:
        """Stop running simulation gracefully."""
        self.is_running = False
```

## Quantum Mechanics

### qbes.quantum_engine.QuantumEngine

Handles quantum mechanical calculations and state evolution.

```python
class QuantumEngine:
    """Quantum mechanics engine for state evolution."""
    
    def __init__(self):
        self.hamiltonian = None
        self.lindblad_operators = []
        self.current_state = None
    
    def initialize_hamiltonian(self, system: QuantumSubsystem) -> np.ndarray:
        """Initialize system Hamiltonian.
        
        Args:
            system: Quantum subsystem definition
            
        Returns:
            Hamiltonian matrix
        """
        pass
    
    def evolve_state(self, 
                    initial_state: DensityMatrix, 
                    time_step: float,
                    hamiltonian: Optional[np.ndarray] = None) -> DensityMatrix:
        """Evolve quantum state by one time step.
        
        Args:
            initial_state: Initial density matrix
            time_step: Time step size
            hamiltonian: Optional time-dependent Hamiltonian
            
        Returns:
            Evolved density matrix
        """
        pass
    
    def apply_lindblad_operators(self, 
                               state: DensityMatrix, 
                               operators: List[LindbladOperator],
                               time_step: float) -> DensityMatrix:
        """Apply Lindblad operators for decoherence.
        
        Args:
            state: Current density matrix
            operators: List of Lindblad operators
            time_step: Time step size
            
        Returns:
            State after decoherence
        """
        pass
    
    def calculate_coherence_measures(self, state: DensityMatrix) -> Dict[str, float]:
        """Calculate various coherence measures.
        
        Args:
            state: Density matrix
            
        Returns:
            Dictionary of coherence measures
        """
        pass
    
    def calculate_populations(self, state: DensityMatrix) -> np.ndarray:
        """Calculate state populations.
        
        Args:
            state: Density matrix
            
        Returns:
            Array of populations
        """
        return np.real(np.diag(state))
    
    def calculate_energy(self, state: DensityMatrix) -> float:
        """Calculate expectation value of energy.
        
        Args:
            state: Density matrix
            
        Returns:
            Energy expectation value
        """
        return np.real(np.trace(state @ self.hamiltonian))
```

### Quantum State Operations

```python
class QuantumStateOperations:
    """Utility functions for quantum state manipulation."""
    
    @staticmethod
    def create_thermal_state(hamiltonian: np.ndarray, temperature: float) -> DensityMatrix:
        """Create thermal equilibrium state.
        
        Args:
            hamiltonian: System Hamiltonian
            temperature: Temperature in Kelvin
            
        Returns:
            Thermal density matrix
        """
        pass
    
    @staticmethod
    def partial_trace(state: DensityMatrix, subsystem_dims: List[int], 
                     trace_over: List[int]) -> DensityMatrix:
        """Calculate partial trace over specified subsystems.
        
        Args:
            state: Full system density matrix
            subsystem_dims: Dimensions of each subsystem
            trace_over: Indices of subsystems to trace over
            
        Returns:
            Reduced density matrix
        """
        pass
    
    @staticmethod
    def fidelity(state1: DensityMatrix, state2: DensityMatrix) -> float:
        """Calculate fidelity between two quantum states.
        
        Args:
            state1, state2: Density matrices
            
        Returns:
            Fidelity (0 to 1)
        """
        pass
```

## Molecular Dynamics

### qbes.md_engine.MDEngine

Handles classical molecular dynamics simulations.

```python
class MDEngine:
    """Molecular dynamics engine using OpenMM."""
    
    def __init__(self):
        self.system = None
        self.simulation = None
        self.topology = None
    
    def initialize_system(self, 
                         pdb_file: str, 
                         force_field: str,
                         solvent_model: Optional[str] = None) -> MDSystem:
        """Initialize MD system from PDB file.
        
        Args:
            pdb_file: Path to PDB structure file
            force_field: Force field name (e.g., 'amber14')
            solvent_model: Solvent model (e.g., 'tip3p')
            
        Returns:
            MDSystem object
        """
        pass
    
    def run_trajectory(self, 
                      duration: float, 
                      time_step: float,
                      temperature: float) -> Trajectory:
        """Run MD trajectory.
        
        Args:
            duration: Simulation duration in seconds
            time_step: MD time step in seconds
            temperature: Temperature in Kelvin
            
        Returns:
            Trajectory object with coordinates and energies
        """
        pass
    
    def extract_quantum_parameters(self, 
                                 trajectory: Trajectory,
                                 quantum_atoms: List[int]) -> ParameterTimeSeries:
        """Extract quantum parameters from MD trajectory.
        
        Args:
            trajectory: MD trajectory
            quantum_atoms: Indices of quantum atoms
            
        Returns:
            Time series of quantum parameters
        """
        pass
    
    def calculate_spectral_density(self, 
                                 fluctuations: TimeSeries) -> SpectralDensity:
        """Calculate spectral density from parameter fluctuations.
        
        Args:
            fluctuations: Time series of parameter fluctuations
            
        Returns:
            SpectralDensity object
        """
        pass
```

## Noise Models

### qbes.noise_models.NoiseModelFactory

Factory for creating different types of noise models.

```python
class NoiseModelFactory:
    """Factory for creating biological noise models."""
    
    def __init__(self):
        self._models = {}
        self._register_default_models()
    
    def create_noise_model(self, 
                          model_type: str, 
                          **parameters) -> NoiseModel:
        """Create noise model of specified type.
        
        Args:
            model_type: Type of noise model
            **parameters: Model-specific parameters
            
        Returns:
            NoiseModel instance
        """
        pass
    
    def register_model(self, name: str, model_class: Type[NoiseModel]) -> None:
        """Register custom noise model.
        
        Args:
            name: Model name
            model_class: NoiseModel subclass
        """
        self._models[name] = model_class
    
    def list_available_models(self) -> List[str]:
        """List all available noise model types."""
        return list(self._models.keys())
```

### Built-in Noise Models

#### ProteinOhmicNoise

```python
class ProteinOhmicNoise(NoiseModel):
    """Ohmic noise model for protein environments."""
    
    def __init__(self, 
                 coupling_strength: float,
                 cutoff_frequency: float,
                 reorganization_energy: float):
        """Initialize protein noise model.
        
        Args:
            coupling_strength: System-bath coupling strength
            cutoff_frequency: Spectral density cutoff frequency
            reorganization_energy: Reorganization energy in cm^-1
        """
        pass
    
    def get_spectral_density(self, frequency: float) -> float:
        """Ohmic spectral density with exponential cutoff."""
        pass
```

#### MembraneNoise

```python
class MembraneNoise(NoiseModel):
    """Noise model for membrane environments."""
    
    def __init__(self, 
                 lipid_composition: Dict[str, float],
                 membrane_thickness: float):
        """Initialize membrane noise model.
        
        Args:
            lipid_composition: Lipid types and fractions
            membrane_thickness: Membrane thickness in nm
        """
        pass
```

## Analysis Tools

### qbes.analysis.ResultsAnalyzer

Main class for analyzing simulation results.

```python
class ResultsAnalyzer:
    """Comprehensive analysis of QBES simulation results."""
    
    def __init__(self):
        self.results = None
    
    def load_results(self, output_directory: str) -> SimulationResults:
        """Load results from output directory.
        
        Args:
            output_directory: Path to simulation output
            
        Returns:
            SimulationResults object
        """
        pass
    
    def calculate_coherence_lifetime(self, 
                                   state_trajectory: List[DensityMatrix]) -> float:
        """Calculate coherence lifetime from state evolution.
        
        Args:
            state_trajectory: List of density matrices over time
            
        Returns:
            Coherence lifetime in seconds
        """
        pass
    
    def calculate_transfer_efficiency(self, results: SimulationResults) -> float:
        """Calculate energy transfer efficiency.
        
        Args:
            results: Simulation results
            
        Returns:
            Transfer efficiency (0 to 1)
        """
        pass
    
    def calculate_quantum_discord(self, 
                                bipartite_state: DensityMatrix) -> float:
        """Calculate quantum discord for bipartite system.
        
        Args:
            bipartite_state: Density matrix of bipartite system
            
        Returns:
            Quantum discord value
        """
        pass
    
    def perform_statistical_analysis(self, 
                                   results: SimulationResults) -> StatisticalSummary:
        """Perform comprehensive statistical analysis.
        
        Args:
            results: Simulation results
            
        Returns:
            Statistical summary with uncertainties
        """
        pass
    
    def validate_energy_conservation(self, 
                                   results: SimulationResults) -> ValidationResult:
        """Validate energy conservation during simulation.
        
        Args:
            results: Simulation results
            
        Returns:
            Validation result
        """
        pass
```

### Specialized Analyzers

#### CoherenceAnalyzer

```python
class CoherenceAnalyzer:
    """Specialized analysis of quantum coherence."""
    
    def calculate_l1_coherence(self, state: DensityMatrix) -> float:
        """Calculate l1-norm coherence measure."""
        pass
    
    def calculate_relative_entropy_coherence(self, state: DensityMatrix) -> float:
        """Calculate relative entropy of coherence."""
        pass
    
    def analyze_coherence_dynamics(self, 
                                 state_trajectory: List[DensityMatrix]) -> Dict:
        """Analyze coherence dynamics over time."""
        pass
```

## Visualization

### qbes.visualization.PlotGenerator

Generate publication-ready plots and visualizations.

```python
class PlotGenerator:
    """Generate plots and visualizations for QBES results."""
    
    def __init__(self, style: str = 'scientific'):
        """Initialize plot generator.
        
        Args:
            style: Plot style ('scientific', 'presentation', 'publication')
        """
        pass
    
    def plot_coherence_evolution(self, 
                               results: SimulationResults,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot coherence measures over time.
        
        Args:
            results: Simulation results
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        pass
    
    def plot_energy_landscape(self, 
                            results: SimulationResults,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot energy landscape and dynamics.
        
        Args:
            results: Simulation results
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        pass
    
    def plot_population_dynamics(self, 
                               results: SimulationResults,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot state population evolution.
        
        Args:
            results: Simulation results
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        pass
    
    def create_animation(self, 
                        results: SimulationResults,
                        animation_type: str = 'coherence',
                        save_path: Optional[str] = None) -> FuncAnimation:
        """Create animation of quantum dynamics.
        
        Args:
            results: Simulation results
            animation_type: Type of animation ('coherence', 'populations', 'energy')
            save_path: Optional path to save animation
            
        Returns:
            Matplotlib animation
        """
        pass
    
    def generate_report_plots(self, 
                            results: SimulationResults,
                            output_directory: str) -> List[str]:
        """Generate complete set of analysis plots.
        
        Args:
            results: Simulation results
            output_directory: Where to save plots
            
        Returns:
            List of generated plot files
        """
        pass
```

## Utilities

### qbes.utils.file_io.FileIOUtils

File input/output utilities.

```python
class FileIOUtils:
    """Utilities for file input/output operations."""
    
    @staticmethod
    def save_results_json(results: SimulationResults, filepath: str) -> None:
        """Save results to JSON format."""
        pass
    
    @staticmethod
    def save_results_hdf5(results: SimulationResults, filepath: str) -> None:
        """Save results to HDF5 format."""
        pass
    
    @staticmethod
    def load_results_json(filepath: str) -> SimulationResults:
        """Load results from JSON format."""
        pass
    
    @staticmethod
    def export_to_csv(results: SimulationResults, output_dir: str) -> None:
        """Export time-series data to CSV files."""
        pass
```

### qbes.utils.validation.ValidationUtils

Parameter validation utilities.

```python
class ValidationUtils:
    """Utilities for parameter validation."""
    
    @staticmethod
    def validate_temperature(temperature: float) -> ValidationResult:
        """Validate temperature parameter."""
        pass
    
    @staticmethod
    def validate_time_step(time_step: float, system_size: int) -> ValidationResult:
        """Validate time step for given system size."""
        pass
    
    @staticmethod
    def validate_pdb_file(pdb_path: str) -> ValidationResult:
        """Validate PDB file format and content."""
        pass
```

### qbes.utils.error_handling.ErrorHandler

Error handling and recovery utilities.

```python
class ErrorHandler:
    """Handle errors and provide recovery options."""
    
    def handle_convergence_failure(self, 
                                 simulation: SimulationEngine) -> RecoveryAction:
        """Handle simulation convergence failures."""
        pass
    
    def handle_memory_error(self, 
                          config: SimulationConfig) -> SimulationConfig:
        """Suggest configuration changes for memory issues."""
        pass
    
    def generate_diagnostic_report(self, 
                                 error: Exception,
                                 context: Dict) -> str:
        """Generate diagnostic report for errors."""
        pass
```

## Type Definitions

Common type aliases used throughout QBES:

```python
from typing import Union, List, Dict, Optional, Tuple
import numpy as np

# Basic types
DensityMatrix = np.ndarray  # Complex 2D array
Hamiltonian = np.ndarray    # Complex 2D array
QuantumState = np.ndarray   # Complex 1D array
TimeSeries = np.ndarray     # Real 1D array

# Complex types
StateTrajectory = List[DensityMatrix]
ParameterDict = Dict[str, Union[float, int, str]]
CoherenceMeasures = Dict[str, float]
```

## Error Classes

Custom exceptions used by QBES:

```python
class QBESError(Exception):
    """Base exception for QBES errors."""
    pass

class ConfigurationError(QBESError):
    """Configuration-related errors."""
    pass

class SimulationError(QBESError):
    """Simulation execution errors."""
    pass

class ConvergenceError(SimulationError):
    """Simulation convergence failures."""
    pass

class ValidationError(QBESError):
    """Parameter validation errors."""
    pass
```

## Constants

Physical constants and default values:

```python
# Physical constants (in SI units)
PLANCK_CONSTANT = 6.62607015e-34  # J⋅s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C

# Unit conversions
EV_TO_JOULE = 1.602176634e-19
CM_INV_TO_JOULE = 1.986445857e-23
FEMTOSECOND = 1e-15  # seconds

# Default parameters
DEFAULT_TIME_STEP = 1e-15  # 1 femtosecond
DEFAULT_TEMPERATURE = 300.0  # Kelvin
DEFAULT_COUPLING_STRENGTH = 1.0
```

This API reference provides comprehensive documentation for all public interfaces in QBES. For implementation details and examples, see the [User Guide](user_guide.md) and [Tutorial](tutorial.md).