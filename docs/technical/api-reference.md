# QBES API Reference

## Core Classes

### ConfigurationManager
Handles loading and validation of simulation configurations.

```python
from qbes import ConfigurationManager

config_manager = ConfigurationManager()
config = config_manager.load_config("config.yaml")
validation = config_manager.validate_parameters(config)
```

#### Methods
- `load_config(filepath)`: Load configuration from YAML/JSON file
- `validate_parameters(config)`: Validate configuration parameters
- `create_config_from_dict(config_dict)`: Create config from dictionary

### SimulationEngine
Main simulation orchestrator for quantum biological systems.

```python
from qbes import SimulationEngine

engine = SimulationEngine()
engine.initialize_simulation(config)
results = engine.run_simulation()
```

#### Methods
- `initialize_simulation(config)`: Initialize simulation with configuration
- `run_simulation()`: Execute complete simulation
- `save_checkpoint(filepath)`: Save simulation checkpoint
- `load_checkpoint(filepath)`: Resume from checkpoint

### QuantumEngine
Quantum mechanics calculations and state evolution.

```python
from qbes.quantum_engine import QuantumEngine

quantum_engine = QuantumEngine()
state = quantum_engine.initialize_state(config)
evolved_state = quantum_engine.evolve_state(state, time_step, hamiltonian)
```

#### Methods
- `initialize_state(config)`: Create initial quantum state
- `evolve_state(state, dt, hamiltonian, lindblad_ops)`: Time evolution
- `calculate_observables(state)`: Compute quantum observables
- `calculate_purity(state)`: Calculate state purity

### ResultsAnalyzer
Analysis and visualization of simulation results.

```python
from qbes.analysis import ResultsAnalyzer

analyzer = ResultsAnalyzer()
results = analyzer.load_results("./output")
lifetime = analyzer.calculate_coherence_lifetime(results)
```

#### Methods
- `load_results(directory)`: Load simulation results
- `calculate_coherence_lifetime(results)`: Compute coherence decay time
- `calculate_transfer_efficiency(results)`: Energy transfer efficiency
- `generate_plots(results, output_dir)`: Create visualization plots

## Data Models

### SimulationConfig
Configuration data structure for simulations.

```python
from qbes.core.data_models import SimulationConfig

config = SimulationConfig(
    system_type="photosystem",
    temperature=300.0,
    simulation_time=1e-12,
    time_step=1e-15
)
```

#### Attributes
- `system_type`: Type of biological system
- `temperature`: System temperature (K)
- `simulation_time`: Total simulation time (s)
- `time_step`: Integration time step (s)
- `quantum_subsystem`: Quantum region definition
- `noise_model`: Environmental noise parameters

### SimulationResults
Container for simulation output data.

```python
# Access results
coherence_lifetime = results.coherence_lifetime
final_populations = results.final_populations
time_points = results.time_points
```

#### Attributes
- `coherence_lifetime`: Quantum coherence decay time
- `decoherence_rate`: Rate of coherence loss
- `final_populations`: Final state populations
- `energy_conservation_error`: Energy conservation check
- `time_points`: Simulation time array
- `coherence_trajectory`: Coherence evolution data

### DensityMatrix
Quantum density matrix representation.

```python
from qbes.core.data_models import DensityMatrix

density_matrix = DensityMatrix(
    matrix=numpy_array,
    basis_labels=["ground", "excited"],
    time=current_time
)
```

#### Methods
- `trace()`: Calculate matrix trace
- `purity()`: Calculate state purity
- `entropy()`: Calculate von Neumann entropy

## Noise Models

### NoiseModelFactory
Factory for creating biological noise models.

```python
from qbes.noise_models import NoiseModelFactory

factory = NoiseModelFactory()
noise_model = factory.create_noise_model(
    "protein_ohmic",
    temperature=300.0,
    coupling_strength=1.0
)
```

#### Available Models
- `protein_ohmic`: Protein environment with Ohmic coupling
- `membrane`: Membrane lipid environment
- `solvent_ionic`: Ionic aqueous solution
- `custom`: User-defined parameters

## Validation Framework

### BenchmarkRunner
Execute validation benchmarks and accuracy tests.

```python
from qbes.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_validation_suite("standard")
```

#### Methods
- `run_validation_suite(suite_type)`: Run validation tests
- `generate_report()`: Create validation report
- `get_failed_tests()`: List failed benchmarks

## Utilities

### FileIOUtils
File input/output utilities for QBES data formats.

```python
from qbes.utils.file_io import FileIOUtils

# Save results
FileIOUtils.save_results(results, "output.json")

# Load checkpoint
checkpoint = FileIOUtils.load_checkpoint("checkpoint.pkl")
```

### PlotGenerator
Visualization and plotting utilities.

```python
from qbes.visualization import PlotGenerator

plotter = PlotGenerator()
plotter.plot_coherence_evolution(results, "coherence.png")
plotter.plot_population_dynamics(results, "populations.png")
```

## Error Handling

### QBESError
Base exception class for QBES-specific errors.

```python
from qbes.utils.error_handling import QBESError

try:
    results = engine.run_simulation()
except QBESError as e:
    print(f"QBES error: {e}")
```

### Common Exceptions
- `ConfigurationError`: Invalid configuration parameters
- `SimulationError`: Simulation execution failure
- `ValidationError`: Validation test failure
- `FileIOError`: File input/output problems

## Example Usage

### Complete Simulation Workflow
```python
from qbes import ConfigurationManager, SimulationEngine
from qbes.analysis import ResultsAnalyzer

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("photosystem.yaml")

# Validate configuration
validation = config_manager.validate_parameters(config)
if not validation.is_valid:
    print("Configuration errors:", validation.errors)
    exit(1)

# Run simulation
engine = SimulationEngine()
engine.initialize_simulation(config)
results = engine.run_simulation()

# Analyze results
analyzer = ResultsAnalyzer()
coherence_lifetime = analyzer.calculate_coherence_lifetime(results)
transfer_efficiency = analyzer.calculate_transfer_efficiency(results)

print(f"Coherence lifetime: {coherence_lifetime:.2e} s")
print(f"Transfer efficiency: {transfer_efficiency:.1%}")

# Generate plots
analyzer.generate_plots(results, "./plots")
```

This API reference provides complete documentation for programmatic access to QBES functionality.