# User Guide

This comprehensive guide covers all aspects of using the Quantum Biological Environment Simulator (QBES) for your research.

## Table of Contents

- [Getting Started](#getting-started)
- [Configuration System](#configuration-system)
- [Running Simulations](#running-simulations)
- [Analyzing Results](#analyzing-results)
- [Command-Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Best Practices](#best-practices)
- [Advanced Usage](#advanced-usage)

## Getting Started

### Your First Simulation

Let's walk through running your first QBES simulation:

1. **Generate a configuration file**:
```bash
qbes generate-config my_first_sim.yaml --template photosystem
```

2. **Edit the configuration** (optional):
```bash
# Edit with your preferred text editor
nano my_first_sim.yaml
```

3. **Validate the configuration**:
```bash
qbes validate my_first_sim.yaml
```

4. **Run the simulation**:
```bash
qbes run my_first_sim.yaml --verbose
```

5. **Check results**:
```bash
qbes status ./photosystem_output
```

### Understanding QBES Workflow

QBES follows a structured workflow:

1. **System Setup**: Load molecular structure (PDB file)
2. **Quantum Subsystem Selection**: Identify quantum-relevant atoms
3. **Environment Modeling**: Set up classical molecular dynamics
4. **Noise Model Application**: Apply biological decoherence models
5. **Simulation Execution**: Run hybrid QM/MM simulation
6. **Analysis**: Calculate quantum coherence measures
7. **Visualization**: Generate plots and reports

## Configuration System

### Configuration File Structure

QBES uses YAML configuration files with the following main sections:

```yaml
# System definition
system:
  pdb_file: "path/to/structure.pdb"
  force_field: "amber14"
  solvent_model: "tip3p"
  ionic_strength: 0.15

# Simulation parameters
simulation:
  temperature: 300.0          # Kelvin
  simulation_time: 1.0e-12    # seconds (1 ps)
  time_step: 1.0e-15          # seconds (1 fs)

# Quantum subsystem selection
quantum_subsystem:
  selection_method: "chromophores"
  custom_selection: "resname CHL BCL"
  max_quantum_atoms: 200

# Noise model configuration
noise_model:
  type: "protein_ohmic"
  coupling_strength: 2.0
  cutoff_frequency: 5.0e13
  reorganization_energy: 35.0  # cm^-1

# Output settings
output:
  directory: "./output"
  save_trajectory: true
  save_checkpoints: true
  checkpoint_interval: 500
  plot_format: "png"
```

### Configuration Templates

QBES provides several pre-configured templates:

#### Photosynthetic Systems
```bash
qbes generate-config photosystem.yaml --template photosystem
```
- Optimized for light-harvesting complexes
- Includes chlorophyll and bacteriochlorophyll selection
- Uses protein environment noise model

#### Enzyme Active Sites
```bash
qbes generate-config enzyme.yaml --template enzyme
```
- Focused on catalytic sites
- Includes common catalytic residues
- Higher temperature for physiological conditions

#### Membrane Proteins
```bash
qbes generate-config membrane.yaml --template membrane
```
- Designed for membrane-bound systems
- Uses lipid environment noise model
- Appropriate for ion channels and transporters

### Parameter Guidelines

#### System Parameters

- **Temperature**: 
  - Photosynthetic systems: 77-300 K
  - Enzymes: 310 K (body temperature)
  - In vitro studies: 298 K (room temperature)

- **Simulation Time**:
  - Coherence studies: 1-100 ps
  - Reaction dynamics: 100 fs - 10 ps
  - Conformational changes: 1-100 ns

- **Time Step**:
  - Quantum dynamics: 0.1-1 fs
  - Classical MD: 1-2 fs
  - Adaptive stepping available

#### Quantum Subsystem Selection

Choose quantum atoms based on your research question:

- **Chromophores**: `resname CHL BCL PEO`
- **Active Sites**: `resname HIS CYS ASP GLU SER THR`
- **Metal Centers**: `resname HEM ZN MG CA`
- **Custom Selection**: Use MDTraj selection syntax

#### Noise Model Parameters

- **Coupling Strength**: 0.5-5.0 (dimensionless)
- **Cutoff Frequency**: 1e13-1e14 Hz
- **Reorganization Energy**: 10-100 cm⁻¹

## Running Simulations

### Basic Simulation Execution

```bash
# Run with default settings
qbes run config.yaml

# Run with verbose output
qbes run config.yaml --verbose

# Specify output directory
qbes run config.yaml --output-dir ./my_results

# Dry run (validation only)
qbes run config.yaml --dry-run
```

### Monitoring Simulations

#### Real-time Monitoring
```bash
# Monitor during simulation
qbes run config.yaml --monitor

# Monitor existing simulation
qbes monitor-sim ./output --interval 5.0
```

#### Status Checking
```bash
# Check simulation status
qbes status ./output

# Detailed status information
qbes status ./output --verbose
```

### Checkpoint and Resume

QBES automatically saves checkpoints during long simulations:

```bash
# Resume from checkpoint
qbes resume ./output/checkpoint_1000.pkl

# Resume with monitoring
qbes resume ./output/checkpoint_1000.pkl --monitor
```

### Parallel Execution

For large systems, use parallel execution:

```bash
# Use multiple CPU cores
export OMP_NUM_THREADS=8
qbes run config.yaml

# MPI parallel execution (if available)
mpirun -np 4 qbes run config.yaml
```

## Analyzing Results

### Output Files

QBES generates several output files:

- `simulation_results.json`: Main results summary
- `state_trajectory.h5`: Quantum state evolution
- `coherence_measures.csv`: Time-series coherence data
- `energy_trajectory.csv`: Energy evolution
- `plots/`: Visualization directory
- `checkpoints/`: Checkpoint files

### Result Structure

```python
# Load results in Python
import json
import h5py
import pandas as pd

# Main results
with open('output/simulation_results.json', 'r') as f:
    results = json.load(f)

print(f"Coherence lifetime: {results['coherence_lifetime']:.2e} s")
print(f"Decoherence rate: {results['decoherence_rate']:.2e} Hz")

# Time-series data
coherence_data = pd.read_csv('output/coherence_measures.csv')
energy_data = pd.read_csv('output/energy_trajectory.csv')

# Quantum state trajectory
with h5py.File('output/state_trajectory.h5', 'r') as f:
    times = f['times'][:]
    states = f['density_matrices'][:]
```

### Visualization

QBES automatically generates publication-ready plots:

- **Coherence Evolution**: Time-dependent coherence measures
- **Energy Landscapes**: Potential energy surfaces
- **Population Dynamics**: State population evolution
- **Spectral Analysis**: Frequency domain analysis

### Statistical Analysis

Results include statistical measures:

- **Uncertainty Estimates**: Confidence intervals
- **Significance Tests**: Statistical validation
- **Outlier Detection**: Data quality assessment
- **Convergence Analysis**: Simulation reliability

## Command-Line Interface

### Main Commands

#### Simulation Management
```bash
qbes run <config>              # Run simulation
qbes resume <checkpoint>       # Resume from checkpoint
qbes status <output_dir>       # Check status
qbes monitor-sim <output_dir>  # Real-time monitoring
```

#### Configuration
```bash
qbes generate-config <file>    # Generate template
qbes validate <config>         # Validate configuration
```

#### Benchmarking
```bash
qbes benchmark run-benchmarks  # Run validation suite
qbes benchmark analyze <dir>   # Analyze results
```

### Command Options

Most commands support common options:

- `--verbose, -v`: Detailed output
- `--output-dir, -o`: Specify output directory
- `--help`: Show command help

### Configuration Generation Options

```bash
# Generate specific templates
qbes generate-config config.yaml --template photosystem
qbes generate-config config.yaml --template enzyme
qbes generate-config config.yaml --template membrane

# Generate default configuration
qbes generate-config config.yaml --template default
```

## Python API

### Basic Usage

```python
from qbes import ConfigurationManager, SimulationEngine
from qbes.analysis import ResultsAnalyzer
from qbes.visualization import PlotGenerator

# Load and validate configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("config.yaml")
validation = config_manager.validate_parameters(config)

if validation.is_valid:
    # Initialize simulation
    engine = SimulationEngine()
    engine.initialize_simulation(config)
    
    # Run simulation
    results = engine.run_simulation()
    
    # Analyze results
    analyzer = ResultsAnalyzer()
    coherence_lifetime = analyzer.calculate_coherence_lifetime(results.state_trajectory)
    
    # Generate plots
    plotter = PlotGenerator()
    plotter.plot_coherence_evolution(results)
    plotter.save_plots("./plots")
```

### Advanced API Usage

#### Custom Quantum Systems

```python
from qbes.core.data_models import QuantumSubsystem, Atom
from qbes.quantum_engine import QuantumEngine

# Define custom quantum system
atoms = [
    Atom(element="C", position=[0, 0, 0]),
    Atom(element="N", position=[1.4, 0, 0])
]

quantum_system = QuantumSubsystem(
    atoms=atoms,
    hamiltonian_parameters={"coupling": 0.1},
    basis_states=["ground", "excited"]
)

# Initialize quantum engine
qm_engine = QuantumEngine()
hamiltonian = qm_engine.initialize_hamiltonian(quantum_system)
```

#### Custom Noise Models

```python
from qbes.noise_models import NoiseModelFactory, CustomNoiseModel

# Create custom noise model
class MyNoiseModel(CustomNoiseModel):
    def generate_lindblad_operators(self, system):
        # Custom implementation
        return operators
    
    def calculate_decoherence_rates(self, temperature):
        # Custom rates
        return rates

# Use in simulation
noise_factory = NoiseModelFactory()
noise_factory.register_model("my_model", MyNoiseModel)
```

#### Batch Processing

```python
from qbes.utils.batch_processing import BatchRunner
import os

# Set up batch simulations
configs = [
    "config_temp_77K.yaml",
    "config_temp_150K.yaml", 
    "config_temp_300K.yaml"
]

batch_runner = BatchRunner()
for config_file in configs:
    batch_runner.add_simulation(config_file)

# Run all simulations
results = batch_runner.run_all(parallel=True, n_jobs=4)

# Analyze batch results
batch_runner.generate_comparison_plots(results)
```

## Best Practices

### System Preparation

1. **PDB File Quality**:
   - Use high-resolution structures (< 2.5 Å)
   - Remove water molecules if not needed
   - Check for missing atoms or residues

2. **Quantum Subsystem Selection**:
   - Keep quantum region small (< 200 atoms)
   - Include chemically relevant atoms
   - Consider electronic coupling

3. **Parameter Selection**:
   - Start with template configurations
   - Validate against experimental data
   - Use appropriate time scales

### Simulation Strategy

1. **Start Small**:
   - Begin with simple systems
   - Use short simulation times
   - Gradually increase complexity

2. **Validation**:
   - Run benchmark tests first
   - Compare with analytical solutions
   - Check energy conservation

3. **Convergence Testing**:
   - Test different time steps
   - Verify statistical convergence
   - Check basis set completeness

### Performance Optimization

1. **Resource Management**:
   - Monitor memory usage
   - Use appropriate hardware
   - Enable GPU acceleration if available

2. **Simulation Parameters**:
   - Optimize time step size
   - Use adaptive integration
   - Set appropriate checkpoint intervals

3. **Parallel Processing**:
   - Use multiple CPU cores
   - Consider MPI for large systems
   - Balance load across resources

## Advanced Usage

### Custom Analysis Scripts

```python
# Custom analysis example
import numpy as np
from qbes.analysis import ResultsAnalyzer

class CustomAnalyzer(ResultsAnalyzer):
    def calculate_custom_metric(self, state_trajectory):
        """Calculate custom quantum metric."""
        metric_values = []
        
        for state in state_trajectory:
            # Custom calculation
            metric = np.trace(state @ self.custom_operator)
            metric_values.append(metric)
        
        return np.array(metric_values)

# Use custom analyzer
analyzer = CustomAnalyzer()
custom_results = analyzer.calculate_custom_metric(results.state_trajectory)
```

### Integration with Other Tools

#### Jupyter Notebooks

```python
# Jupyter notebook integration
%matplotlib inline
import matplotlib.pyplot as plt
from qbes import *

# Interactive analysis
results = load_results("./output")
plot_interactive_coherence(results)
```

#### External Analysis Tools

```python
# Export to other formats
from qbes.utils.export import DataExporter

exporter = DataExporter()
exporter.to_matlab("results.mat", results)
exporter.to_csv("results.csv", results)
exporter.to_hdf5("results.h5", results)
```

### Extending QBES

#### Custom Modules

```python
# Create custom module
from qbes.core.interfaces import SimulationModule

class MyCustomModule(SimulationModule):
    def initialize(self, config):
        # Custom initialization
        pass
    
    def step(self, current_state, time_step):
        # Custom simulation step
        return new_state
    
    def finalize(self, results):
        # Custom finalization
        return processed_results

# Register module
from qbes.simulation_engine import SimulationEngine
engine = SimulationEngine()
engine.register_module("my_module", MyCustomModule)
```

### Troubleshooting Simulations

#### Common Issues

1. **Convergence Problems**:
   - Reduce time step
   - Check initial conditions
   - Verify parameter ranges

2. **Memory Issues**:
   - Reduce system size
   - Increase checkpoint frequency
   - Use sparse matrix representations

3. **Performance Issues**:
   - Enable compiler optimizations
   - Use appropriate hardware
   - Profile code bottlenecks

#### Debugging Tools

```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Use simulation debugger
from qbes.utils.debugging import SimulationDebugger

debugger = SimulationDebugger()
debugger.attach_to_simulation(engine)
debugger.set_breakpoint("quantum_step")
```

## Next Steps

After mastering the basics:

1. Explore the [API Reference](api_reference.md) for detailed documentation
2. Study the [Theory and Methods](theory.md) for scientific background
3. Try advanced examples in the `examples/` directory
4. Join the community discussions
5. Contribute to the project development

For specific questions or issues, consult the [Troubleshooting Guide](troubleshooting.md) or contact the development team.