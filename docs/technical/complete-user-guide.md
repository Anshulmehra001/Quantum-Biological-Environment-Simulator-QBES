# QBES Complete User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Getting Started](#getting-started)
4. [Command Line Interface](#command-line-interface)
5. [Configuration System](#configuration-system)
6. [Running Simulations](#running-simulations)
7. [Understanding Results](#understanding-results)
8. [Python API](#python-api)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Introduction

QBES (Quantum Biological Environment Simulator) is a development-phase software platform for simulating quantum mechanical effects in biological systems. This guide provides comprehensive instructions for using all features of QBES.

**Developer**: Aniket Mehra (aniketmehra715@gmail.com)  
**Repository**: https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-  
**Version**: 1.2.0-dev  
**License**: Creative Commons BY-NC-SA 4.0  
**Status**: Development/Academic Project  

## Installation and Setup

### System Requirements
- **Python**: 3.8+ (3.9+ recommended)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 2GB+ free disk space
- **CPU**: Multi-core processor recommended

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-
cd QBES
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv qbes_env
source qbes_env/bin/activate  # Linux/Mac
qbes_env\Scripts\activate     # Windows

# Using conda
conda create -n qbes python=3.9
conda activate qbes
```

#### 3. Install Dependencies
```bash
# Install QBES
pip install -e .

# Or install with all dependencies
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
# Test import
python -c "import qbes; print('QBES version:', qbes.__version__)"

# Run quick demo
python demo_qbes.py

# Test interactive interface
python qbes_interactive.py
```

### Troubleshooting Installation
- **Import errors**: Check Python path and reinstall with `pip install -e . --force-reinstall`
- **Missing dependencies**: Install manually with `pip install numpy scipy matplotlib`
- **Permission errors**: Use virtual environment or `--user` flag

## Getting Started

### Quick Start (5 minutes)

#### Method 1: Interactive Interface (Easiest)
```bash
python qbes_interactive.py
```
Follow the menu:
1. Choose "1" for Quick Start
2. Choose "1" for Photosynthesis simulation
3. Type "y" to run simulation
4. Choose "1" for standard run

#### Method 2: Command Line
```bash
# Generate configuration
python -m qbes.cli generate-config my_first_sim.yaml --template photosystem

# Run simulation
python -m qbes.cli run my_first_sim.yaml --verbose

# View results
python -m qbes.cli view ./photosystem_output
```

### Understanding the Interface

#### Interactive Menu System
```
🎯 What would you like to do?
========================================
1. 🚀 Quick Start - Run a simulation
2. ⚙️  Create Configuration
3. 🔍 Validate System
4. 📊 View Results
5. 🧪 Run Benchmarks
6. 🛠️  Advanced Options
7. 📚 Help & Documentation
8. 🌐 Web Interface
9. ❌ Exit
```

Each option provides guided workflows for different tasks.

## Command Line Interface

### Main Commands

#### qbes generate-config
Create configuration files for simulations.

```bash
qbes generate-config OUTPUT_FILE [OPTIONS]

Options:
  --template [photosystem|enzyme|membrane|default]  Template type
  --interactive                                     Use guided wizard
```

**Examples:**
```bash
# Interactive wizard
qbes generate-config my_config.yaml --interactive

# Use template
qbes generate-config photosystem.yaml --template photosystem
```

#### qbes run
Execute quantum biological simulations.

```bash
qbes run CONFIG_FILE [OPTIONS]

Options:
  --verbose, -v              Detailed output
  --dry-run                  Test configuration only
  --debug-level LEVEL        Set logging level (DEBUG, INFO, WARNING, ERROR)
  --save-snapshots N         Save quantum states every N steps
  --output-dir DIR           Override output directory
```

**Examples:**
```bash
# Basic run
qbes run config.yaml

# Verbose with debugging
qbes run config.yaml --verbose --debug-level DEBUG

# Dry run (test only)
qbes run config.yaml --dry-run

# Save quantum state snapshots
qbes run config.yaml --save-snapshots 100
```

#### qbes validate
Run validation tests to verify system accuracy.

```bash
qbes validate [OPTIONS]

Options:
  --suite [quick|standard|full]  Validation suite type
  --output-dir DIR               Results directory
  --tolerance FLOAT              Accuracy tolerance
  --verbose, -v                  Detailed output
```

**Examples:**
```bash
# Quick validation (2 minutes)
qbes validate --suite quick

# Full validation (30 minutes)
qbes validate --suite full --verbose
```

#### qbes view
Display simulation results.

```bash
qbes view RESULTS_DIR [OPTIONS]

Options:
  --verbose, -v    Detailed results display
```

### Additional Commands

#### qbes status
Check simulation status.
```bash
qbes status OUTPUT_DIR
```

#### qbes info
Display system information.
```bash
qbes info
```

## Configuration System

### Configuration File Structure

QBES uses YAML configuration files with the following structure:

```yaml
# Basic configuration template
system:
  type: photosystem              # System type
  temperature: 300.0             # Temperature in Kelvin
  
simulation:
  simulation_time: 1.0e-12       # Total time in seconds
  time_step: 1.0e-15            # Integration step in seconds
  method: lindblad              # Simulation method
  
quantum_subsystem:
  selection_method: chromophores # How to select quantum atoms
  max_quantum_atoms: 150        # Maximum quantum atoms
  include_environment: true     # Include classical environment
  environment_radius: 8.0       # Environment radius in Angstroms
  
noise_model:
  type: protein_ohmic           # Noise model type
  coupling_strength: 1.0        # System-environment coupling
  cutoff_frequency: 5.0e13      # Spectral density cutoff
  reorganization_energy: 35.0   # Reorganization energy (cm⁻¹)
  temperature: 300.0            # Environment temperature
  
output:
  directory: ./results          # Output directory
  save_trajectory: true         # Save quantum state evolution
  save_snapshots: false        # Save intermediate states
  analysis: ["coherence", "purity", "energy"]  # Analysis types
```

### Configuration Parameters

#### System Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | "photosystem" | System type (photosystem, enzyme, membrane, default) |
| `temperature` | float | 300.0 | System temperature in Kelvin |

#### Simulation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulation_time` | float | 1.0e-12 | Total simulation time in seconds |
| `time_step` | float | 1.0e-15 | Integration time step in seconds |
| `method` | string | "lindblad" | Simulation method |

#### Quantum Subsystem Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selection_method` | string | "chromophores" | Atom selection method |
| `max_quantum_atoms` | int | 150 | Maximum atoms in quantum subsystem |
| `include_environment` | bool | true | Include classical environment |
| `environment_radius` | float | 8.0 | Environment radius in Angstroms |

#### Noise Model Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | "protein_ohmic" | Noise model type |
| `coupling_strength` | float | 1.0 | System-environment coupling |
| `cutoff_frequency` | float | 5.0e13 | Spectral density cutoff (Hz) |
| `reorganization_energy` | float | 35.0 | Reorganization energy (cm⁻¹) |

### Configuration Templates

#### Photosystem Template
```yaml
system:
  type: photosystem
  temperature: 300.0
  
simulation:
  simulation_time: 5.0e-12
  time_step: 1.0e-15
  
quantum_subsystem:
  selection_method: chromophores
  max_quantum_atoms: 150
  
noise_model:
  type: protein_ohmic
  coupling_strength: 1.0
  reorganization_energy: 35.0
```

#### Enzyme Template
```yaml
system:
  type: enzyme
  temperature: 310.0
  
simulation:
  simulation_time: 10.0e-12
  time_step: 2.0e-15
  
quantum_subsystem:
  selection_method: active_site
  max_quantum_atoms: 100
  
noise_model:
  type: protein_ohmic
  coupling_strength: 1.5
  reorganization_energy: 50.0
```

## Running Simulations

### Simulation Workflow

#### 1. Prepare Configuration
```bash
# Interactive configuration
qbes generate-config my_sim.yaml --interactive

# Or use template
qbes generate-config my_sim.yaml --template photosystem
```

#### 2. Validate Configuration
```bash
qbes validate my_sim.yaml
```

#### 3. Run Simulation
```bash
# Standard run
qbes run my_sim.yaml --verbose

# With debugging
qbes run my_sim.yaml --debug-level DEBUG --save-snapshots 50
```

#### 4. Monitor Progress
During simulation, you'll see output like:
```
Loading configuration from: my_sim.yaml
✅ Configuration loaded and validated successfully
🔧 Initializing molecular dynamics system...
⚛️  Setting up quantum subsystem...
🧮 Constructing system Hamiltonian...
🎯 Running quantum evolution...
📊 Analyzing results...
✅ Simulation completed successfully!
```

#### 5. View Results
```bash
qbes view ./results_directory
```

### Simulation Types

#### Photosynthesis Simulation
Studies quantum coherence in light-harvesting complexes:
- Energy transfer efficiency
- Coherence lifetimes
- Temperature effects
- Environmental decoherence

#### Enzyme Catalysis Simulation
Models quantum tunneling in enzyme active sites:
- Tunneling rates
- Activation energy reduction
- Temperature dependence
- Catalytic enhancement

#### Membrane Protein Simulation
Analyzes quantum effects in membrane proteins:
- Ion channel dynamics
- Conformational changes
- Lipid-protein interactions
- Electrostatic effects

## Understanding Results

### Output Files

After simulation, QBES creates several output files:

```
results_directory/
├── simulation_results.json      # Complete results data
├── simulation_summary.txt       # Human-readable summary
├── time_evolution_data.csv      # Time-series data (Excel-compatible)
├── detailed_analysis_report.txt # Scientific analysis
└── plots/                       # Visualization plots
    ├── coherence_evolution.png
    ├── population_dynamics.png
    └── energy_conservation.png
```

### Key Results Interpretation

#### Coherence Lifetime
- **Definition**: How long quantum coherence persists
- **Units**: Femtoseconds (fs)
- **Typical Values**: 100-500 fs for biological systems
- **Interpretation**: Longer lifetimes indicate stronger quantum effects

#### Transfer Efficiency
- **Definition**: Percentage of energy successfully transferred
- **Units**: Percentage (%)
- **Typical Values**: 85-95% for efficient systems
- **Interpretation**: Higher efficiency indicates quantum advantage

#### Purity
- **Definition**: Measure of quantum vs classical behavior
- **Range**: 0.5 (classical) to 1.0 (pure quantum)
- **Interpretation**: Higher purity indicates more quantum behavior

#### Decoherence Rate
- **Definition**: Rate at which quantum coherence is lost
- **Units**: Inverse time (ps⁻¹)
- **Interpretation**: Lower rates indicate better coherence preservation

### Sample Results Analysis

```
QBES Simulation Results Summary
========================================

🔬 SYSTEM CONFIGURATION
System Type: photosystem
Temperature: 300.0 K
Quantum Atoms: 150
Simulation Time: 1.0 ps

⚛️  QUANTUM PROPERTIES
Coherence Lifetime: 245.0 fs
Decoherence Rate: 4.08 ps⁻¹
Initial Coherence: 0.993
Final Coherence: 0.015
Initial Purity: 0.997
Final Purity: 0.508

📊 PERFORMANCE METRICS
Transfer Efficiency: 94.2%
Energy Conservation: ±0.001004 eV
Quantum Advantage: Significant - coherence enhances efficiency
```

**Interpretation**:
- **Coherence Lifetime (245 fs)**: Excellent for biological systems
- **Transfer Efficiency (94.2%)**: Near-optimal performance
- **Energy Conservation**: Excellent numerical accuracy
- **Quantum Advantage**: Coherence significantly enhances energy transfer

## Python API

### Basic Usage

```python
from qbes import ConfigurationManager, SimulationEngine
from qbes.analysis import ResultsAnalyzer

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("config.yaml")

# Run simulation
engine = SimulationEngine()
engine.initialize_simulation(config)
results = engine.run_simulation()

# Analyze results
analyzer = ResultsAnalyzer()
coherence_lifetime = analyzer.calculate_coherence_lifetime(results)
efficiency = analyzer.calculate_transfer_efficiency(results)

print(f"Coherence lifetime: {coherence_lifetime*1e15:.1f} fs")
print(f"Transfer efficiency: {efficiency*100:.1f}%")
```

### Advanced API Usage

#### Custom Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Extract time evolution data
time_points = results.time_points
coherence_data = results.coherence_trajectory

# Custom coherence analysis
def analyze_coherence_decay(time, coherence):
    """Analyze coherence decay and fit exponential"""
    from scipy.optimize import curve_fit
    
    def exponential_decay(t, A, tau):
        return A * np.exp(-t / tau)
    
    popt, _ = curve_fit(exponential_decay, time, coherence)
    lifetime = popt[1]
    return lifetime

lifetime = analyze_coherence_decay(time_points, coherence_data)
print(f"Fitted coherence lifetime: {lifetime*1e15:.1f} fs")

# Custom plotting
plt.figure(figsize=(10, 6))
plt.plot(time_points * 1e15, coherence_data, 'b-', linewidth=2)
plt.xlabel('Time (fs)')
plt.ylabel('Quantum Coherence')
plt.title('Coherence Evolution in Biological System')
plt.grid(True, alpha=0.3)
plt.savefig('custom_coherence_plot.png', dpi=300)
```

#### Configuration Manipulation
```python
# Create custom configuration
from qbes.core.data_models import SimulationConfig

config = SimulationConfig(
    system_type="photosystem",
    temperature=77.0,  # Low temperature
    simulation_time=10e-12,  # 10 ps
    time_step=1e-15,
    quantum_subsystem_selection="chromophores",
    max_quantum_atoms=200,
    noise_model_type="protein_ohmic",
    coupling_strength=0.5,  # Weaker coupling
    output_directory="./low_temp_results"
)

# Run with custom configuration
engine = SimulationEngine()
results = engine.run_simulation(config)
```

## Advanced Features

### Debugging and Development

#### Dry-Run Mode
Test configuration without full simulation:
```bash
qbes run config.yaml --dry-run
```

This validates:
- Configuration file syntax
- System setup and initialization
- Memory requirements estimation
- Expected runtime calculation

#### Debug Logging
Enable detailed debugging information:
```bash
qbes run config.yaml --debug-level DEBUG
```

Debug output includes:
- Step-by-step simulation progress
- Numerical stability checks
- Memory usage monitoring
- Performance timing information

#### State Snapshots
Save intermediate quantum states for analysis:
```bash
qbes run config.yaml --save-snapshots 100
```

Snapshots are saved every 100 steps and can be analyzed separately:
```python
from qbes.utils.file_io import FileIOUtils

# Load specific snapshot
state = FileIOUtils.load_snapshot("output/snapshots/snapshot_step_1000.pkl")

# Analyze snapshot
from qbes.analysis import ResultsAnalyzer
analyzer = ResultsAnalyzer()
coherence = analyzer.calculate_coherence_measures(state.density_matrix)
```

### Performance Optimization

#### Memory Management
For large systems, optimize memory usage:
```yaml
simulation:
  memory_optimization: true
  chunk_size: 500  # Process atoms in chunks

output:
  save_trajectory: false  # Don't save full trajectory
  compress_output: true   # Compress output files
```

#### Parallel Processing
Enable multi-core processing (if available):
```bash
export OMP_NUM_THREADS=8
qbes run config.yaml
```

### Validation and Quality Assurance

#### Validation Suites
QBES includes three validation levels:

1. **Quick Validation** (2 minutes):
   ```bash
   qbes validate --suite quick
   ```
   - Basic functionality tests
   - Installation verification
   - Core algorithm validation

2. **Standard Validation** (10 minutes):
   ```bash
   qbes validate --suite standard
   ```
   - Comprehensive functionality tests
   - Accuracy validation against analytical solutions
   - Performance benchmarking

3. **Full Validation** (30 minutes):
   ```bash
   qbes validate --suite full
   ```
   - Complete test suite
   - Literature benchmark comparisons
   - Stress testing and edge cases

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems

**Issue**: Import errors after installation
```
ImportError: No module named 'qbes'
```
**Solution**:
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall QBES
pip install -e . --force-reinstall

# Or install in development mode
pip install -e .[dev]
```

**Issue**: Missing dependencies
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific packages
pip install numpy scipy matplotlib qutip
```

#### Configuration Errors

**Issue**: Configuration validation fails
```
Configuration validation failed: Invalid parameter 'xyz'
```
**Solution**:
1. Check parameter names and values in configuration file
2. Use template as starting point:
   ```bash
   qbes generate-config new_config.yaml --template photosystem
   ```
3. Validate configuration:
   ```bash
   qbes validate new_config.yaml
   ```

#### Simulation Errors

**Issue**: Simulation fails with numerical errors
```
RuntimeError: Numerical instability detected
```
**Solution**:
1. Reduce time step:
   ```yaml
   simulation:
     time_step: 5.0e-16  # Smaller time step
   ```
2. Enable debugging:
   ```bash
   qbes run config.yaml --debug-level DEBUG
   ```
3. Use dry-run to check setup:
   ```bash
   qbes run config.yaml --dry-run
   ```

**Issue**: Memory errors with large systems
```
MemoryError: Unable to allocate array
```
**Solution**:
1. Reduce system size:
   ```yaml
   quantum_subsystem:
     max_quantum_atoms: 50  # Smaller system
   ```
2. Enable memory optimization:
   ```yaml
   simulation:
     memory_optimization: true
   ```
3. Use shorter simulation time:
   ```yaml
   simulation:
     simulation_time: 1.0e-12  # 1 ps instead of 10 ps
   ```

#### Performance Issues

**Issue**: Slow simulation speed
**Solutions**:
1. Reduce system size or simulation time
2. Increase time step (if numerically stable)
3. Disable trajectory saving for production runs:
   ```yaml
   output:
     save_trajectory: false
   ```
4. Use performance profiling:
   ```bash
   qbes run config.yaml --debug-level INFO --verbose
   ```

### Getting Help

#### Documentation Resources
- **User Guide**: This document
- **API Reference**: `docs/technical/api-reference.md`
- **Theory Guide**: `docs/technical/theory.md`
- **Examples**: `docs/examples/`

#### Support Channels
- **Email**: aniketmehra715@gmail.com
- **Repository**: https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-
- **Issues**: Report bugs and request features via GitHub issues

#### Self-Help Tools
```bash
# System information
qbes info

# Validation tests
qbes validate --suite quick

# Interactive help
qbes --help
qbes run --help
```

---

**QBES Complete User Guide v1.2.0-dev**  
**Author**: Aniket Mehra  
**Contact**: aniketmehra715@gmail.com  
**Repository**: https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-  
**Last Updated**: October 2025