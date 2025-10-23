# QBES User Guide

## Overview
QBES (Quantum Biological Environment Simulator) is a comprehensive platform for simulating quantum mechanical effects in biological systems.

## Command Line Interface

### Main Commands

#### Generate Configuration
```bash
qbes generate-config CONFIG_FILE [OPTIONS]
```
- `--template`: photosystem, enzyme, membrane, default
- `--interactive`: Use guided wizard

#### Run Simulation
```bash
qbes run CONFIG_FILE [OPTIONS]
```
- `--verbose`: Detailed output
- `--dry-run`: Test configuration only
- `--debug-level DEBUG`: Enable debugging
- `--save-snapshots N`: Save quantum states

#### Validate System
```bash
qbes validate [OPTIONS]
```
- `--suite quick`: 2-minute validation
- `--suite standard`: 10-minute validation  
- `--suite full`: 30-minute validation

#### View Results
```bash
qbes view RESULTS_DIR
```

## Configuration Files

### Basic Structure
```yaml
system:
  type: photosystem
  temperature: 300.0
  
simulation:
  simulation_time: 1.0e-12
  time_step: 1.0e-15
  
quantum_subsystem:
  selection_method: chromophores
  max_quantum_atoms: 150
  
noise_model:
  type: protein_ohmic
  coupling_strength: 1.0
  
output:
  directory: ./results
  save_trajectory: true
```

### Templates Available
- **photosystem**: Photosynthetic complexes
- **enzyme**: Enzyme active sites
- **membrane**: Membrane proteins
- **default**: Basic quantum systems

## Python API

### Basic Usage
```python
from qbes import ConfigurationManager, SimulationEngine

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("config.yaml")

# Run simulation
engine = SimulationEngine()
results = engine.run_simulation(config)

# Analyze results
print(f"Coherence lifetime: {results.coherence_lifetime:.2e} s")
```

### Analysis Tools
```python
from qbes.analysis import ResultsAnalyzer

analyzer = ResultsAnalyzer()
results = analyzer.load_results("./output")
lifetime = analyzer.calculate_coherence_lifetime(results)
efficiency = analyzer.calculate_transfer_efficiency(results)
```

## Biological Systems

### Photosynthetic Complexes
- Light-harvesting complexes (LHC-II, FMO)
- Energy transfer efficiency
- Quantum coherence effects

### Enzyme Catalysis
- Active site quantum tunneling
- Catalytic rate enhancement
- Temperature effects

### Membrane Proteins
- Ion channel dynamics
- Membrane protein conformations
- Lipid-protein interactions

## Troubleshooting

### Common Issues
1. **Configuration errors**: Use `qbes validate config.yaml`
2. **Memory issues**: Reduce `max_quantum_atoms`
3. **Slow performance**: Use `--dry-run` to test first

### Getting Help
- Check error messages for specific guidance
- Use `--verbose` flag for detailed output
- Run validation tests to verify installation