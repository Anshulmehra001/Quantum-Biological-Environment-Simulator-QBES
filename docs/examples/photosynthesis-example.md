# Photosynthesis Simulation Example

## Overview
This example demonstrates how to simulate quantum coherence effects in photosynthetic light-harvesting complexes using QBES.

## Scientific Background
Photosynthetic complexes use quantum coherence to achieve near-perfect energy transfer efficiency. This simulation models:
- Quantum coherence between chromophore sites
- Environmental decoherence from protein dynamics
- Energy transfer pathways and efficiency

## Configuration

### Basic Photosystem Configuration
```yaml
# photosystem_example.yaml
system:
  type: photosystem
  temperature: 300.0  # Room temperature (K)
  
simulation:
  simulation_time: 5.0e-12  # 5 picoseconds
  time_step: 1.0e-15       # 1 femtosecond
  method: lindblad
  
quantum_subsystem:
  selection_method: chromophores
  max_quantum_atoms: 150
  include_environment: true
  environment_radius: 8.0
  
noise_model:
  type: protein_ohmic
  coupling_strength: 1.0
  cutoff_frequency: 5.0e13
  reorganization_energy: 35.0
  temperature: 300.0
  
output:
  directory: ./photosystem_results
  save_trajectory: true
  save_snapshots: true
  analysis: ["coherence", "efficiency", "energy_transfer"]
```

## Running the Simulation

### Step 1: Generate Configuration
```bash
# Use template
python -m qbes.cli generate-config photosystem.yaml --template photosystem

# Or create custom configuration file with above content
```

### Step 2: Validate Configuration
```bash
python -m qbes.cli validate photosystem.yaml
```

### Step 3: Run Simulation
```bash
# Standard run
python -m qbes.cli run photosystem.yaml --verbose

# With debugging and snapshots
python -m qbes.cli run photosystem.yaml --debug-level DEBUG --save-snapshots 100
```

### Step 4: View Results
```bash
python -m qbes.cli view ./photosystem_results
```

## Expected Results

### Typical Output Values
- **Coherence Lifetime**: 200-500 femtoseconds
- **Energy Transfer Efficiency**: 90-95%
- **Final Purity**: 0.6-0.8
- **Decoherence Rate**: 2-5 ps⁻¹

### Result Interpretation
```
Coherence Lifetime: 245 fs
- Excellent for biological systems at room temperature
- Enables efficient energy transfer before decoherence

Transfer Efficiency: 94.2%
- Near-optimal performance demonstrates quantum advantage
- Consistent with experimental observations

Energy Conservation: ±0.001 eV
- Excellent conservation validates simulation accuracy
```

## Python API Example

### Complete Workflow
```python
from qbes import ConfigurationManager, SimulationEngine
from qbes.analysis import ResultsAnalyzer

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("photosystem.yaml")

# Run simulation
engine = SimulationEngine()
engine.initialize_simulation(config)
results = engine.run_simulation()

# Analyze results
analyzer = ResultsAnalyzer()
coherence_lifetime = analyzer.calculate_coherence_lifetime(results)
efficiency = analyzer.calculate_transfer_efficiency(results)

print(f"Coherence lifetime: {coherence_lifetime*1e15:.1f} fs")
print(f"Energy transfer efficiency: {efficiency*100:.1f}%")

# Generate plots
analyzer.generate_plots(results, "./plots")
```

### Custom Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Extract time evolution data
time_points = results.time_points
coherence_data = results.coherence_trajectory
population_data = results.population_trajectory

# Plot coherence decay
plt.figure(figsize=(10, 6))
plt.plot(time_points * 1e15, coherence_data, 'b-', linewidth=2)
plt.xlabel('Time (fs)')
plt.ylabel('Quantum Coherence')
plt.title('Coherence Decay in Photosynthetic Complex')
plt.grid(True, alpha=0.3)
plt.savefig('coherence_decay.png', dpi=300)

# Fit exponential decay
def exponential_decay(t, A, tau):
    return A * np.exp(-t / tau)

from scipy.optimize import curve_fit
popt, _ = curve_fit(exponential_decay, time_points, coherence_data)
lifetime = popt[1]
print(f"Fitted coherence lifetime: {lifetime*1e15:.1f} fs")
```

## Variations and Extensions

### Low Temperature Study
```yaml
# Study quantum effects at low temperature
system:
  temperature: 77.0  # Liquid nitrogen temperature
  
noise_model:
  temperature: 77.0
  coupling_strength: 0.5  # Reduced coupling at low T
```

### Different Chromophore Arrangements
```yaml
quantum_subsystem:
  selection_method: custom
  custom_selection: "resname CHL BCL"  # Chlorophyll and bacteriochlorophyll
  max_quantum_atoms: 200
```

### Extended Simulation Time
```yaml
simulation:
  simulation_time: 50.0e-12  # 50 picoseconds
  time_step: 2.0e-15        # Larger time step for longer runs
```

## Biological Significance

### Research Applications
- **Solar Cell Design**: Understanding optimal energy transfer mechanisms
- **Photosynthetic Efficiency**: Factors affecting energy conversion
- **Environmental Effects**: Impact of temperature and disorder
- **Evolutionary Optimization**: How nature optimizes quantum effects

### Key Insights
1. **Quantum Coherence**: Enables wavelike energy transport
2. **Environmental Tuning**: Protein environment optimizes coherence lifetime
3. **Temperature Effects**: Balance between coherence and thermal activation
4. **Efficiency Optimization**: Near-perfect energy transfer through quantum effects

## Troubleshooting

### Common Issues
1. **Long simulation times**: Reduce system size or simulation time
2. **Memory usage**: Use memory optimization options
3. **Convergence problems**: Check time step size and numerical stability

### Performance Tips
- Start with smaller systems (50-100 atoms)
- Use dry-run mode to estimate computational requirements
- Monitor memory usage during simulation
- Save checkpoints for long simulations

This example demonstrates the power of QBES for studying quantum effects in photosynthetic systems and provides a foundation for more advanced research applications.