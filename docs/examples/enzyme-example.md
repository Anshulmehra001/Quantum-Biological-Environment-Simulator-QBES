# Enzyme Catalysis Simulation Example

## Overview
This example demonstrates how to simulate quantum tunneling effects in enzyme active sites using QBES.

## Scientific Background
Enzymes use quantum tunneling to enhance reaction rates by orders of magnitude. This simulation models:
- Quantum tunneling through reaction barriers
- Active site dynamics and conformational changes
- Temperature effects on catalytic efficiency

## Configuration

### Enzyme Active Site Configuration
```yaml
# enzyme_example.yaml
system:
  type: enzyme
  temperature: 310.0  # Body temperature (K)
  
simulation:
  simulation_time: 10.0e-12  # 10 picoseconds
  time_step: 2.0e-15        # 2 femtoseconds
  method: lindblad
  
quantum_subsystem:
  selection_method: active_site
  max_quantum_atoms: 100
  include_environment: true
  environment_radius: 6.0
  
noise_model:
  type: protein_ohmic
  coupling_strength: 1.5
  reorganization_energy: 50.0
  temperature: 310.0
  
output:
  directory: ./enzyme_results
  save_trajectory: true
  analysis: ["tunneling_rate", "activation_energy", "reaction_coordinate"]
```

## Running the Simulation

### Step 1: Generate Configuration
```bash
python -m qbes.cli generate-config enzyme.yaml --template enzyme
```

### Step 2: Run Simulation
```bash
python -m qbes.cli run enzyme.yaml --verbose --save-snapshots 50
```

## Expected Results

### Typical Output Values
- **Tunneling Enhancement**: 10-100x rate increase
- **Activation Energy**: Reduced by 5-15 kJ/mol
- **Reaction Rate**: 10³-10⁶ s⁻¹
- **Temperature Dependence**: Non-Arrhenius behavior

### Result Interpretation
```
Quantum Tunneling Rate: 2.1 × 10¹² Hz
- Significant enhancement over classical rate
- Enables catalysis at biological temperatures

Activation Energy Reduction: 12.3 kJ/mol
- Quantum tunneling lowers effective barrier
- Consistent with experimental observations

Temperature Coefficient: 1.8
- Reduced temperature dependence vs classical
- Indicates quantum tunneling contribution
```

## Python API Example

```python
from qbes import ConfigurationManager, SimulationEngine
from qbes.analysis import ResultsAnalyzer

# Load and run simulation
config_manager = ConfigurationManager()
config = config_manager.load_config("enzyme.yaml")

engine = SimulationEngine()
results = engine.run_simulation(config)

# Analyze tunneling effects
analyzer = ResultsAnalyzer()
tunneling_rate = analyzer.calculate_tunneling_rate(results)
enhancement_factor = analyzer.calculate_quantum_enhancement(results)

print(f"Tunneling rate: {tunneling_rate:.2e} Hz")
print(f"Quantum enhancement: {enhancement_factor:.1f}x")
```

## Biological Applications

### Research Areas
- **Drug Design**: Understanding enzyme-inhibitor interactions
- **Metabolic Engineering**: Optimizing enzymatic pathways
- **Protein Evolution**: How enzymes evolved quantum effects
- **Disease Mechanisms**: Enzyme dysfunction in disease

### Key Insights
1. **Quantum Tunneling**: Enables reactions impossible classically
2. **Active Site Design**: Precise positioning for optimal tunneling
3. **Dynamic Effects**: Protein motions modulate tunneling rates
4. **Temperature Independence**: Quantum effects persist at body temperature

This example shows how QBES can reveal quantum mechanical aspects of enzyme catalysis crucial for understanding biological function.