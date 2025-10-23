# Water Box Example

**Difficulty:** Beginner  
**System:** 10 water molecules in a periodic box  
**Purpose:** Learn basic QBES concepts and quantum coherence in simple systems  
**Estimated Runtime:** 2-5 minutes  

## Overview

This example demonstrates quantum coherence effects in a simple water box system. Water molecules exhibit quantum behavior on ultrafast timescales, making this an ideal system for learning QBES fundamentals.

## Scientific Background

Water molecules can exhibit quantum coherence through:
- Hydrogen bond dynamics
- Vibrational coupling between molecules
- Collective quantum effects in the liquid phase

At room temperature, quantum coherence in water typically lasts 100-500 femtoseconds before environmental decoherence destroys quantum effects.

## System Details

- **Molecules:** 10 water molecules (30 atoms total)
- **Quantum subsystem:** 6 atoms (2 water molecules)
- **Classical environment:** 24 atoms (8 water molecules)
- **Temperature:** 300 K (room temperature)
- **Simulation time:** 1 picosecond
- **Time step:** 1 femtosecond

## Files

- `system.pdb` - Water box structure
- `config.yaml` - QBES configuration
- `expected_results.json` - Reference results
- `analyze_results.py` - Analysis script

## Running the Example

### Step 1: Run Simulation

```bash
cd examples/water_box/
qbes run config.yaml --verbose
```

Expected output:
```
🚀 Starting QBES simulation...
📁 Output directory: ./results
🔧 Loading configuration...

📊 System Information:
   • Molecules: 10 water molecules
   • Quantum atoms: 6 (2 molecules)
   • Classical atoms: 24 (8 molecules)
   • Temperature: 300.0 K

⚡ Starting MD initialization...
✅ MD system initialized (0.5s)

🔬 Constructing quantum Hamiltonian...
✅ Hamiltonian constructed (0.2s)

🌊 Starting quantum evolution...
Progress: 100.0% | Elapsed: 45.2s

📈 Analyzing results...
✅ Analysis complete (1.1s)

📊 SIMULATION RESULTS SUMMARY
================================================================================
System: Water Box (6 quantum atoms, 24 classical atoms)
Temperature: 300.0 K | Simulation Time: 1.0 ps

QUANTUM COHERENCE ANALYSIS:
• Coherence Lifetime:           245.7 ± 12.3 fs
• Decoherence Rate:            4.07e+12 ± 2.1e+11 Hz
• Final Quantum Purity:        0.234 ± 0.015

ENERGY ANALYSIS:
• Initial Energy:              -76.3 kcal/mol
• Final Energy:                -76.2 kcal/mol
• Energy Conservation Error:    0.13%

✅ Simulation completed successfully!
```

### Step 2: Analyze Results

```bash
python analyze_results.py
```

This script will:
- Load simulation results
- Calculate additional coherence measures
- Generate publication-quality plots
- Compare with expected values

### Step 3: View Results

```bash
ls results/
```

Output files:
```
simulation_results.json    # Complete results data
time_evolution_data.csv   # Time-series data
results_summary.txt       # Human-readable summary
analysis_report.txt       # Detailed analysis
plots/                    # Generated figures
├── coherence_evolution.png
├── population_dynamics.png
├── energy_conservation.png
└── hydrogen_bond_analysis.png
```

## Expected Results

### Key Metrics

| Metric | Expected Value | Tolerance | Physical Meaning |
|--------|----------------|-----------|------------------|
| Coherence Lifetime | 245.7 fs | ±25 fs | How long quantum effects persist |
| Decoherence Rate | 4.07×10¹² Hz | ±20% | Rate of quantum → classical transition |
| Final Purity | 0.234 | ±0.05 | Quantum vs classical character |
| Energy Conservation | <1% error | <2% | Simulation accuracy check |

### Physical Interpretation

**Coherence Lifetime (245.7 fs):** This is typical for water at room temperature. Quantum coherence is destroyed by thermal motion and intermolecular interactions on this timescale.

**Decoherence Rate (4.07×10¹² Hz):** The inverse of the coherence lifetime, indicating how quickly the environment destroys quantum superposition states.

**Final Purity (0.234):** Shows the system becomes mostly classical (0.0) rather than purely quantum (1.0) due to environmental decoherence.

## Learning Objectives

After completing this example, you should understand:

1. **Basic QBES workflow:** Configuration → Simulation → Analysis
2. **Quantum coherence:** How quantum effects decay over time
3. **Decoherence:** How environment destroys quantum behavior
4. **Energy conservation:** Checking simulation accuracy
5. **Result interpretation:** Understanding physical meaning of outputs

## Exercises

### Exercise 1: Temperature Effects

Modify the temperature in `config.yaml` and observe how it affects coherence:

```yaml
simulation:
  temperature: 77.0    # Try: 77K, 150K, 200K, 300K, 400K
```

**Question:** How does coherence lifetime change with temperature?

### Exercise 2: System Size Effects

Change the quantum subsystem size:

```yaml
quantum_subsystem:
  max_quantum_atoms: 12    # Try: 3, 6, 9, 12, 15
```

**Question:** How does quantum subsystem size affect results?

### Exercise 3: Coupling Strength

Modify the environmental coupling:

```yaml
noise_model:
  coupling_strength: 2.0    # Try: 0.5, 1.0, 2.0, 5.0
```

**Question:** How does stronger coupling affect decoherence?

## Troubleshooting

### Issue: Simulation runs but gives unrealistic results

**Possible causes:**
- Time step too large
- System not equilibrated
- Incorrect force field parameters

**Solutions:**
```yaml
simulation:
  time_step: 5.0e-16    # Reduce time step
  equilibration_time: 1.0e-13    # Add equilibration
```

### Issue: Very slow simulation

**Solutions:**
- Reduce simulation time: `simulation_time: 5.0e-13`
- Reduce quantum atoms: `max_quantum_atoms: 3`
- Increase time step: `time_step: 2.0e-15`

### Issue: Memory errors

**Solutions:**
- Use smaller system
- Disable trajectory saving: `save_trajectory: false`
- Increase checkpoint interval: `checkpoint_interval: 2000`

## Next Steps

After mastering this example:

1. **Try the benzene example** for aromatic systems
2. **Modify parameters** to see their effects
3. **Create your own water system** with different sizes
4. **Read about hydrogen bonding** in quantum systems
5. **Explore the photosystem example** for biological relevance

## References

1. Coherence in liquid water: *Nature* **434**, 199-202 (2005)
2. Quantum effects in water: *Science* **306**, 851-853 (2004)
3. Hydrogen bond dynamics: *J. Chem. Phys.* **125**, 194521 (2006)

## Validation

To validate your results against expected values:

```bash
python validate_results.py
```

This compares your simulation results with reference data and reports any significant deviations.