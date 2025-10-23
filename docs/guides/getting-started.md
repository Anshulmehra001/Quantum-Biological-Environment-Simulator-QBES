# Getting Started with QBES

## Quick Start (5 minutes)

### Step 1: Verify Installation
```bash
python -c "import qbes; print('QBES version:', qbes.__version__)"
```

### Step 2: Run Interactive Interface
```bash
python qbes_interactive.py
# Choose: 1 (Quick Start) → 1 (Photosynthesis) → y (Run)
```

### Step 3: View Sample Results
```bash
# View pre-generated results
notepad demo_simulation_results\simulation_summary.txt  # Windows
cat demo_simulation_results/simulation_summary.txt      # Mac/Linux
```

## Your First Simulation

### Generate Configuration
```bash
python -m qbes.cli generate-config my_simulation.yaml --template photosystem
```

### Run Simulation
```bash
python -m qbes.cli run my_simulation.yaml --verbose
```

### View Results
```bash
python -m qbes.cli view ./photosystem_output
```

## Understanding Results

### Key Metrics
- **Coherence Lifetime**: How long quantum effects last (femtoseconds)
- **Transfer Efficiency**: Energy transfer effectiveness (percentage)
- **Purity**: Quantum vs classical behavior (0.5-1.0)

### Typical Values
- **Good Coherence**: 100-500 fs
- **Excellent Efficiency**: >90%
- **Pure Quantum State**: Purity = 1.0

## Next Steps
1. Read the User Guide for detailed instructions
2. Explore example configurations
3. Run validation tests to verify accuracy
4. Try different biological systems (enzyme, membrane, etc.)