# QBES Complete User Guide
## Quantum Biological Environment Simulator

### 🚀 Quick Start

#### Option 1: Easy Startup (Recommended)
```bash
python start_qbes.py
```
This launches an interactive menu with all options.

#### Option 2: Direct Web Interface
```bash
cd website
python app.py
```
Then open: http://localhost:5000

#### Option 3: Command Line Interface
```bash
python qbes_cli.py interactive
```

---

## 🌐 Web Interface Guide

### Starting the Web Interface
1. Run `python start_qbes.py` and choose option 1
2. Or directly: `cd website && python app.py`
3. Open browser to: http://localhost:5000

### Using the Simulator
1. **Home Page**: Overview and system status
2. **Simulator Page**: Full interactive interface
   - Choose preset configurations or customize parameters
   - Adjust temperature, energy gap, coupling strength
   - Select noise model (protein, membrane, solvent, thermal)
   - Set simulation time
   - Click "Run Simulation"

### Features
- **Real-time Results**: Instant visualization of quantum properties
- **Interactive Plots**: Coherence, purity, and combined views
- **Export Options**: Download results as JSON or CSV
- **Simulation History**: Track previous runs
- **Mobile Friendly**: Works on tablets and phones

### Preset Configurations
- **Photosynthetic Complex**: Energy transfer in light-harvesting
- **Enzyme Active Site**: Quantum tunneling in catalysis
- **Membrane Protein**: Quantum effects in membranes
- **Cryogenic System**: Low-temperature long coherence

---

## 💻 Command Line Interface Guide

### Basic Commands
```bash
# Interactive mode (recommended)
python qbes_cli.py interactive

# Create configuration templates
python qbes_cli.py create-config photosynthesis
python qbes_cli.py create-config enzyme
python qbes_cli.py create-config membrane
python qbes_cli.py create-config basic

# Run simulations
python qbes_cli.py run configs/photosynthesis_config.json

# List all simulation results
python qbes_cli.py list

# View specific results
python qbes_cli.py view simulation_results/sim_20250102_143022

# Start web interface from CLI
python qbes_cli.py web
```

### Interactive Mode Commands
```
QBES> create photosynthesis    # Create config template
QBES> run configs/photosynthesis_config.json  # Run simulation
QBES> list                     # List results
QBES> view simulation_results/sim_20250102_143022  # View results
QBES> web                      # Start web interface
QBES> help                     # Show help
QBES> quit                     # Exit
```

---

## 📊 Understanding Results

### Key Metrics
- **Coherence Lifetime**: How long quantum effects persist (picoseconds)
- **Decoherence Rate**: Speed of quantum decay (ps⁻¹)
- **Purity**: Measure of quantum state "purity" (0.5-1.0)
- **Final Coherence**: Remaining coherence at end of simulation

### Typical Values
- **Photosynthesis**: 10-50 ps coherence lifetime
- **Enzymes**: 1-10 ps coherence lifetime  
- **Membranes**: 5-20 ps coherence lifetime
- **Cold Systems**: 50-500 ps coherence lifetime

### File Outputs
- `simulation_results.json`: Complete numerical data
- `time_evolution_data.csv`: Time series for plotting
- `simulation_summary.txt`: Human-readable summary
- `analysis_report.txt`: Detailed scientific analysis

---

## 🔧 Configuration Guide

### Configuration File Structure
```json
{
  "system": {
    "type": "two_level",
    "temperature": 300.0,
    "hamiltonian": {
      "energy_gap": 2.0,
      "coupling": 0.1
    }
  },
  "simulation": {
    "time_step": 0.1,
    "total_time": 10.0,
    "method": "lindblad"
  },
  "noise": {
    "model": "protein_ohmic",
    "strength": 0.1
  },
  "output": {
    "save_trajectory": true,
    "analysis": ["coherence", "purity", "energy"]
  }
}
```

### Parameter Meanings
- **temperature**: System temperature in Kelvin (77-400K typical)
- **energy_gap**: Energy difference between quantum states (eV)
- **coupling**: Interaction strength between states (eV)
- **time_step**: Simulation time resolution (ps)
- **total_time**: Total simulation duration (ps)
- **noise.model**: Environmental noise type
- **noise.strength**: Noise intensity (0.01-1.0)

### Noise Models
- **protein_ohmic**: Protein environment with Ohmic spectral density
- **membrane_fluctuations**: Lipid membrane dynamics
- **solvent_dynamics**: Aqueous solvent effects
- **thermal_bath**: General thermal environment

---

## 📈 Visualization and Analysis

### Built-in Plots
1. **Coherence Evolution**: Shows quantum coherence decay over time
2. **Purity Evolution**: Shows quantum state purity changes
3. **Combined View**: Both coherence and purity together

### External Analysis
1. **Excel/Spreadsheet**: Open `time_evolution_data.csv`
2. **Python/Matplotlib**: Use provided plotting scripts
3. **Online Tools**: Upload CSV to plot.ly, Google Sheets
4. **Jupyter Notebooks**: Import data for custom analysis

### Creating Custom Plots
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('simulation_results/sim_20250102_143022/time_evolution_data.csv')

# Plot coherence
plt.figure(figsize=(10, 6))
plt.plot(df['Time_ps'], df['Coherence'], 'b-', linewidth=2)
plt.xlabel('Time (ps)')
plt.ylabel('Quantum Coherence')
plt.title('Coherence Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🔬 Scientific Applications

### Photosynthesis Research
```bash
# Create photosynthesis config
python qbes_cli.py create-config photosynthesis

# Modify for different conditions
# Edit configs/photosynthesis_config.json
# Change temperature, coupling strength

# Run simulation
python qbes_cli.py run configs/photosynthesis_config.json
```

### Enzyme Catalysis Studies
```bash
# Create enzyme config
python qbes_cli.py create-config enzyme

# Run at different temperatures
# Edit temperature in config file (280K, 300K, 320K)
python qbes_cli.py run configs/enzyme_config.json
```

### Membrane Protein Analysis
```bash
# Create membrane config
python qbes_cli.py create-config membrane

# Study different lipid environments
# Modify noise strength for different membrane types
python qbes_cli.py run configs/membrane_config.json
```

---

## 🛠️ Installation and Setup

### Requirements
- Python 3.7 or higher
- Required packages: Flask, NumPy, Matplotlib, Pandas

### Automatic Installation
```bash
# Run startup script (installs dependencies automatically)
python start_qbes.py

# Or install manually
pip install -r requirements_web.txt
```

### Manual Installation
```bash
pip install flask flask-cors numpy matplotlib pandas scipy
```

### Verification
```bash
# Check system status
python start_qbes.py
# Choose option 5 (System Check)
```

---

## 🚨 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Install missing packages
pip install flask numpy matplotlib pandas

# Or use startup script
python start_qbes.py
```

#### Web interface won't start
```bash
# Check if port 5000 is available
# Try different port:
cd website
python app.py  # Edit app.py to change port if needed
```

#### Simulation fails
- Check configuration file syntax (valid JSON)
- Ensure parameters are in reasonable ranges
- Check file permissions in results directory

#### No plots showing
- Ensure matplotlib is installed: `pip install matplotlib`
- Check if running in headless environment
- Try exporting data and plotting externally

### Getting Help
1. Run `python qbes_cli.py interactive` and type `help`
2. Check the web interface help section
3. Review configuration examples in `configs/` directory
4. Check simulation results for error messages

---

## 📚 Advanced Usage

### Batch Processing
```bash
# Create multiple configurations
python qbes_cli.py create-config photosynthesis -o config1.json
python qbes_cli.py create-config enzyme -o config2.json

# Run batch simulations
for config in config*.json; do
    python qbes_cli.py run "$config"
done
```

### Parameter Sweeps
```python
# Python script for parameter sweeps
import json
import subprocess

temperatures = [250, 275, 300, 325, 350]
for temp in temperatures:
    # Load base config
    with open('configs/basic_config.json', 'r') as f:
        config = json.load(f)
    
    # Modify temperature
    config['system']['temperature'] = temp
    
    # Save modified config
    config_file = f'temp_sweep_{temp}K.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run simulation
    subprocess.run(['python', 'qbes_cli.py', 'run', config_file])
```

### Integration with Other Tools
```python
# Load QBES results in Python
import json
import pandas as pd

# Load simulation results
with open('simulation_results/sim_20250102_143022/simulation_results.json', 'r') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame({
    'time': results['time_evolution']['time'],
    'coherence': results['time_evolution']['coherence'],
    'purity': results['time_evolution']['purity']
})

# Further analysis...
```

---

## 🎯 Best Practices

### Configuration
- Start with preset configurations
- Make small parameter changes to understand effects
- Keep simulation times reasonable (1-100 ps)
- Use appropriate noise models for your system

### Analysis
- Always check energy conservation
- Compare results with literature values
- Run multiple simulations with different random seeds
- Export data for external analysis tools

### Performance
- Shorter time steps = more accurate but slower
- Longer simulations = more data but slower
- Use appropriate total time for your system
- Monitor memory usage for very long simulations

### Documentation
- Keep notes on parameter choices
- Document the biological system being modeled
- Save configuration files with descriptive names
- Export results with clear naming conventions

---

## 📖 Example Workflows

### Workflow 1: Basic Quantum Biology Study
1. `python start_qbes.py` → Choose web interface
2. Select "Photosynthetic Complex" preset
3. Adjust temperature to room temperature (300K)
4. Run simulation
5. Export results as CSV
6. Create plots in Excel or Python

### Workflow 2: Temperature Dependence Study
1. `python qbes_cli.py create-config enzyme`
2. Edit config file for different temperatures
3. Run simulations at 280K, 300K, 320K
4. Compare coherence lifetimes
5. Plot temperature vs. lifetime

### Workflow 3: Noise Model Comparison
1. Use web interface
2. Run same system with different noise models
3. Compare decoherence rates
4. Export all results
5. Create comparative analysis

---

This guide covers all aspects of using QBES effectively. Start with the web interface for ease of use, then move to CLI for advanced workflows and batch processing.