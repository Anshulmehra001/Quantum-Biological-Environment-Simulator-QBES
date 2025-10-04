# QBES Tutorial: Step-by-Step Examples

This tutorial provides hands-on examples to help you get started with the Quantum Biological Environment Simulator (QBES). We'll walk through several realistic scenarios from simple to complex.

## Table of Contents

- [Tutorial 1: Simple Two-Level System](#tutorial-1-simple-two-level-system)
- [Tutorial 2: Photosynthetic Light-Harvesting Complex](#tutorial-2-photosynthetic-light-harvesting-complex)
- [Tutorial 3: Enzyme Active Site Dynamics](#tutorial-3-enzyme-active-site-dynamics)
- [Tutorial 4: Custom Analysis and Visualization](#tutorial-4-custom-analysis-and-visualization)
- [Tutorial 5: Batch Processing and Parameter Studies](#tutorial-5-batch-processing-and-parameter-studies)

## Prerequisites

Before starting, ensure you have:
- QBES installed and verified (see [Installation Guide](installation.md))
- Basic understanding of quantum mechanics
- Familiarity with molecular structures (PDB files)

## Tutorial 1: Simple Two-Level System

Let's start with the simplest possible quantum system to understand QBES basics.

### Step 1: Create Configuration

```bash
# Generate a basic configuration
qbes generate-config tutorial1.yaml --template default
```

Edit the configuration file:

```yaml
# tutorial1.yaml - Simple two-level system
system:
  pdb_file: null  # We'll use a built-in model system
  model_system: "two_level"
  force_field: null
  temperature: 300.0

simulation:
  simulation_time: 1.0e-12    # 1 picosecond
  time_step: 1.0e-15          # 1 femtosecond
  integration_method: "runge_kutta"

quantum_subsystem:
  model_type: "two_level"
  energy_gap: 1.0             # eV
  coupling_strength: 0.1      # eV

noise_model:
  type: "ohmic"
  coupling_strength: 0.05
  cutoff_frequency: 1.0e13    # Hz
  temperature: 300.0          # K

output:
  directory: "./tutorial1_output"
  save_trajectory: true
  plot_coherence: true
  plot_populations: true
```

### Step 2: Run Simulation

```bash
# Validate configuration
qbes validate tutorial1.yaml

# Run simulation
qbes run tutorial1.yaml --verbose
```

### Step 3: Analyze Results

```bash
# Check simulation status
qbes status ./tutorial1_output

# View results
ls ./tutorial1_output/plots/
```

### Step 4: Python Analysis

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open('tutorial1_output/simulation_results.json', 'r') as f:
    results = json.load(f)

print(f"Coherence lifetime: {results['coherence_lifetime']:.2e} s")
print(f"Decoherence rate: {results['decoherence_rate']:.2e} Hz")

# Load time-series data
coherence_data = pd.read_csv('tutorial1_output/coherence_measures.csv')

# Plot coherence decay
plt.figure(figsize=(10, 6))
plt.plot(coherence_data['time'], coherence_data['coherence'], 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Quantum Coherence')
plt.title('Coherence Decay in Two-Level System')
plt.grid(True, alpha=0.3)
plt.show()
```

### Expected Results

- Exponential coherence decay
- Decoherence time ~ 100 fs
- Population oscillations at energy gap frequency

## Tutorial 2: Photosynthetic Light-Harvesting Complex

Now let's simulate a realistic biological system - a photosynthetic complex.

### Step 1: Prepare System

For this tutorial, we'll use a simplified light-harvesting complex. In practice, you would use a real PDB file.

```bash
# Generate photosystem configuration
qbes generate-config tutorial2.yaml --template photosystem
```

### Step 2: Configure Simulation

Edit `tutorial2.yaml`:

```yaml
# tutorial2.yaml - Photosynthetic complex
system:
  pdb_file: "examples/lhc_complex.pdb"  # Example structure
  force_field: "amber14"
  solvent_model: "tip3p"
  ionic_strength: 0.15

simulation:
  temperature: 77.0           # Low temperature (liquid nitrogen)
  simulation_time: 5.0e-12    # 5 picoseconds
  time_step: 1.0e-15          # 1 femtosecond

quantum_subsystem:
  selection_method: "chromophores"
  custom_selection: "resname CHL BCL"  # Chlorophylls and bacteriochlorophylls
  max_quantum_atoms: 150
  include_coupling: true

noise_model:
  type: "protein_ohmic"
  coupling_strength: 2.0
  cutoff_frequency: 5.0e13
  reorganization_energy: 35.0  # cm^-1
  include_vibrational_modes: true

output:
  directory: "./tutorial2_output"
  save_trajectory: true
  save_checkpoints: true
  checkpoint_interval: 1000
  generate_movies: true
```

### Step 3: Run Extended Simulation

```bash
# Run with monitoring
qbes run tutorial2.yaml --monitor --verbose

# If simulation is long, you can monitor in another terminal
qbes monitor-sim ./tutorial2_output --interval 2.0
```

### Step 4: Advanced Analysis

```python
from qbes.analysis import ResultsAnalyzer
from qbes.visualization import PlotGenerator
import numpy as np

# Load and analyze results
analyzer = ResultsAnalyzer()
results = analyzer.load_results("tutorial2_output")

# Calculate advanced coherence measures
quantum_discord = analyzer.calculate_quantum_discord(results.state_trajectory)
entanglement_measures = analyzer.calculate_entanglement(results.state_trajectory)

# Energy transfer analysis
transfer_efficiency = analyzer.calculate_transfer_efficiency(results)
transfer_pathways = analyzer.identify_transfer_pathways(results)

print(f"Energy transfer efficiency: {transfer_efficiency:.2%}")
print(f"Average quantum discord: {np.mean(quantum_discord):.3f}")

# Generate comprehensive plots
plotter = PlotGenerator()
plotter.plot_energy_transfer_network(results, transfer_pathways)
plotter.plot_coherence_map(results)
plotter.plot_spectral_analysis(results)
plotter.save_all_plots("tutorial2_output/advanced_plots")
```

### Step 5: Compare with Experimental Data

```python
# Load experimental data (example)
experimental_data = pd.read_csv("experimental_coherence_data.csv")

# Compare simulation with experiment
comparison = analyzer.compare_with_experiment(
    simulation_results=results,
    experimental_data=experimental_data,
    metrics=['coherence_lifetime', 'transfer_efficiency']
)

print("Simulation vs Experiment Comparison:")
for metric, values in comparison.items():
    sim_val, exp_val, error = values
    print(f"{metric}: Sim={sim_val:.3f}, Exp={exp_val:.3f}, Error={error:.1%}")
```

### Expected Results

- Long-lived quantum coherence (> 1 ps at 77K)
- Efficient energy transfer (> 90%)
- Multiple coherent pathways
- Temperature-dependent decoherence

## Tutorial 3: Enzyme Active Site Dynamics

Let's explore quantum effects in enzymatic catalysis.

### Step 1: System Setup

```bash
# Generate enzyme configuration
qbes generate-config tutorial3.yaml --template enzyme
```

### Step 2: Configure for Catalysis Study

```yaml
# tutorial3.yaml - Enzyme active site
system:
  pdb_file: "examples/enzyme_active_site.pdb"
  force_field: "amber14"
  solvent_model: "tip3p"
  ionic_strength: 0.15
  include_substrate: true

simulation:
  temperature: 310.0          # Body temperature
  simulation_time: 2.0e-12    # 2 picoseconds
  time_step: 5.0e-16          # 0.5 femtoseconds (smaller for reactions)

quantum_subsystem:
  selection_method: "active_site"
  custom_selection: "resname HIS CYS ASP GLU SER THR and within 5.0 of substrate"
  max_quantum_atoms: 80
  include_substrate: true
  reaction_coordinate: "bond_breaking"

noise_model:
  type: "protein_ohmic"
  coupling_strength: 1.5
  cutoff_frequency: 2.0e13
  reorganization_energy: 50.0  # Higher for enzyme environment
  include_conformational_noise: true

analysis:
  calculate_reaction_rates: true
  track_tunneling_probability: true
  monitor_active_site_dynamics: true

output:
  directory: "./tutorial3_output"
  save_reaction_trajectory: true
  plot_reaction_coordinate: true
  plot_tunneling_analysis: true
```

### Step 3: Run Catalysis Simulation

```bash
# Run enzyme simulation
qbes run tutorial3.yaml --verbose --checkpoint-interval 500
```

### Step 4: Analyze Catalytic Mechanism

```python
from qbes.analysis import EnzymeAnalyzer
import matplotlib.pyplot as plt

# Specialized enzyme analysis
enzyme_analyzer = EnzymeAnalyzer()
results = enzyme_analyzer.load_results("tutorial3_output")

# Calculate reaction rates
classical_rate = enzyme_analyzer.calculate_classical_rate(results)
quantum_rate = enzyme_analyzer.calculate_quantum_rate(results)
tunneling_contribution = enzyme_analyzer.calculate_tunneling_contribution(results)

print(f"Classical reaction rate: {classical_rate:.2e} s^-1")
print(f"Quantum reaction rate: {quantum_rate:.2e} s^-1")
print(f"Tunneling enhancement: {quantum_rate/classical_rate:.1f}x")
print(f"Tunneling contribution: {tunneling_contribution:.1%}")

# Analyze reaction pathway
reaction_pathway = enzyme_analyzer.analyze_reaction_pathway(results)
transition_states = enzyme_analyzer.identify_transition_states(results)

# Plot reaction coordinate
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(results.time, results.reaction_coordinate)
plt.xlabel('Time (s)')
plt.ylabel('Reaction Coordinate')
plt.title('Reaction Progress')

plt.subplot(2, 2, 2)
plt.plot(results.time, results.tunneling_probability)
plt.xlabel('Time (s)')
plt.ylabel('Tunneling Probability')
plt.title('Quantum Tunneling')

plt.subplot(2, 2, 3)
enzyme_analyzer.plot_energy_profile(reaction_pathway)
plt.title('Reaction Energy Profile')

plt.subplot(2, 2, 4)
enzyme_analyzer.plot_active_site_dynamics(results)
plt.title('Active Site Fluctuations')

plt.tight_layout()
plt.savefig('tutorial3_output/enzyme_analysis.png', dpi=300)
plt.show()
```

### Expected Results

- Quantum tunneling enhancement of reaction rate
- Temperature-dependent tunneling probability
- Active site conformational coupling
- Comparison of classical vs quantum pathways

## Tutorial 4: Custom Analysis and Visualization

Learn to create custom analysis tools for your specific research needs.

### Step 1: Custom Analysis Class

```python
# custom_analysis.py
from qbes.analysis import ResultsAnalyzer
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit

class CustomCoherenceAnalyzer(ResultsAnalyzer):
    """Custom analyzer for specialized coherence studies."""
    
    def __init__(self):
        super().__init__()
        self.custom_metrics = {}
    
    def calculate_coherence_spectrum(self, state_trajectory, time_points):
        """Calculate frequency spectrum of coherence oscillations."""
        coherence_values = []
        
        for state in state_trajectory:
            # Calculate off-diagonal coherence
            coherence = np.abs(state[0, 1])  # Assuming 2x2 density matrix
            coherence_values.append(coherence)
        
        coherence_array = np.array(coherence_values)
        
        # Calculate power spectral density
        frequencies, psd = signal.welch(
            coherence_array, 
            fs=1.0/(time_points[1] - time_points[0]),
            nperseg=len(coherence_array)//4
        )
        
        return frequencies, psd
    
    def fit_decoherence_model(self, time_points, coherence_values):
        """Fit exponential decay model to coherence data."""
        
        def exponential_decay(t, A, gamma, phi):
            return A * np.exp(-gamma * t) * np.cos(phi * t)
        
        # Initial parameter guess
        p0 = [1.0, 1e12, 1e13]  # Amplitude, decay rate, frequency
        
        try:
            popt, pcov = curve_fit(exponential_decay, time_points, coherence_values, p0=p0)
            
            # Calculate fit quality
            fitted_values = exponential_decay(time_points, *popt)
            r_squared = 1 - np.sum((coherence_values - fitted_values)**2) / np.sum((coherence_values - np.mean(coherence_values))**2)
            
            return {
                'amplitude': popt[0],
                'decay_rate': popt[1],
                'frequency': popt[2],
                'r_squared': r_squared,
                'parameter_errors': np.sqrt(np.diag(pcov))
            }
        except:
            return None
    
    def calculate_quantum_efficiency(self, results):
        """Calculate quantum efficiency for energy transfer."""
        initial_excitation = results.initial_populations[1]  # Excited state
        final_target = results.final_populations[-1]  # Target state
        
        efficiency = final_target / initial_excitation
        return efficiency
    
    def generate_custom_report(self, results, output_file):
        """Generate comprehensive custom analysis report."""
        
        report = []
        report.append("Custom Coherence Analysis Report")
        report.append("=" * 40)
        report.append("")
        
        # Basic metrics
        coherence_lifetime = self.calculate_coherence_lifetime(results.state_trajectory)
        report.append(f"Coherence Lifetime: {coherence_lifetime:.2e} s")
        
        # Spectral analysis
        frequencies, psd = self.calculate_coherence_spectrum(
            results.state_trajectory, results.time_points
        )
        peak_frequency = frequencies[np.argmax(psd)]
        report.append(f"Peak Coherence Frequency: {peak_frequency:.2e} Hz")
        
        # Decoherence model fitting
        coherence_values = [np.abs(state[0, 1]) for state in results.state_trajectory]
        fit_results = self.fit_decoherence_model(results.time_points, coherence_values)
        
        if fit_results:
            report.append(f"Fitted Decay Rate: {fit_results['decay_rate']:.2e} Hz")
            report.append(f"Fitted Frequency: {fit_results['frequency']:.2e} Hz")
            report.append(f"Fit Quality (R²): {fit_results['r_squared']:.3f}")
        
        # Quantum efficiency
        if hasattr(results, 'initial_populations'):
            efficiency = self.calculate_quantum_efficiency(results)
            report.append(f"Quantum Efficiency: {efficiency:.1%}")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        return report

# Usage example
analyzer = CustomCoherenceAnalyzer()
results = analyzer.load_results("tutorial2_output")
report = analyzer.generate_custom_report(results, "custom_analysis_report.txt")
print('\n'.join(report))
```

### Step 2: Custom Visualization

```python
# custom_plots.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns

class CustomPlotter:
    """Custom plotting tools for QBES results."""
    
    def __init__(self, style='scientific'):
        if style == 'scientific':
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
    
    def plot_coherence_heatmap(self, results, save_path=None):
        """Create heatmap of coherence evolution."""
        
        # Extract coherence matrix elements over time
        n_states = results.state_trajectory[0].shape[0]
        coherence_matrix = np.zeros((len(results.time_points), n_states, n_states))
        
        for i, state in enumerate(results.state_trajectory):
            coherence_matrix[i] = np.abs(state)
        
        # Create subplots for each matrix element
        fig, axes = plt.subplots(n_states, n_states, figsize=(12, 10))
        
        for i in range(n_states):
            for j in range(n_states):
                if n_states > 1:
                    ax = axes[i, j]
                else:
                    ax = axes
                
                im = ax.imshow(
                    coherence_matrix[:, i, j].reshape(-1, 1).T,
                    aspect='auto',
                    extent=[results.time_points[0], results.time_points[-1], 0, 1],
                    cmap='viridis'
                )
                ax.set_title(f'ρ_{i}{j}')
                ax.set_xlabel('Time (s)')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_coherence_animation(self, results, save_path=None):
        """Create animation of quantum state evolution."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initialize plots
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        line2, = ax2.plot([], [], 'r-', linewidth=2)
        
        ax1.set_xlim(0, len(results.time_points))
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Population')
        ax1.set_title('State Populations')
        
        ax2.set_xlim(0, len(results.time_points))
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Coherence')
        ax2.set_title('Quantum Coherence')
        
        def animate(frame):
            # Update population plot
            populations = [np.real(state[i, i]) for state in results.state_trajectory[:frame+1]]
            line1.set_data(range(frame+1), populations)
            
            # Update coherence plot
            coherences = [np.abs(state[0, 1]) for state in results.state_trajectory[:frame+1]]
            line2.set_data(range(frame+1), coherences)
            
            return line1, line2
        
        anim = FuncAnimation(
            fig, animate, frames=len(results.time_points),
            interval=50, blit=True, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.show()
        return anim
    
    def plot_parameter_sensitivity(self, parameter_study_results):
        """Plot sensitivity analysis results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract parameter values and metrics
        params = list(parameter_study_results.keys())
        coherence_lifetimes = [r['coherence_lifetime'] for r in parameter_study_results.values()]
        transfer_efficiencies = [r['transfer_efficiency'] for r in parameter_study_results.values()]
        
        # Plot coherence lifetime vs parameters
        axes[0, 0].plot(params, coherence_lifetimes, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Parameter Value')
        axes[0, 0].set_ylabel('Coherence Lifetime (s)')
        axes[0, 0].set_title('Coherence Lifetime Sensitivity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot transfer efficiency vs parameters
        axes[0, 1].plot(params, transfer_efficiencies, 's-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Parameter Value')
        axes[0, 1].set_ylabel('Transfer Efficiency')
        axes[0, 1].set_title('Transfer Efficiency Sensitivity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Correlation plot
        axes[1, 0].scatter(coherence_lifetimes, transfer_efficiencies, s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Coherence Lifetime (s)')
        axes[1, 0].set_ylabel('Transfer Efficiency')
        axes[1, 0].set_title('Lifetime vs Efficiency Correlation')
        
        # Parameter distribution
        axes[1, 1].hist(params, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Parameter Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Parameter Distribution')
        
        plt.tight_layout()
        plt.show()

# Usage example
plotter = CustomPlotter()
results = analyzer.load_results("tutorial2_output")

# Create custom visualizations
plotter.plot_coherence_heatmap(results, "coherence_heatmap.png")
animation = plotter.create_coherence_animation(results, "coherence_evolution.gif")
```

## Tutorial 5: Batch Processing and Parameter Studies

Learn to run systematic studies across parameter ranges.

### Step 1: Parameter Study Setup

```python
# parameter_study.py
import yaml
import os
import numpy as np
from qbes import ConfigurationManager, SimulationEngine
from qbes.utils.batch_processing import BatchRunner

class ParameterStudy:
    """Systematic parameter study framework."""
    
    def __init__(self, base_config_file):
        self.base_config_file = base_config_file
        self.batch_runner = BatchRunner()
        
        # Load base configuration
        with open(base_config_file, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def temperature_study(self, temperatures, output_base_dir):
        """Study temperature dependence."""
        
        configs = []
        for temp in temperatures:
            # Create modified configuration
            config = self.base_config.copy()
            config['simulation']['temperature'] = temp
            config['output']['directory'] = f"{output_base_dir}/temp_{temp}K"
            
            # Save configuration file
            config_file = f"config_temp_{temp}K.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            configs.append(config_file)
        
        return configs
    
    def coupling_strength_study(self, coupling_values, output_base_dir):
        """Study coupling strength dependence."""
        
        configs = []
        for coupling in coupling_values:
            config = self.base_config.copy()
            config['noise_model']['coupling_strength'] = coupling
            config['output']['directory'] = f"{output_base_dir}/coupling_{coupling}"
            
            config_file = f"config_coupling_{coupling}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            configs.append(config_file)
        
        return configs
    
    def run_parameter_study(self, config_files, parallel=True, n_jobs=4):
        """Run all configurations in the study."""
        
        results = {}
        
        if parallel:
            # Run simulations in parallel
            from joblib import Parallel, delayed
            
            def run_single_sim(config_file):
                engine = SimulationEngine()
                config_manager = ConfigurationManager()
                config = config_manager.load_config(config_file)
                
                engine.initialize_simulation(config)
                result = engine.run_simulation()
                
                return config_file, result
            
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(run_single_sim)(config) for config in config_files
            )
            
            for config_file, result in parallel_results:
                results[config_file] = result
        
        else:
            # Run simulations sequentially
            for config_file in config_files:
                engine = SimulationEngine()
                config_manager = ConfigurationManager()
                config = config_manager.load_config(config_file)
                
                engine.initialize_simulation(config)
                result = engine.run_simulation()
                results[config_file] = result
        
        return results
    
    def analyze_study_results(self, results, study_type):
        """Analyze parameter study results."""
        
        analysis = {}
        
        for config_file, result in results.items():
            # Extract parameter value from filename
            if study_type == 'temperature':
                param_value = float(config_file.split('_')[2].replace('K.yaml', ''))
            elif study_type == 'coupling':
                param_value = float(config_file.split('_')[2].replace('.yaml', ''))
            
            # Calculate key metrics
            coherence_lifetime = self.calculate_coherence_lifetime(result.state_trajectory)
            transfer_efficiency = self.calculate_transfer_efficiency(result)
            
            analysis[param_value] = {
                'coherence_lifetime': coherence_lifetime,
                'transfer_efficiency': transfer_efficiency,
                'final_populations': result.final_populations
            }
        
        return analysis

# Run temperature study
study = ParameterStudy("tutorial2.yaml")

# Define temperature range
temperatures = np.linspace(77, 300, 10)  # 77K to 300K
temp_configs = study.temperature_study(temperatures, "./temperature_study")

# Run all simulations
print("Running temperature study...")
temp_results = study.run_parameter_study(temp_configs, parallel=True, n_jobs=4)

# Analyze results
temp_analysis = study.analyze_study_results(temp_results, 'temperature')

# Plot results
import matplotlib.pyplot as plt

temps = list(temp_analysis.keys())
lifetimes = [temp_analysis[t]['coherence_lifetime'] for t in temps]
efficiencies = [temp_analysis[t]['transfer_efficiency'] for t in temps]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(temps, lifetimes, 'o-', linewidth=2, markersize=8)
plt.xlabel('Temperature (K)')
plt.ylabel('Coherence Lifetime (s)')
plt.title('Temperature Dependence of Coherence')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(temps, efficiencies, 's-', linewidth=2, markersize=8)
plt.xlabel('Temperature (K)')
plt.ylabel('Transfer Efficiency')
plt.title('Temperature Dependence of Efficiency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_study_results.png', dpi=300)
plt.show()
```

### Step 2: Statistical Analysis

```python
# statistical_analysis.py
import numpy as np
from scipy import stats
import pandas as pd

def statistical_analysis_of_study(parameter_values, metric_values, metric_name):
    """Perform statistical analysis of parameter study."""
    
    # Basic statistics
    mean_val = np.mean(metric_values)
    std_val = np.std(metric_values)
    
    # Correlation analysis
    correlation, p_value = stats.pearsonr(parameter_values, metric_values)
    
    # Linear regression
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(parameter_values, metric_values)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'parameter': parameter_values,
        'metric': metric_values
    })
    
    # Statistical summary
    summary = {
        'metric_name': metric_name,
        'mean': mean_val,
        'std': std_val,
        'correlation': correlation,
        'correlation_p_value': p_value,
        'regression_slope': slope,
        'regression_intercept': intercept,
        'r_squared': r_value**2,
        'regression_p_value': p_value_reg,
        'data': results_df
    }
    
    return summary

# Analyze temperature study results
temp_stats = statistical_analysis_of_study(
    list(temp_analysis.keys()),
    [temp_analysis[t]['coherence_lifetime'] for t in temp_analysis.keys()],
    'Coherence Lifetime'
)

print(f"Temperature Study Statistical Analysis:")
print(f"Correlation coefficient: {temp_stats['correlation']:.3f}")
print(f"R-squared: {temp_stats['r_squared']:.3f}")
print(f"Regression slope: {temp_stats['regression_slope']:.2e}")
```

## Summary and Next Steps

Congratulations! You've completed the QBES tutorial series. You should now be able to:

1. **Set up and run basic quantum simulations**
2. **Configure realistic biological systems**
3. **Analyze quantum coherence and energy transfer**
4. **Create custom analysis tools**
5. **Perform systematic parameter studies**

### Next Steps

1. **Explore Advanced Features**:
   - GPU acceleration
   - MPI parallel processing
   - Custom noise models

2. **Apply to Your Research**:
   - Use your own PDB structures
   - Develop system-specific analysis
   - Compare with experimental data

3. **Contribute to QBES**:
   - Report bugs and suggest features
   - Contribute analysis tools
   - Share example systems

4. **Learn More**:
   - Read the [Theory and Methods](theory.md) documentation
   - Study the [API Reference](api_reference.md)
   - Join the community discussions

### Additional Resources

- **Example Systems**: Check the `examples/` directory for more complex systems
- **Validation Studies**: See `benchmarks/` for scientific validation examples
- **Community**: Join our GitHub discussions for questions and collaboration
- **Publications**: List of papers using QBES (to be updated)

Happy simulating! 🚀