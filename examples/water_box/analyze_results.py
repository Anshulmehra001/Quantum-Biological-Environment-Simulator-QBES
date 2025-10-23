#!/usr/bin/env python3
"""
Water Box Example - Results Analysis Script

This script analyzes the results from the water box QBES simulation,
generates plots, and compares with expected values.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os

# Add QBES to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from qbes.analysis import ResultsAnalyzer
    from qbes.visualization import PlotGenerator
except ImportError:
    print("Warning: QBES analysis modules not available. Using basic analysis.")
    ResultsAnalyzer = None
    PlotGenerator = None

def load_simulation_results():
    """Load simulation results from the output directory."""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ Results directory not found. Please run the simulation first:")
        print("   qbes run config.yaml")
        return None
    
    # Load main results file
    results_file = results_dir / "simulation_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print("✅ Loaded simulation results")
        return results
    else:
        print("❌ Results file not found. Simulation may not have completed successfully.")
        return None

def load_time_series_data():
    """Load time-series data for detailed analysis."""
    results_dir = Path("results")
    
    # Load time evolution data
    time_file = results_dir / "time_evolution_data.csv"
    if time_file.exists():
        time_data = pd.read_csv(time_file)
        print("✅ Loaded time-series data")
        return time_data
    else:
        print("⚠️  Time-series data not found. Using basic analysis only.")
        return None

def load_expected_results():
    """Load expected results for comparison."""
    expected_file = Path("expected_results.json")
    if expected_file.exists():
        with open(expected_file, 'r') as f:
            expected = json.load(f)
        print("✅ Loaded expected results")
        return expected
    else:
        print("⚠️  Expected results file not found. Skipping validation.")
        return None

def analyze_coherence(time_data):
    """Analyze quantum coherence evolution."""
    print("\n" + "="*60)
    print(" QUANTUM COHERENCE ANALYSIS")
    print("="*60)
    
    if time_data is None or 'coherence' not in time_data.columns:
        print("❌ Coherence data not available")
        return {}
    
    time = time_data['time'].values
    coherence = time_data['coherence'].values
    
    # Calculate coherence lifetime (1/e decay time)
    initial_coherence = coherence[0]
    target_coherence = initial_coherence / np.e
    
    # Find when coherence drops below 1/e
    decay_indices = np.where(coherence <= target_coherence)[0]
    if len(decay_indices) > 0:
        coherence_lifetime = time[decay_indices[0]]
    else:
        coherence_lifetime = time[-1]  # Didn't decay enough
    
    # Calculate decoherence rate
    decoherence_rate = 1.0 / coherence_lifetime if coherence_lifetime > 0 else 0
    
    # Final purity (assuming coherence relates to purity)
    final_purity = coherence[-1]
    
    results = {
        'coherence_lifetime': coherence_lifetime,
        'decoherence_rate': decoherence_rate,
        'initial_coherence': initial_coherence,
        'final_coherence': coherence[-1],
        'final_purity': final_purity
    }
    
    print(f"Coherence Lifetime:     {coherence_lifetime*1e15:.1f} fs")
    print(f"Decoherence Rate:       {decoherence_rate:.2e} Hz")
    print(f"Initial Coherence:      {initial_coherence:.3f}")
    print(f"Final Coherence:        {coherence[-1]:.3f}")
    print(f"Final Purity:           {final_purity:.3f}")
    
    return results

def analyze_energy_conservation(time_data):
    """Analyze energy conservation during simulation."""
    print("\n" + "="*60)
    print(" ENERGY CONSERVATION ANALYSIS")
    print("="*60)
    
    if time_data is None or 'energy' not in time_data.columns:
        print("❌ Energy data not available")
        return {}
    
    energy = time_data['energy'].values
    initial_energy = energy[0]
    final_energy = energy[-1]
    
    # Calculate energy conservation error
    energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
    
    # Calculate energy drift
    time = time_data['time'].values
    total_time = time[-1] - time[0]
    energy_drift = (final_energy - initial_energy) / total_time
    
    # Energy statistics
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    
    results = {
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_conservation_error': energy_error,
        'energy_drift': energy_drift,
        'mean_energy': mean_energy,
        'energy_fluctuation': std_energy
    }
    
    print(f"Initial Energy:         {initial_energy:.2f} kcal/mol")
    print(f"Final Energy:           {final_energy:.2f} kcal/mol")
    print(f"Energy Change:          {final_energy - initial_energy:.3f} kcal/mol")
    print(f"Conservation Error:     {energy_error*100:.3f}%")
    print(f"Energy Drift:           {energy_drift:.2e} kcal/mol/s")
    print(f"Energy Fluctuation:     {std_energy:.3f} kcal/mol")
    
    # Check if energy is well conserved
    if energy_error < 0.01:
        print("✅ Excellent energy conservation")
    elif energy_error < 0.05:
        print("✅ Good energy conservation")
    else:
        print("⚠️  Poor energy conservation - check simulation parameters")
    
    return results

def analyze_population_dynamics(time_data):
    """Analyze quantum state population dynamics."""
    print("\n" + "="*60)
    print(" POPULATION DYNAMICS ANALYSIS")
    print("="*60)
    
    if time_data is None:
        print("❌ Population data not available")
        return {}
    
    # Look for population columns
    pop_columns = [col for col in time_data.columns if col.startswith('population_')]
    
    if not pop_columns:
        print("❌ No population data found")
        return {}
    
    results = {}
    
    for col in pop_columns:
        state_num = col.split('_')[1]
        population = time_data[col].values
        
        initial_pop = population[0]
        final_pop = population[-1]
        max_pop = np.max(population)
        min_pop = np.min(population)
        
        results[f'state_{state_num}'] = {
            'initial': initial_pop,
            'final': final_pop,
            'maximum': max_pop,
            'minimum': min_pop
        }
        
        print(f"State {state_num}:")
        print(f"  Initial Population:   {initial_pop:.3f}")
        print(f"  Final Population:     {final_pop:.3f}")
        print(f"  Population Range:     {min_pop:.3f} - {max_pop:.3f}")
    
    return results

def compare_with_expected(calculated_results, expected_results):
    """Compare calculated results with expected values."""
    print("\n" + "="*60)
    print(" VALIDATION AGAINST EXPECTED RESULTS")
    print("="*60)
    
    if expected_results is None:
        print("❌ No expected results available for comparison")
        return
    
    validation_passed = True
    
    # Check coherence results
    if 'coherence_lifetime' in calculated_results:
        calc_lifetime = calculated_results['coherence_lifetime']
        exp_lifetime = expected_results['quantum_coherence']['coherence_lifetime']['value']
        tolerance = expected_results['quantum_coherence']['coherence_lifetime']['tolerance']
        
        diff = abs(calc_lifetime - exp_lifetime)
        if diff <= tolerance:
            print(f"✅ Coherence Lifetime: {calc_lifetime*1e15:.1f} fs (expected: {exp_lifetime*1e15:.1f} ± {tolerance*1e15:.1f} fs)")
        else:
            print(f"❌ Coherence Lifetime: {calc_lifetime*1e15:.1f} fs (expected: {exp_lifetime*1e15:.1f} ± {tolerance*1e15:.1f} fs)")
            validation_passed = False
    
    # Check energy conservation
    if 'energy_conservation_error' in calculated_results:
        calc_error = calculated_results['energy_conservation_error']
        exp_error = expected_results['energy_analysis']['energy_conservation_error']['value']
        tolerance = expected_results['energy_analysis']['energy_conservation_error']['tolerance']
        
        if calc_error <= tolerance:
            print(f"✅ Energy Conservation: {calc_error*100:.3f}% error (expected: < {tolerance*100:.1f}%)")
        else:
            print(f"❌ Energy Conservation: {calc_error*100:.3f}% error (expected: < {tolerance*100:.1f}%)")
            validation_passed = False
    
    # Check final purity
    if 'final_purity' in calculated_results:
        calc_purity = calculated_results['final_purity']
        exp_purity = expected_results['quantum_coherence']['final_purity']['value']
        tolerance = expected_results['quantum_coherence']['final_purity']['tolerance']
        
        diff = abs(calc_purity - exp_purity)
        if diff <= tolerance:
            print(f"✅ Final Purity: {calc_purity:.3f} (expected: {exp_purity:.3f} ± {tolerance:.3f})")
        else:
            print(f"❌ Final Purity: {calc_purity:.3f} (expected: {exp_purity:.3f} ± {tolerance:.3f})")
            validation_passed = False
    
    if validation_passed:
        print("\n🎉 All validation checks passed!")
    else:
        print("\n⚠️  Some validation checks failed. Check simulation parameters.")

def generate_plots(time_data, analysis_results):
    """Generate analysis plots."""
    print("\n" + "="*60)
    print(" GENERATING PLOTS")
    print("="*60)
    
    if time_data is None:
        print("❌ No time-series data available for plotting")
        return
    
    # Create plots directory
    plots_dir = Path("results/plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Set up matplotlib for publication-quality plots
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    
    # Plot 1: Coherence Evolution
    if 'coherence' in time_data.columns:
        plt.figure(figsize=(10, 6))
        time_fs = time_data['time'] * 1e15  # Convert to femtoseconds
        plt.plot(time_fs, time_data['coherence'], 'b-', linewidth=2, label='Quantum Coherence')
        
        # Add exponential fit if possible
        if 'coherence_lifetime' in analysis_results:
            lifetime_fs = analysis_results['coherence_lifetime'] * 1e15
            exp_fit = np.exp(-time_fs / lifetime_fs)
            plt.plot(time_fs, exp_fit, 'r--', linewidth=2, alpha=0.7, 
                    label=f'Exponential fit (τ = {lifetime_fs:.1f} fs)')
        
        plt.xlabel('Time (fs)')
        plt.ylabel('Quantum Coherence')
        plt.title('Quantum Coherence Evolution in Water Box')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'coherence_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated coherence_evolution.png")
    
    # Plot 2: Population Dynamics
    pop_columns = [col for col in time_data.columns if col.startswith('population_')]
    if pop_columns:
        plt.figure(figsize=(10, 6))
        time_fs = time_data['time'] * 1e15
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, col in enumerate(pop_columns):
            state_num = col.split('_')[1]
            color = colors[i % len(colors)]
            plt.plot(time_fs, time_data[col], color=color, linewidth=2, 
                    label=f'State {state_num}')
        
        plt.xlabel('Time (fs)')
        plt.ylabel('Population')
        plt.title('Quantum State Population Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'population_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated population_dynamics.png")
    
    # Plot 3: Energy Conservation
    if 'energy' in time_data.columns:
        plt.figure(figsize=(10, 6))
        time_fs = time_data['time'] * 1e15
        energy = time_data['energy']
        
        plt.plot(time_fs, energy, 'g-', linewidth=2, label='Total Energy')
        plt.axhline(y=energy.iloc[0], color='r', linestyle='--', alpha=0.7, 
                   label='Initial Energy')
        
        plt.xlabel('Time (fs)')
        plt.ylabel('Energy (kcal/mol)')
        plt.title('Energy Conservation Check')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'energy_conservation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated energy_conservation.png")
    
    # Plot 4: Summary Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    time_fs = time_data['time'] * 1e15
    
    # Coherence (top-left)
    if 'coherence' in time_data.columns:
        axes[0, 0].plot(time_fs, time_data['coherence'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (fs)')
        axes[0, 0].set_ylabel('Coherence')
        axes[0, 0].set_title('Quantum Coherence')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Energy (top-right)
    if 'energy' in time_data.columns:
        axes[0, 1].plot(time_fs, time_data['energy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time (fs)')
        axes[0, 1].set_ylabel('Energy (kcal/mol)')
        axes[0, 1].set_title('Total Energy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Populations (bottom-left)
    if pop_columns:
        for i, col in enumerate(pop_columns[:3]):  # Show first 3 states
            state_num = col.split('_')[1]
            color = colors[i % len(colors)]
            axes[1, 0].plot(time_fs, time_data[col], color=color, linewidth=2, 
                           label=f'State {state_num}')
        axes[1, 0].set_xlabel('Time (fs)')
        axes[1, 0].set_ylabel('Population')
        axes[1, 0].set_title('State Populations')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary text (bottom-right)
    axes[1, 1].axis('off')
    summary_text = "SIMULATION SUMMARY\n\n"
    if 'coherence_lifetime' in analysis_results:
        lifetime_fs = analysis_results['coherence_lifetime'] * 1e15
        summary_text += f"Coherence Lifetime: {lifetime_fs:.1f} fs\n"
    if 'decoherence_rate' in analysis_results:
        rate = analysis_results['decoherence_rate']
        summary_text += f"Decoherence Rate: {rate:.2e} Hz\n"
    if 'energy_conservation_error' in analysis_results:
        error = analysis_results['energy_conservation_error'] * 100
        summary_text += f"Energy Error: {error:.3f}%\n"
    if 'final_purity' in analysis_results:
        purity = analysis_results['final_purity']
        summary_text += f"Final Purity: {purity:.3f}\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated analysis_dashboard.png")

def main():
    """Main analysis function."""
    print("Water Box Example - Results Analysis")
    print("=" * 50)
    
    # Load all data
    sim_results = load_simulation_results()
    time_data = load_time_series_data()
    expected_results = load_expected_results()
    
    if sim_results is None and time_data is None:
        print("\n❌ No simulation data found. Please run the simulation first:")
        print("   qbes run config.yaml")
        return
    
    # Perform analyses
    analysis_results = {}
    
    # Analyze coherence
    coherence_results = analyze_coherence(time_data)
    analysis_results.update(coherence_results)
    
    # Analyze energy conservation
    energy_results = analyze_energy_conservation(time_data)
    analysis_results.update(energy_results)
    
    # Analyze population dynamics
    population_results = analyze_population_dynamics(time_data)
    analysis_results.update(population_results)
    
    # Compare with expected results
    compare_with_expected(analysis_results, expected_results)
    
    # Generate plots
    generate_plots(time_data, analysis_results)
    
    # Save analysis results
    output_file = Path("results/detailed_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"\n✅ Detailed analysis saved to {output_file}")
    
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE")
    print("="*60)
    print("Check the results/plots/ directory for generated figures.")
    print("See results/detailed_analysis.json for numerical results.")

if __name__ == "__main__":
    main()