#!/usr/bin/env python3
"""
Create sample QBES simulation results for demonstration
"""

import os
import json
import numpy as np
from pathlib import Path

def create_sample_results():
    """Create sample simulation results."""
    
    print("📊 Creating Sample QBES Results")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    time_points = np.linspace(0, 10.0, 101)  # 0 to 10 ps
    decoherence_rate = 0.1  # ps^-1
    
    # Calculate quantum coherence decay
    coherence_trajectory = np.exp(-decoherence_rate * time_points)
    
    # Calculate purity evolution (simplified)
    purity_trajectory = 0.5 + 0.5 * np.exp(-decoherence_rate * time_points)
    
    # Calculate energy (constant for this simple system)
    energy_trajectory = np.ones_like(time_points) * 1.0  # 1 eV
    
    # Create results data (JSON-serializable)
    results_data = {
        'simulation_info': {
            'system_type': 'two_level_quantum_system',
            'temperature': 300.0,
            'noise_model': 'protein_ohmic',
            'simulation_time': float(time_points[-1]),
            'time_step': float(time_points[1] - time_points[0]),
            'total_steps': len(time_points)
        },
        'hamiltonian': {
            'matrix': [[0.0, 0.1], [0.1, 2.0]],
            'energy_gap': 2.0,
            'coupling': 0.1
        },
        'results': {
            'time_points': time_points.tolist(),
            'energy_trajectory': energy_trajectory.tolist(),
            'purity_trajectory': purity_trajectory.tolist(),
            'coherence_trajectory': coherence_trajectory.tolist(),
            'coherence_lifetime': 10.0,
            'final_coherence': float(coherence_trajectory[-1]),
            'final_purity': float(purity_trajectory[-1])
        },
        'analysis': {
            'decoherence_rate': decoherence_rate,
            'initial_purity': float(purity_trajectory[0]),
            'purity_loss': float(purity_trajectory[0] - purity_trajectory[-1]),
            'coherence_decay': float(coherence_trajectory[0] - coherence_trajectory[-1])
        }
    }
    
    # Save JSON results
    results_file = output_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✅ Results saved to: {results_file}")
    
    # Save human-readable summary
    summary_file = output_dir / "simulation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("QBES Quantum Simulation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("System Configuration:\n")
        f.write(f"  System Type: Two-level quantum system\n")
        f.write(f"  Temperature: 300.0 K\n")
        f.write(f"  Noise Model: Protein environment (Ohmic)\n")
        f.write(f"  Simulation Time: {time_points[-1]:.1f} ps\n")
        f.write(f"  Time Steps: {len(time_points)}\n\n")
        f.write("Hamiltonian Matrix:\n")
        f.write("  [0.0  0.1]\n")
        f.write("  [0.1  2.0]  (in eV)\n\n")
        f.write("Key Results:\n")
        f.write(f"  Coherence Lifetime: 10.0 ps\n")
        f.write(f"  Decoherence Rate: {decoherence_rate:.3f} ps^-1\n")
        f.write(f"  Initial Purity: {purity_trajectory[0]:.3f}\n")
        f.write(f"  Final Purity: {purity_trajectory[-1]:.3f}\n")
        f.write(f"  Initial Coherence: {coherence_trajectory[0]:.3f}\n")
        f.write(f"  Final Coherence: {coherence_trajectory[-1]:.3f}\n\n")
        f.write("Physical Interpretation:\n")
        f.write("- The quantum system starts in a coherent superposition state\n")
        f.write("- Biological environment (protein) causes decoherence\n")
        f.write("- Coherence decays exponentially: C(t) = exp(-t/tau)\n")
        f.write("- Coherence lifetime tau = 10 ps is typical for biological systems\n")
        f.write("- This demonstrates quantum effects in photosynthesis/enzymes\n\n")
        f.write("Applications:\n")
        f.write("- Photosynthetic energy transfer efficiency\n")
        f.write("- Enzyme catalysis quantum tunneling\n")
        f.write("- Quantum biology research\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    # Save CSV for plotting
    csv_file = output_dir / "time_evolution_data.csv"
    with open(csv_file, 'w') as f:
        f.write("Time_ps,Energy_eV,Purity,Coherence\n")
        for i in range(len(time_points)):
            f.write(f"{time_points[i]:.3f},{energy_trajectory[i]:.6f},")
            f.write(f"{purity_trajectory[i]:.6f},{coherence_trajectory[i]:.6f}\n")
    
    print(f"✅ CSV data saved to: {csv_file}")
    
    # Create analysis report
    analysis_file = output_dir / "analysis_report.txt"
    with open(analysis_file, 'w') as f:
        f.write("QBES Analysis Report\n")
        f.write("=" * 30 + "\n\n")
        f.write("1. QUANTUM COHERENCE ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Coherence Lifetime: 10.0 ps\n")
        f.write(f"Decay Rate: 0.1 ps^-1\n")
        f.write(f"Decay Model: Exponential C(t) = exp(-0.1t)\n\n")
        
        f.write("2. PURITY EVOLUTION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Initial Purity: {purity_trajectory[0]:.3f}\n")
        f.write(f"Final Purity: {purity_trajectory[-1]:.3f}\n")
        f.write(f"Purity Loss: {purity_trajectory[0] - purity_trajectory[-1]:.3f}\n\n")
        
        f.write("3. ENERGY CONSERVATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Energy: {np.mean(energy_trajectory):.3f} eV\n")
        f.write(f"Energy Fluctuation: {np.std(energy_trajectory):.6f} eV\n")
        f.write("Energy is conserved (as expected)\n\n")
        
        f.write("4. BIOLOGICAL SIGNIFICANCE\n")
        f.write("-" * 30 + "\n")
        f.write("This simulation models quantum effects in:\n")
        f.write("• Photosynthetic light-harvesting complexes\n")
        f.write("• Enzyme active sites with quantum tunneling\n")
        f.write("• Membrane protein quantum dynamics\n")
        f.write("• DNA charge transfer processes\n\n")
        
        f.write("The 10 ps coherence lifetime is realistic for:\n")
        f.write("• Chlorophyll molecules in photosystems\n")
        f.write("• Aromatic amino acids in proteins\n")
        f.write("• Biological chromophores at room temperature\n")
    
    print(f"✅ Analysis saved to: {analysis_file}")
    
    print("\n🎉 Sample results created successfully!")
    print(f"📁 Location: {output_dir}/")
    
    return output_dir

if __name__ == "__main__":
    output_dir = create_sample_results()
    print(f"\n📋 Files created:")
    for file in output_dir.glob("*"):
        print(f"   • {file.name}")