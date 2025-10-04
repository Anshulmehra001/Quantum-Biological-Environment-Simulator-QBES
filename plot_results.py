#!/usr/bin/env python3
"""
QBES Results Plotting Script
Simple script to create plots from simulation results
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_simulation_results(results_dir, output_dir=None, show_plots=True):
    """Plot simulation results from a results directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    # Load CSV data
    csv_file = results_path / "time_evolution_data.csv"
    if not csv_file.exists():
        print(f"❌ CSV data file not found: {csv_file}")
        return False
    
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded data: {len(df)} time points")
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'QBES Simulation Results: {results_path.name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Coherence evolution
        axes[0, 0].plot(df['Time_ps'], df['Coherence'], 'b-', linewidth=2, label='Quantum Coherence')
        axes[0, 0].set_xlabel('Time (ps)')
        axes[0, 0].set_ylabel('Coherence')
        axes[0, 0].set_title('Quantum Coherence Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Purity evolution
        axes[0, 1].plot(df['Time_ps'], df['Purity'], 'r-', linewidth=2, label='State Purity')
        axes[0, 1].set_xlabel('Time (ps)')
        axes[0, 1].set_ylabel('Purity')
        axes[0, 1].set_title('Quantum State Purity Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Energy evolution
        axes[1, 0].plot(df['Time_ps'], df['Energy_eV'], 'g-', linewidth=2, label='System Energy')
        axes[1, 0].set_xlabel('Time (ps)')
        axes[1, 0].set_ylabel('Energy (eV)')
        axes[1, 0].set_title('System Energy Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Combined coherence and purity
        ax4 = axes[1, 1]
        ax4.plot(df['Time_ps'], df['Coherence'], 'b-', linewidth=2, label='Coherence')
        ax4.plot(df['Time_ps'], df['Purity'], 'r-', linewidth=2, label='Purity')
        ax4.set_xlabel('Time (ps)')
        ax4.set_ylabel('Quantum Properties')
        ax4.set_title('Combined Quantum Properties')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plots
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / f"{results_path.name}_plots.png"
        else:
            plot_file = results_path / "simulation_plots.png"
        
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"💾 Plots saved to: {plot_file}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        return False

def find_latest_results():
    """Find the most recent simulation results directory"""
    results_dir = Path("simulation_results")
    if not results_dir.exists():
        return None
    
    sim_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('sim_')]
    if not sim_dirs:
        return None
    
    # Return most recent based on name (timestamp)
    return max(sim_dirs, key=lambda x: x.name)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Plot QBES simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Plot latest results
  %(prog)s simulation_results/sim_20250102_143022  # Plot specific results
  %(prog)s --no-show --output plots/         # Save plots without showing
        """
    )
    
    parser.add_argument('results_dir', nargs='?', help='Results directory path (default: latest)')
    parser.add_argument('--output', '-o', help='Output directory for plot files')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display plots (just save)')
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = find_latest_results()
        if not results_dir:
            print("❌ No simulation results found")
            print("💡 Run a simulation first: python qbes_cli.py run configs/photosynthesis_config.json")
            return
        print(f"📊 Using latest results: {results_dir}")
    
    # Create plots
    success = plot_simulation_results(
        results_dir, 
        output_dir=args.output,
        show_plots=not args.no_show
    )
    
    if success:
        print("✅ Plotting completed successfully!")
    else:
        print("❌ Plotting failed")

if __name__ == "__main__":
    main()