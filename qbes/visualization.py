"""
Visualization and plotting tools for QBES results.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

from .core.interfaces import VisualizationInterface
from .core.data_models import DensityMatrix, SimulationResults, CoherenceMetrics, StatisticalSummary


class VisualizationEngine(VisualizationInterface):
    """
    Provides visualization and plotting capabilities for simulation results.
    
    This class creates publication-ready plots and animations of quantum
    state evolution and analysis results.
    """
    
    def __init__(self):
        """Initialize the visualization engine."""
        self.plot_style = "scientific"
        self.figure_cache = {}
        self._setup_plot_style()
    
    def plot_state_evolution(self, state_trajectory: List[DensityMatrix], 
                           output_path: str) -> bool:
        """Create plots showing quantum state evolution over time."""
        try:
            if not state_trajectory:
                raise ValueError("State trajectory is empty")
            
            # Extract time points and population data
            times = [state.time for state in state_trajectory]
            n_states = len(state_trajectory[0].basis_labels)
            populations = np.zeros((len(times), n_states))
            
            for i, state in enumerate(state_trajectory):
                # Extract diagonal elements (populations)
                populations[i, :] = np.real(np.diag(state.matrix))
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Population dynamics
            for j in range(n_states):
                ax1.plot(times, populations[:, j], 
                        label=f'State {state_trajectory[0].basis_labels[j]}',
                        linewidth=2)
            
            ax1.set_xlabel('Time (ps)')
            ax1.set_ylabel('Population')
            ax1.set_title('Quantum State Population Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Plot 2: Coherence matrix evolution (heatmap of final state)
            final_state = state_trajectory[-1]
            coherence_matrix = np.abs(final_state.matrix)
            
            im = ax2.imshow(coherence_matrix, cmap='viridis', aspect='equal')
            ax2.set_title('Final State Coherence Matrix (|ρ_ij|)')
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Basis State')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('|ρ_ij|')
            
            # Set tick labels
            ax2.set_xticks(range(n_states))
            ax2.set_yticks(range(n_states))
            ax2.set_xticklabels(state_trajectory[0].basis_labels)
            ax2.set_yticklabels(state_trajectory[0].basis_labels)
            
            plt.tight_layout()
            
            # Save the plot
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error plotting state evolution: {e}")
            return False
    
    def plot_coherence_measures(self, coherence_data: Dict[str, List[float]], 
                               output_path: str) -> bool:
        """Plot various coherence measures vs time."""
        try:
            if not coherence_data:
                raise ValueError("Coherence data is empty")
            
            # Determine number of subplots needed
            n_measures = len(coherence_data)
            if n_measures == 0:
                return False
            
            # Create figure with subplots
            fig, axes = plt.subplots(n_measures, 1, figsize=(10, 3*n_measures))
            if n_measures == 1:
                axes = [axes]
            
            # Plot each coherence measure
            for i, (measure_name, values) in enumerate(coherence_data.items()):
                # Create time points for this specific measure
                n_points = len(values)
                times = np.linspace(0, 1, n_points)  # Default time range
                ax = axes[i]
                
                # Plot the data
                ax.plot(times, values, linewidth=2, color=f'C{i}')
                ax.set_ylabel(self._format_measure_name(measure_name))
                ax.set_title(f'{self._format_measure_name(measure_name)} vs Time')
                ax.grid(True, alpha=0.3)
                
                # Set appropriate y-limits based on measure type
                if 'purity' in measure_name.lower():
                    ax.set_ylim(0, 1)
                elif 'entropy' in measure_name.lower():
                    ax.set_ylim(0, None)
                
                # Add exponential fit for coherence lifetime if applicable
                if 'coherence' in measure_name.lower() and len(values) > 10:
                    self._add_exponential_fit(ax, times, values)
            
            # Set x-label only on bottom plot
            axes[-1].set_xlabel('Time (ps)')
            
            plt.tight_layout()
            
            # Save the plot
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error plotting coherence measures: {e}")
            return False
    
    def plot_energy_landscape(self, energy_trajectory: List[float], 
                             output_path: str) -> bool:
        """Plot energy evolution during simulation."""
        try:
            if not energy_trajectory:
                raise ValueError("Energy trajectory is empty")
            
            times = np.linspace(0, 1, len(energy_trajectory))  # Default time range
            energies = np.array(energy_trajectory)
            
            # Create figure with multiple panels
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
            
            # Panel 1: Energy vs time
            ax1.plot(times, energies, linewidth=2, color='darkblue')
            ax1.set_ylabel('Energy (a.u.)')
            ax1.set_title('Energy Evolution During Simulation')
            ax1.grid(True, alpha=0.3)
            
            # Add running average
            if len(energies) > 10:
                window_size = max(5, len(energies) // 20)
                running_avg = self._calculate_running_average(energies, window_size)
                ax1.plot(times, running_avg, '--', color='red', 
                        label=f'Running Average (window={window_size})', linewidth=2)
                ax1.legend()
            
            # Panel 2: Energy distribution (histogram)
            ax2.hist(energies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Energy (a.u.)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Energy Distribution')
            ax2.axvline(np.mean(energies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(energies):.4f}')
            ax2.legend()
            
            # Panel 3: Energy fluctuations (deviations from mean)
            mean_energy = np.mean(energies)
            fluctuations = energies - mean_energy
            ax3.plot(times, fluctuations, linewidth=1, color='green', alpha=0.7)
            ax3.set_xlabel('Time (ps)')
            ax3.set_ylabel('Energy Fluctuation (a.u.)')
            ax3.set_title('Energy Fluctuations (E - <E>)')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
            
            # Add statistics text
            std_energy = np.std(energies)
            ax3.text(0.02, 0.98, f'σ = {std_energy:.4f}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error plotting energy landscape: {e}")
            return False
    
    def create_publication_figure(self, results: SimulationResults, 
                                 figure_type: str, output_path: str) -> bool:
        """Create publication-ready figures with proper formatting."""
        try:
            if figure_type == "overview":
                return self._create_overview_figure(results, output_path)
            elif figure_type == "coherence_analysis":
                return self._create_coherence_figure(results, output_path)
            elif figure_type == "energy_analysis":
                return self._create_energy_figure(results, output_path)
            else:
                raise ValueError(f"Unknown figure type: {figure_type}")
        except Exception as e:
            print(f"Error creating publication figure: {e}")
            return False
    
    def generate_animation(self, state_trajectory: List[DensityMatrix], 
                          output_path: str, fps: int = 10) -> bool:
        """Generate animation of quantum state evolution."""
        try:
            # For now, create a series of static frames showing state evolution
            # Full animation would require additional dependencies like matplotlib.animation
            
            n_frames = min(len(state_trajectory), 20)  # Limit frames for performance
            frame_indices = np.linspace(0, len(state_trajectory)-1, n_frames, dtype=int)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for i, frame_idx in enumerate(frame_indices):
                ax.clear()
                state = state_trajectory[frame_idx]
                
                # Plot density matrix as heatmap
                im = ax.imshow(np.abs(state.matrix), cmap='viridis', 
                              vmin=0, vmax=1, aspect='equal')
                ax.set_title(f'Density Matrix Evolution - Time: {state.time:.3f} ps')
                ax.set_xlabel('Basis State')
                ax.set_ylabel('Basis State')
                
                # Save frame
                frame_path = output_path.replace('.png', f'_frame_{i:03d}.png')
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"Error generating animation: {e}")
            return False
    
    def _create_overview_figure(self, results: SimulationResults, output_path: str) -> bool:
        """Create comprehensive overview figure."""
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Population evolution
        ax1 = fig.add_subplot(gs[0, :2])
        times = [state.time for state in results.state_trajectory]
        n_states = len(results.state_trajectory[0].basis_labels)
        populations = np.zeros((len(times), n_states))
        
        for i, state in enumerate(results.state_trajectory):
            populations[i, :] = np.real(np.diag(state.matrix))
        
        for j in range(n_states):
            ax1.plot(times, populations[:, j], 
                    label=f'State {results.state_trajectory[0].basis_labels[j]}')
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Population')
        ax1.set_title('Population Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Energy evolution
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(times, results.energy_trajectory)
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Energy (a.u.)')
        ax2.set_title('Energy Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Coherence measures
        ax3 = fig.add_subplot(gs[2, :2])
        for measure_name, values in results.coherence_measures.items():
            if len(values) == len(times):
                ax3.plot(times, values, label=self._format_measure_name(measure_name))
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Coherence Measures')
        ax3.set_title('Coherence Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Final state density matrix
        ax4 = fig.add_subplot(gs[:, 2])
        final_state = results.state_trajectory[-1]
        im = ax4.imshow(np.abs(final_state.matrix), cmap='viridis', aspect='equal')
        ax4.set_title('Final State |ρ|')
        plt.colorbar(im, ax=ax4)
        
        plt.suptitle('Quantum Biological Simulation Overview', fontsize=16)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    
    def _create_coherence_figure(self, results: SimulationResults, output_path: str) -> bool:
        """Create detailed coherence analysis figure."""
        return self.plot_coherence_measures(results.coherence_measures, output_path)
    
    def _create_energy_figure(self, results: SimulationResults, output_path: str) -> bool:
        """Create detailed energy analysis figure."""
        return self.plot_energy_landscape(results.energy_trajectory, output_path)
    
    def create_multi_panel_figure(self, results: SimulationResults, 
                                 panels: List[str], output_path: str,
                                 figure_title: str = None, 
                                 add_captions: bool = True) -> bool:
        """Create a multi-panel publication figure with automatic layout and captions."""
        try:
            n_panels = len(panels)
            if n_panels == 0:
                return False
            
            # Determine optimal layout
            if n_panels <= 2:
                rows, cols = 1, n_panels
                figsize = (6 * cols, 5)
            elif n_panels <= 4:
                rows, cols = 2, 2
                figsize = (12, 10)
            elif n_panels <= 6:
                rows, cols = 2, 3
                figsize = (18, 10)
            else:
                rows, cols = 3, 3
                figsize = (18, 15)
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if n_panels == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Panel labels for publication
            panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            
            for i, panel_type in enumerate(panels):
                ax = axes[i]
                
                if panel_type == 'state_evolution':
                    self._plot_state_evolution_panel(ax, results.state_trajectory)
                elif panel_type == 'coherence_measures':
                    self._plot_coherence_panel(ax, results.coherence_measures)
                elif panel_type == 'energy_evolution':
                    self._plot_energy_panel(ax, results.energy_trajectory)
                elif panel_type == 'final_state_matrix':
                    self._plot_final_state_panel(ax, results.state_trajectory[-1])
                elif panel_type == 'decoherence_rates':
                    self._plot_decoherence_panel(ax, results.decoherence_rates)
                elif panel_type == 'statistical_summary':
                    self._plot_statistics_panel(ax, results.statistical_summary)
                
                # Add panel label
                if add_captions:
                    ax.text(-0.1, 1.05, f'({panel_labels[i]})', 
                           transform=ax.transAxes, fontsize=16, fontweight='bold')
            
            # Hide unused subplots
            for i in range(n_panels, len(axes)):
                axes[i].set_visible(False)
            
            # Add overall title
            if figure_title:
                fig.suptitle(figure_title, fontsize=18, fontweight='bold', y=0.98)
            
            # Add metadata caption
            if add_captions:
                caption = self._generate_figure_caption(results, panels)
                fig.text(0.02, 0.02, caption, fontsize=10, style='italic', 
                        wrap=True, verticalalignment='bottom')
            
            plt.tight_layout()
            if figure_title or add_captions:
                plt.subplots_adjust(top=0.93, bottom=0.15)
            
            # Save with publication quality
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating multi-panel figure: {e}")
            return False
    
    def _plot_state_evolution_panel(self, ax, state_trajectory: List[DensityMatrix]):
        """Plot state evolution in a single panel."""
        times = [state.time for state in state_trajectory]
        n_states = len(state_trajectory[0].basis_labels)
        populations = np.zeros((len(times), n_states))
        
        for i, state in enumerate(state_trajectory):
            populations[i, :] = np.real(np.diag(state.matrix))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_states))
        for j in range(n_states):
            ax.plot(times, populations[:, j], 
                   label=f'{state_trajectory[0].basis_labels[j]}',
                   linewidth=2, color=colors[j])
        
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Population')
        ax.set_title('Quantum State Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_coherence_panel(self, ax, coherence_measures: Dict[str, List[float]]):
        """Plot coherence measures in a single panel."""
        if not coherence_measures:
            ax.text(0.5, 0.5, 'No coherence data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        # Plot the most important coherence measure
        primary_measure = 'coherence_lifetime' if 'coherence_lifetime' in coherence_measures else list(coherence_measures.keys())[0]
        values = coherence_measures[primary_measure]
        times = np.linspace(0, 1, len(values))
        
        ax.plot(times, values, linewidth=2, color='darkblue')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel(self._format_measure_name(primary_measure))
        ax.set_title('Coherence Evolution')
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_panel(self, ax, energy_trajectory: List[float]):
        """Plot energy evolution in a single panel."""
        times = np.linspace(0, 1, len(energy_trajectory))
        energies = np.array(energy_trajectory)
        
        ax.plot(times, energies, linewidth=2, color='darkred')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Energy (a.u.)')
        ax.set_title('Energy Evolution')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_energy = np.mean(energies)
        ax.axhline(mean_energy, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_energy:.3f}')
        ax.legend()
    
    def _plot_final_state_panel(self, ax, final_state: DensityMatrix):
        """Plot final state density matrix in a single panel."""
        matrix = np.abs(final_state.matrix)
        im = ax.imshow(matrix, cmap='viridis', aspect='equal')
        ax.set_title('Final State |ρ|')
        ax.set_xlabel('Basis State')
        ax.set_ylabel('Basis State')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|ρ_ij|')
        
        # Set tick labels
        n_states = len(final_state.basis_labels)
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(final_state.basis_labels)
        ax.set_yticklabels(final_state.basis_labels)
    
    def _plot_decoherence_panel(self, ax, decoherence_rates: Dict[str, float]):
        """Plot decoherence rates as a bar chart."""
        if not decoherence_rates:
            ax.text(0.5, 0.5, 'No decoherence data', ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        rates = list(decoherence_rates.values())
        labels = [key.replace('_', ' ').title() for key in decoherence_rates.keys()]
        
        bars = ax.bar(labels, rates, color='skyblue', edgecolor='black')
        ax.set_ylabel('Rate (ps⁻¹)')
        ax.set_title('Decoherence Rates')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{rate:.3f}', ha='center', va='bottom')
    
    def _plot_statistics_panel(self, ax, stats: StatisticalSummary):
        """Plot statistical summary as a table."""
        # Create a simple statistics display
        stats_text = []
        stats_text.append("Statistical Summary")
        stats_text.append("-" * 20)
        stats_text.append(f"Sample Size: {stats.sample_size}")
        stats_text.append("")
        
        for key, mean_val in stats.mean_values.items():
            std_val = stats.std_deviations.get(key, 0)
            stats_text.append(f"{key.replace('_', ' ').title()}:")
            stats_text.append(f"  Mean: {mean_val:.4f}")
            stats_text.append(f"  Std:  {std_val:.4f}")
            
            if key in stats.confidence_intervals:
                ci = stats.confidence_intervals[key]
                stats_text.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            stats_text.append("")
        
        ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _generate_figure_caption(self, results: SimulationResults, panels: List[str]) -> str:
        """Generate automatic figure caption based on content."""
        caption_parts = []
        
        # Basic simulation info
        config = results.simulation_config
        caption_parts.append(f"Quantum biological simulation results for {config.quantum_subsystem_selection} ")
        caption_parts.append(f"at T = {config.temperature} K. ")
        
        # Panel descriptions
        panel_descriptions = {
            'state_evolution': 'Population dynamics of quantum states over time',
            'coherence_measures': 'Evolution of quantum coherence measures',
            'energy_evolution': 'System energy fluctuations during simulation',
            'final_state_matrix': 'Final density matrix showing quantum correlations',
            'decoherence_rates': 'Environmental decoherence rates for different processes',
            'statistical_summary': 'Statistical analysis of simulation results'
        }
        
        if len(panels) > 1:
            descriptions = [panel_descriptions.get(panel, panel) for panel in panels]
            caption_parts.append("Panels show: " + "; ".join(descriptions) + ". ")
        
        # Simulation parameters
        caption_parts.append(f"Simulation time: {config.simulation_time} ps, ")
        caption_parts.append(f"time step: {config.time_step} ps. ")
        caption_parts.append(f"Noise model: {config.noise_model_type}. ")
        caption_parts.append(f"Computation time: {results.computation_time:.1f} s.")
        
        return "".join(caption_parts)
    
    def export_publication_data(self, results: SimulationResults, 
                               output_dir: str, formats: List[str] = None) -> bool:
        """Export data in publication-ready formats."""
        try:
            if formats is None:
                formats = ['csv', 'json', 'hdf5']
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export state trajectory data
            if 'csv' in formats:
                self._export_csv_data(results, output_path)
            
            if 'json' in formats:
                self._export_json_data(results, output_path)
            
            if 'hdf5' in formats:
                self._export_hdf5_data(results, output_path)
            
            return True
            
        except Exception as e:
            print(f"Error exporting publication data: {e}")
            return False
    
    def _export_csv_data(self, results: SimulationResults, output_path: Path):
        """Export data to CSV format."""
        import pandas as pd
        
        # State populations over time
        times = [state.time for state in results.state_trajectory]
        n_states = len(results.state_trajectory[0].basis_labels)
        populations = np.zeros((len(times), n_states))
        
        for i, state in enumerate(results.state_trajectory):
            populations[i, :] = np.real(np.diag(state.matrix))
        
        # Create DataFrame
        columns = ['time'] + [f'pop_{label}' for label in results.state_trajectory[0].basis_labels]
        data = np.column_stack([times, populations])
        df_states = pd.DataFrame(data, columns=columns)
        df_states.to_csv(output_path / 'state_populations.csv', index=False)
        
        # Energy trajectory
        df_energy = pd.DataFrame({
            'time': times,
            'energy': results.energy_trajectory
        })
        df_energy.to_csv(output_path / 'energy_trajectory.csv', index=False)
        
        # Coherence measures
        if results.coherence_measures:
            coherence_data = {'time': times}
            for measure, values in results.coherence_measures.items():
                if len(values) == len(times):
                    coherence_data[measure] = values
            
            df_coherence = pd.DataFrame(coherence_data)
            df_coherence.to_csv(output_path / 'coherence_measures.csv', index=False)
    
    def _export_json_data(self, results: SimulationResults, output_path: Path):
        """Export metadata to JSON format."""
        import json
        
        metadata = {
            'simulation_config': {
                'system_pdb': results.simulation_config.system_pdb,
                'temperature': results.simulation_config.temperature,
                'simulation_time': results.simulation_config.simulation_time,
                'time_step': results.simulation_config.time_step,
                'quantum_subsystem_selection': results.simulation_config.quantum_subsystem_selection,
                'noise_model_type': results.simulation_config.noise_model_type
            },
            'decoherence_rates': results.decoherence_rates,
            'statistical_summary': {
                'mean_values': results.statistical_summary.mean_values,
                'std_deviations': results.statistical_summary.std_deviations,
                'confidence_intervals': results.statistical_summary.confidence_intervals,
                'sample_size': results.statistical_summary.sample_size
            },
            'computation_time': results.computation_time
        }
        
        with open(output_path / 'simulation_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _export_hdf5_data(self, results: SimulationResults, output_path: Path):
        """Export complete data to HDF5 format."""
        try:
            import h5py
            
            with h5py.File(output_path / 'simulation_results.h5', 'w') as f:
                # State trajectory
                times = [state.time for state in results.state_trajectory]
                f.create_dataset('times', data=times)
                
                # Store density matrices
                n_states = len(results.state_trajectory[0].basis_labels)
                matrices = np.zeros((len(times), n_states, n_states), dtype=complex)
                for i, state in enumerate(results.state_trajectory):
                    matrices[i] = state.matrix
                
                f.create_dataset('density_matrices', data=matrices)
                f.create_dataset('basis_labels', data=[s.encode() for s in results.state_trajectory[0].basis_labels])
                
                # Energy trajectory
                f.create_dataset('energy_trajectory', data=results.energy_trajectory)
                
                # Coherence measures
                if results.coherence_measures:
                    coherence_group = f.create_group('coherence_measures')
                    for measure, values in results.coherence_measures.items():
                        coherence_group.create_dataset(measure, data=values)
                
                # Metadata
                meta_group = f.create_group('metadata')
                meta_group.attrs['temperature'] = results.simulation_config.temperature
                meta_group.attrs['simulation_time'] = results.simulation_config.simulation_time
                meta_group.attrs['time_step'] = results.simulation_config.time_step
                meta_group.attrs['computation_time'] = results.computation_time
                
        except ImportError:
            print("h5py not available, skipping HDF5 export")
    
    def set_plot_style(self, style: str = "scientific") -> bool:
        """Set plotting style for consistent appearance."""
        try:
            self.plot_style = style
            self._setup_plot_style()
            return True
        except Exception as e:
            print(f"Error setting plot style: {e}")
            return False
    
    def _setup_plot_style(self):
        """Configure matplotlib for scientific plotting."""
        # Set non-interactive backend for headless environments
        import matplotlib
        matplotlib.use('Agg')
        
        # Set scientific plotting style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn' in plt.style.available else 'default')
        
        # Configure font and sizes for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'serif',
            'mathtext.fontset': 'dejavuserif',
            'lines.linewidth': 2,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            # Publication-specific settings
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.edgecolor': 'black',
            'legend.facecolor': 'white',
            'legend.framealpha': 0.9
        })
    
    def _format_measure_name(self, measure_name: str) -> str:
        """Format measure names for display."""
        formatting_map = {
            'coherence_lifetime': 'Coherence Lifetime (ps)',
            'quantum_discord': 'Quantum Discord',
            'entanglement_measure': 'Entanglement Measure',
            'purity': 'Purity',
            'von_neumann_entropy': 'Von Neumann Entropy'
        }
        return formatting_map.get(measure_name, measure_name.replace('_', ' ').title())
    
    def _add_exponential_fit(self, ax, times: np.ndarray, values: np.ndarray):
        """Add exponential decay fit to coherence data."""
        try:
            # Simple exponential fit: y = A * exp(-t/τ)
            from scipy.optimize import curve_fit
            
            def exp_decay(t, A, tau):
                return A * np.exp(-t / tau)
            
            # Only fit if we have positive values
            if np.all(np.array(values) > 0):
                popt, _ = curve_fit(exp_decay, times, values, 
                                  p0=[values[0], times[-1]/3])
                
                fit_times = np.linspace(times[0], times[-1], 100)
                fit_values = exp_decay(fit_times, *popt)
                
                ax.plot(fit_times, fit_values, '--', alpha=0.8, 
                       label=f'Exp fit: τ = {popt[1]:.3f} ps')
                ax.legend()
        except:
            # If fitting fails, just continue without the fit
            pass
    
    def _calculate_running_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate running average of data."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')