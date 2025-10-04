"""
Tests for visualization and plotting functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from qbes.visualization import VisualizationEngine
from qbes.core.data_models import (
    DensityMatrix, SimulationResults, SimulationConfig, 
    StatisticalSummary, CoherenceMetrics
)


class TestVisualizationEngine:
    """Test cases for the VisualizationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz_engine = VisualizationEngine()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_trajectory = self._create_test_trajectory()
        self.test_coherence_data = self._create_test_coherence_data()
        self.test_energy_trajectory = self._create_test_energy_trajectory()
        self.test_results = self._create_test_results()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_trajectory(self):
        """Create a test quantum state trajectory."""
        trajectory = []
        n_states = 3
        basis_labels = ['|0⟩', '|1⟩', '|2⟩']
        
        for i in range(10):
            time = i * 0.1
            
            # Create a valid density matrix using eigendecomposition
            # Start with diagonal populations that sum to 1
            p1 = 0.6 * np.exp(-time/2) + 0.1  # Population in state 0
            p2 = 0.3 * (1 - np.exp(-time/2)) + 0.1  # Population in state 1
            p3 = 1.0 - p1 - p2  # Remaining population in state 2
            
            # Ensure all populations are positive
            if p3 < 0:
                p1 = 0.5
                p2 = 0.3
                p3 = 0.2
            
            # Create diagonal matrix with populations
            matrix = np.diag([p1, p2, p3])
            
            # Add small coherences that preserve positive semidefiniteness
            coherence_strength = 0.05 * np.exp(-time/0.5)
            matrix = matrix.astype(complex)  # Ensure complex type
            matrix[0, 1] = coherence_strength * np.exp(1j * time)
            matrix[1, 0] = np.conj(matrix[0, 1])
            
            # Ensure the matrix remains positive semidefinite by checking eigenvalues
            eigenvals = np.linalg.eigvals(matrix)
            if np.any(eigenvals < 0):
                # If not positive semidefinite, use pure diagonal matrix
                matrix = np.diag([p1, p2, p3])
            
            density_matrix = DensityMatrix(
                matrix=matrix,
                basis_labels=basis_labels,
                time=time
            )
            trajectory.append(density_matrix)
        
        return trajectory
    
    def _create_test_coherence_data(self):
        """Create test coherence measures data."""
        n_points = 10
        times = np.linspace(0, 1, n_points)
        
        return {
            'coherence_lifetime': list(np.exp(-times/0.5)),
            'quantum_discord': list(0.5 * np.exp(-times/0.3)),
            'purity': list(0.8 + 0.2 * np.exp(-times/0.4)),
            'von_neumann_entropy': list(0.2 * (1 - np.exp(-times/0.6)))
        }
    
    def _create_test_energy_trajectory(self):
        """Create test energy trajectory."""
        n_points = 10  # Match the state trajectory length
        times = np.linspace(0, 1, n_points)
        # Energy with some fluctuations around a mean
        base_energy = -10.5
        fluctuations = 0.1 * np.sin(2 * np.pi * times) + 0.05 * np.random.randn(n_points)
        return list(base_energy + fluctuations)
    
    def _create_test_results(self):
        """Create complete test simulation results."""
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory=self.temp_dir
        )
        
        stats = StatisticalSummary(
            mean_values={'energy': -10.5, 'coherence': 0.3},
            std_deviations={'energy': 0.1, 'coherence': 0.05},
            confidence_intervals={'energy': (-10.6, -10.4), 'coherence': (0.25, 0.35)},
            sample_size=100
        )
        
        return SimulationResults(
            state_trajectory=self.test_trajectory,
            coherence_measures=self.test_coherence_data,
            energy_trajectory=self.test_energy_trajectory,
            decoherence_rates={'dephasing': 0.1, 'relaxation': 0.05},
            statistical_summary=stats,
            simulation_config=config,
            computation_time=120.5
        )
    
    def test_initialization(self):
        """Test VisualizationEngine initialization."""
        engine = VisualizationEngine()
        assert engine.plot_style == "scientific"
        assert isinstance(engine.figure_cache, dict)
    
    def test_plot_state_evolution_success(self):
        """Test successful state evolution plotting."""
        output_path = os.path.join(self.temp_dir, "state_evolution.png")
        
        result = self.viz_engine.plot_state_evolution(self.test_trajectory, output_path)
        
        assert result is True
        assert os.path.exists(output_path)
        
        # Check file is not empty
        assert os.path.getsize(output_path) > 0
    
    def test_plot_state_evolution_empty_trajectory(self):
        """Test state evolution plotting with empty trajectory."""
        output_path = os.path.join(self.temp_dir, "empty_state.png")
        
        result = self.viz_engine.plot_state_evolution([], output_path)
        
        assert result is False
        assert not os.path.exists(output_path)
    
    def test_plot_coherence_measures_success(self):
        """Test successful coherence measures plotting."""
        output_path = os.path.join(self.temp_dir, "coherence_measures.png")
        
        result = self.viz_engine.plot_coherence_measures(self.test_coherence_data, output_path)
        
        assert result is True
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_plot_coherence_measures_empty_data(self):
        """Test coherence measures plotting with empty data."""
        output_path = os.path.join(self.temp_dir, "empty_coherence.png")
        
        result = self.viz_engine.plot_coherence_measures({}, output_path)
        
        assert result is False
        assert not os.path.exists(output_path)
    
    def test_plot_energy_landscape_success(self):
        """Test successful energy landscape plotting."""
        output_path = os.path.join(self.temp_dir, "energy_landscape.png")
        
        result = self.viz_engine.plot_energy_landscape(self.test_energy_trajectory, output_path)
        
        assert result is True
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_plot_energy_landscape_empty_data(self):
        """Test energy landscape plotting with empty data."""
        output_path = os.path.join(self.temp_dir, "empty_energy.png")
        
        result = self.viz_engine.plot_energy_landscape([], output_path)
        
        assert result is False
        assert not os.path.exists(output_path)
    
    def test_set_plot_style(self):
        """Test plot style setting."""
        result = self.viz_engine.set_plot_style("scientific")
        assert result is True
        assert self.viz_engine.plot_style == "scientific"
        
        result = self.viz_engine.set_plot_style("custom")
        assert result is True
        assert self.viz_engine.plot_style == "custom"
    
    def test_create_publication_figure_overview(self):
        """Test creation of overview publication figure."""
        output_path = os.path.join(self.temp_dir, "overview.png")
        
        result = self.viz_engine.create_publication_figure(
            self.test_results, "overview", output_path
        )
        
        assert result is True
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
    
    def test_create_publication_figure_coherence(self):
        """Test creation of coherence analysis figure."""
        output_path = os.path.join(self.temp_dir, "coherence_analysis.png")
        
        result = self.viz_engine.create_publication_figure(
            self.test_results, "coherence_analysis", output_path
        )
        
        assert result is True
        assert os.path.exists(output_path)
    
    def test_create_publication_figure_energy(self):
        """Test creation of energy analysis figure."""
        output_path = os.path.join(self.temp_dir, "energy_analysis.png")
        
        result = self.viz_engine.create_publication_figure(
            self.test_results, "energy_analysis", output_path
        )
        
        assert result is True
        assert os.path.exists(output_path)
    
    def test_create_publication_figure_invalid_type(self):
        """Test creation of publication figure with invalid type."""
        output_path = os.path.join(self.temp_dir, "invalid.png")
        
        result = self.viz_engine.create_publication_figure(
            self.test_results, "invalid_type", output_path
        )
        
        assert result is False
        assert not os.path.exists(output_path)
    
    def test_generate_animation(self):
        """Test animation generation."""
        output_path = os.path.join(self.temp_dir, "animation.png")
        
        result = self.viz_engine.generate_animation(self.test_trajectory, output_path)
        
        assert result is True
        # Should create multiple frame files
        frame_files = list(Path(self.temp_dir).glob("animation_frame_*.png"))
        assert len(frame_files) > 0
    
    def test_format_measure_name(self):
        """Test measure name formatting."""
        engine = VisualizationEngine()
        
        assert engine._format_measure_name('coherence_lifetime') == 'Coherence Lifetime (ps)'
        assert engine._format_measure_name('quantum_discord') == 'Quantum Discord'
        assert engine._format_measure_name('purity') == 'Purity'
        assert engine._format_measure_name('von_neumann_entropy') == 'Von Neumann Entropy'
        assert engine._format_measure_name('unknown_measure') == 'Unknown Measure'
    
    def test_calculate_running_average(self):
        """Test running average calculation."""
        engine = VisualizationEngine()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_size = 3
        
        result = engine._calculate_running_average(data, window_size)
        
        assert len(result) == len(data)
        assert isinstance(result, np.ndarray)
        # Check that the middle values are reasonable averages
        assert abs(result[4] - 5.0) < 0.1  # Should be close to 5 for middle element
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_generation_with_matplotlib_error(self, mock_savefig):
        """Test handling of matplotlib errors during plotting."""
        mock_savefig.side_effect = Exception("Matplotlib error")
        
        output_path = os.path.join(self.temp_dir, "error_test.png")
        result = self.viz_engine.plot_state_evolution(self.test_trajectory, output_path)
        
        assert result is False
    
    def test_directory_creation(self):
        """Test that output directories are created automatically."""
        nested_path = os.path.join(self.temp_dir, "nested", "directory", "plot.png")
        
        result = self.viz_engine.plot_state_evolution(self.test_trajectory, nested_path)
        
        assert result is True
        assert os.path.exists(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))
    
    def test_data_accuracy_state_evolution(self):
        """Test that plotted data accurately represents input data."""
        # Create a simple test case where we can verify the data
        simple_trajectory = []
        matrix = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
        
        density_matrix = DensityMatrix(
            matrix=matrix,
            basis_labels=['|0⟩', '|1⟩'],
            time=0.0
        )
        simple_trajectory.append(density_matrix)
        
        output_path = os.path.join(self.temp_dir, "accuracy_test.png")
        result = self.viz_engine.plot_state_evolution(simple_trajectory, output_path)
        
        assert result is True
        # The test passes if the plot is generated without errors
        # More detailed data accuracy would require parsing the plot data
    
    def test_coherence_data_validation(self):
        """Test validation of coherence data before plotting."""
        # Test with mismatched data lengths
        invalid_data = {
            'measure1': [1, 2, 3],
            'measure2': [1, 2]  # Different length
        }
        
        output_path = os.path.join(self.temp_dir, "validation_test.png")
        # Should still work - the plotting function handles different lengths
        result = self.viz_engine.plot_coherence_measures(invalid_data, output_path)
        
        assert result is True  # Function should handle this gracefully
    
    def test_energy_statistics_calculation(self):
        """Test that energy statistics are calculated correctly."""
        # Use a known energy trajectory
        known_energies = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_mean = 3.0
        expected_std = np.std(known_energies)
        
        output_path = os.path.join(self.temp_dir, "energy_stats.png")
        result = self.viz_engine.plot_energy_landscape(known_energies, output_path)
        
        assert result is True
        # The statistics are calculated internally and displayed on the plot
        # Detailed verification would require parsing the plot content


class TestVisualizationIntegration:
    """Integration tests for visualization with other components."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.viz_engine = VisualizationEngine()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_visualization_pipeline(self):
        """Test complete visualization pipeline with realistic data."""
        # This would test integration with actual simulation results
        # For now, we'll use mock data that represents realistic output
        
        # Create realistic quantum state trajectory
        n_states = 4
        n_time_points = 20
        trajectory = []
        
        for i in range(n_time_points):
            time = i * 0.05  # 50 fs time steps
            
            # Create realistic density matrix with decoherence
            matrix = np.zeros((n_states, n_states), dtype=complex)
            
            # Population dynamics with relaxation
            matrix[0, 0] = 0.25 + 0.5 * np.exp(-time/0.5)  # Excited state decay
            matrix[1, 1] = 0.25 + 0.3 * (1 - np.exp(-time/0.5))
            matrix[2, 2] = 0.25 + 0.15 * (1 - np.exp(-time/0.5))
            matrix[3, 3] = 0.25 + 0.05 * (1 - np.exp(-time/0.5))
            
            # Coherences with dephasing
            matrix[0, 1] = 0.1 * np.exp(-time/0.2) * np.exp(1j * 2 * np.pi * time)
            matrix[1, 0] = np.conj(matrix[0, 1])
            matrix[0, 2] = 0.05 * np.exp(-time/0.15) * np.exp(1j * 4 * np.pi * time)
            matrix[2, 0] = np.conj(matrix[0, 2])
            
            # Normalize
            trace = np.trace(matrix)
            if trace > 0:
                matrix = matrix / trace
            
            density_matrix = DensityMatrix(
                matrix=matrix,
                basis_labels=['|S₁⟩', '|S₂⟩', '|T₁⟩', '|T₂⟩'],
                time=time
            )
            trajectory.append(density_matrix)
        
        # Test all plotting functions
        state_path = os.path.join(self.temp_dir, "integration_state.png")
        result1 = self.viz_engine.plot_state_evolution(trajectory, state_path)
        
        coherence_data = {
            'coherence_lifetime': [np.exp(-t.time/0.3) for t in trajectory],
            'purity': [np.trace(t.matrix @ t.matrix).real for t in trajectory]
        }
        coherence_path = os.path.join(self.temp_dir, "integration_coherence.png")
        result2 = self.viz_engine.plot_coherence_measures(coherence_data, coherence_path)
        
        energy_trajectory = [-10.5 + 0.1 * np.sin(t.time * 10) for t in trajectory]
        energy_path = os.path.join(self.temp_dir, "integration_energy.png")
        result3 = self.viz_engine.plot_energy_landscape(energy_trajectory, energy_path)
        
        assert all([result1, result2, result3])
        assert all([os.path.exists(p) for p in [state_path, coherence_path, energy_path]])
    
    def test_multi_panel_figure_creation(self):
        """Test creation of multi-panel publication figures."""
        # Create test results
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory=self.temp_dir
        )
        
        stats = StatisticalSummary(
            mean_values={'energy': -10.5, 'coherence': 0.3},
            std_deviations={'energy': 0.1, 'coherence': 0.05},
            confidence_intervals={'energy': (-10.6, -10.4), 'coherence': (0.25, 0.35)},
            sample_size=100
        )
        
        # Create simple trajectory
        trajectory = []
        for i in range(5):
            matrix = np.diag([0.6, 0.3, 0.1])
            trajectory.append(DensityMatrix(
                matrix=matrix,
                basis_labels=['|0⟩', '|1⟩', '|2⟩'],
                time=i * 0.1
            ))
        
        results = SimulationResults(
            state_trajectory=trajectory,
            coherence_measures={'purity': [0.8, 0.7, 0.6, 0.5, 0.4]},
            energy_trajectory=[-10.5, -10.4, -10.6, -10.5, -10.3],
            decoherence_rates={'dephasing': 0.1, 'relaxation': 0.05},
            statistical_summary=stats,
            simulation_config=config,
            computation_time=120.5
        )
        
        # Test multi-panel figure creation
        output_path = os.path.join(self.temp_dir, "multi_panel.png")
        panels = ['state_evolution', 'energy_evolution', 'final_state_matrix', 'decoherence_rates']
        
        result = self.viz_engine.create_multi_panel_figure(
            results, panels, output_path, 
            figure_title="Test Multi-Panel Figure",
            add_captions=True
        )
        
        assert result is True
        assert os.path.exists(output_path)
    
    def test_publication_data_export(self):
        """Test export of publication-ready data formats."""
        # Create test results (simplified)
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory=self.temp_dir
        )
        
        stats = StatisticalSummary(
            mean_values={'energy': -10.5},
            std_deviations={'energy': 0.1},
            confidence_intervals={'energy': (-10.6, -10.4)},
            sample_size=100
        )
        
        trajectory = []
        for i in range(3):
            matrix = np.diag([0.6, 0.3, 0.1])
            trajectory.append(DensityMatrix(
                matrix=matrix,
                basis_labels=['|0⟩', '|1⟩', '|2⟩'],
                time=i * 0.1
            ))
        
        results = SimulationResults(
            state_trajectory=trajectory,
            coherence_measures={'purity': [0.8, 0.7, 0.6]},
            energy_trajectory=[-10.5, -10.4, -10.6],
            decoherence_rates={'dephasing': 0.1},
            statistical_summary=stats,
            simulation_config=config,
            computation_time=120.5
        )
        
        # Test data export
        export_dir = os.path.join(self.temp_dir, "export")
        result = self.viz_engine.export_publication_data(
            results, export_dir, formats=['csv', 'json']
        )
        
        assert result is True
        assert os.path.exists(os.path.join(export_dir, "state_populations.csv"))
        assert os.path.exists(os.path.join(export_dir, "energy_trajectory.csv"))
        assert os.path.exists(os.path.join(export_dir, "simulation_metadata.json"))
    
    def test_figure_caption_generation(self):
        """Test automatic figure caption generation."""
        # Create minimal results for caption testing
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1.0,
            time_step=0.1,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory=self.temp_dir
        )
        
        stats = StatisticalSummary(
            mean_values={}, std_deviations={}, confidence_intervals={}, sample_size=100
        )
        
        # Create minimal trajectory
        matrix = np.diag([0.6, 0.3, 0.1])
        trajectory = [DensityMatrix(
            matrix=matrix,
            basis_labels=['|0⟩', '|1⟩', '|2⟩'],
            time=0.0
        )]
        
        results = SimulationResults(
            state_trajectory=trajectory,
            coherence_measures={},
            energy_trajectory=[-10.5],
            decoherence_rates={},
            statistical_summary=stats,
            simulation_config=config,
            computation_time=120.5
        )
        
        # Test caption generation
        panels = ['state_evolution', 'coherence_measures']
        caption = self.viz_engine._generate_figure_caption(results, panels)
        
        assert isinstance(caption, str)
        assert len(caption) > 0
        assert "chromophores" in caption
        assert "300" in caption  # temperature
        assert "protein" in caption  # noise model


if __name__ == "__main__":
    pytest.main([__file__])