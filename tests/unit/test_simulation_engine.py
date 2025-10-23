"""
Unit tests for the SimulationEngine class.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from qbes.simulation_engine import SimulationEngine
from qbes.core.data_models import SimulationConfig, ValidationResult


class TestSimulationEngine:
    """Test suite for SimulationEngine functionality."""
    
    def test_initialization(self, simulation_engine):
        """Test SimulationEngine initialization."""
        assert simulation_engine is not None
        assert simulation_engine.progress == 0.0
        assert not simulation_engine.is_running
        assert not simulation_engine.is_paused
    
    def test_set_verbose_logging(self, simulation_engine):
        """Test setting verbose logging."""
        simulation_engine.set_verbose_logging(True)
        assert simulation_engine.enhanced_logger.verbose
        
        simulation_engine.set_verbose_logging(False)
        assert not simulation_engine.enhanced_logger.verbose
    
    def test_get_progress(self, simulation_engine):
        """Test progress tracking."""
        initial_progress = simulation_engine.get_progress()
        assert initial_progress == 0.0
        
        # Simulate some progress
        simulation_engine.progress = 50.0
        assert simulation_engine.get_progress() == 50.0
    
    def test_pause_resume_simulation(self, simulation_engine):
        """Test pause and resume functionality."""
        # Initially not running, pause should return False
        assert not simulation_engine.pause_simulation()
        
        # Simulate running state
        simulation_engine.is_running = True
        assert simulation_engine.pause_simulation()
        assert simulation_engine.is_paused
        
        # Test resume
        assert simulation_engine.resume_simulation()
        assert not simulation_engine.is_paused
    
    def test_dry_run_validation_basic(self, simulation_engine, sample_config_dict, 
                                     temp_output_dir, sample_pdb_content):
        """Test basic dry-run validation."""
        # Create a temporary PDB file
        pdb_file = os.path.join(temp_output_dir, "test_system.pdb")
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        # Update config with correct paths
        sample_config_dict["system"]["pdb_file"] = pdb_file
        sample_config_dict["output"]["directory"] = temp_output_dir
        
        # Create config object (mock for now)
        config = Mock()
        config.system_pdb = pdb_file
        config.output_directory = temp_output_dir
        config.quantum_subsystem_selection = "chromophores"
        config.force_field = "amber14"
        config.solvent_model = "tip3p"
        config.ionic_strength = 0.15
        config.noise_model_type = "protein_ohmic"
        config.temperature = 300.0
        config.simulation_time = 1e-12
        config.time_step = 1e-15
        
        # Mock the config manager and other dependencies
        with patch.object(simulation_engine, 'config_manager') as mock_config_manager:
            mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
            mock_config_manager.parse_pdb.return_value = Mock()
            mock_config_manager.identify_quantum_subsystem.return_value = Mock()
            
            with patch.object(simulation_engine, 'quantum_engine') as mock_quantum_engine:
                mock_quantum_engine.initialize_hamiltonian.return_value = Mock()
                
                with patch.object(simulation_engine, 'md_engine') as mock_md_engine:
                    mock_md_engine.initialize_system.return_value = Mock()
                    mock_md_engine.setup_environment.return_value = None
                    mock_md_engine.minimize_energy.return_value = -100.0
                    
                    with patch.object(simulation_engine, 'noise_factory') as mock_noise_factory:
                        mock_noise_factory.create_noise_model.return_value = Mock()
                        
                        result = simulation_engine.perform_dry_run_validation(config)
                        
                        assert isinstance(result, ValidationResult)
                        # The result might not be valid due to mocking, but it should not crash
    
    def test_checkpoint_save_load(self, simulation_engine, temp_output_dir):
        """Test checkpoint save and load functionality."""
        checkpoint_file = os.path.join(temp_output_dir, "test_checkpoint.pkl")
        
        # Set some state
        simulation_engine.progress = 75.0
        simulation_engine.current_time = 0.5
        simulation_engine.time_step_count = 1000
        
        # Save checkpoint
        success = simulation_engine.save_checkpoint(checkpoint_file)
        assert success
        assert os.path.exists(checkpoint_file)
        
        # Create new engine and load checkpoint
        new_engine = SimulationEngine()
        load_success = new_engine.load_checkpoint(checkpoint_file)
        assert load_success
        assert new_engine.progress == 75.0
        assert new_engine.current_time == 0.5
        assert new_engine.time_step_count == 1000
    
    def test_checkpoint_invalid_file(self, simulation_engine):
        """Test checkpoint operations with invalid files."""
        # Test loading non-existent file
        assert not simulation_engine.load_checkpoint("nonexistent.pkl")
        
        # Test saving to invalid path
        assert not simulation_engine.save_checkpoint("/invalid/path/checkpoint.pkl")
    
    @pytest.mark.parametrize("progress_value", [0.0, 25.5, 50.0, 99.9, 100.0])
    def test_progress_tracking(self, simulation_engine, progress_value):
        """Test progress tracking with different values."""
        simulation_engine.progress = progress_value
        assert simulation_engine.get_progress() == progress_value
    
    def test_state_management(self, simulation_engine):
        """Test simulation state management."""
        # Initial state
        assert not simulation_engine.is_running
        assert not simulation_engine.is_paused
        
        # Start simulation (mock)
        simulation_engine.is_running = True
        assert simulation_engine.is_running
        
        # Pause
        simulation_engine.pause_simulation()
        assert simulation_engine.is_paused
        assert simulation_engine.is_running
        
        # Resume
        simulation_engine.resume_simulation()
        assert not simulation_engine.is_paused
        assert simulation_engine.is_running
        
        # Stop
        simulation_engine.is_running = False
        assert not simulation_engine.is_running
    
    def test_error_handling(self, simulation_engine):
        """Test error handling in simulation engine."""
        # Test with None config
        with pytest.raises(RuntimeError):
            simulation_engine.run_simulation()
    
    def test_initialization_with_config(self, simulation_engine, sample_config_dict, 
                                       temp_output_dir, sample_pdb_content):
        """Test initialization with configuration."""
        # Create temporary PDB file
        pdb_file = os.path.join(temp_output_dir, "test_system.pdb")
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        # Create mock config
        config = Mock()
        config.system_pdb = pdb_file
        config.output_directory = temp_output_dir
        config.quantum_subsystem_selection = "chromophores"
        config.force_field = "amber14"
        config.solvent_model = "tip3p"
        config.ionic_strength = 0.15
        config.noise_model_type = "protein_ohmic"
        config.temperature = 300.0
        config.simulation_time = 1e-12
        config.time_step = 1e-15
        config.debug_level = "INFO"
        
        # Mock dependencies
        with patch.object(simulation_engine, 'config_manager') as mock_config_manager:
            mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
            mock_config_manager.parse_pdb.return_value = Mock()
            mock_config_manager.identify_quantum_subsystem.return_value = Mock()
            
            with patch.object(simulation_engine, 'quantum_engine') as mock_quantum_engine:
                mock_quantum_engine.initialize_hamiltonian.return_value = Mock()
                
                with patch.object(simulation_engine, 'md_engine') as mock_md_engine:
                    mock_md_engine.initialize_system.return_value = Mock()
                    mock_md_engine.setup_environment.return_value = None
                    mock_md_engine.minimize_energy.return_value = -100.0
                    
                    with patch.object(simulation_engine, 'noise_factory') as mock_noise_factory:
                        mock_noise_factory.create_noise_model.return_value = Mock()
                        
                        result = simulation_engine.initialize_simulation(config)
                        
                        assert isinstance(result, ValidationResult)
                        assert simulation_engine.config == config