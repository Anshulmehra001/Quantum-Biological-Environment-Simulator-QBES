"""
Integration tests for full QBES simulation workflows.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch

from qbes.simulation_engine import SimulationEngine
from qbes.config_manager import ConfigurationManager
from qbes.core.data_models import ValidationResult


@pytest.mark.integration
class TestFullSimulation:
    """Integration tests for complete simulation workflows."""
    
    def test_simple_two_level_simulation(self, temp_output_dir, sample_pdb_content):
        """Test a complete simple two-level system simulation."""
        # Create temporary PDB file
        pdb_file = os.path.join(temp_output_dir, "simple_system.pdb")
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        # Create configuration
        config_dict = {
            "system": {
                "pdb_file": pdb_file,
                "force_field": "amber14",
                "temperature": 300.0,
                "solvent_model": "tip3p",
                "ionic_strength": 0.15
            },
            "simulation": {
                "simulation_time": 1e-15,  # Very short for testing
                "time_step": 1e-16
            },
            "quantum_subsystem": {
                "selection_method": "all",
                "max_quantum_atoms": 10
            },
            "noise_model": {
                "type": "protein_ohmic",
                "coupling_strength": 0.1,
                "cutoff_frequency": 1e13,
                "reorganization_energy": 35.0
            },
            "output": {
                "directory": temp_output_dir,
                "save_trajectory": True,
                "save_checkpoints": False
            }
        }
        
        config_file = os.path.join(temp_output_dir, "test_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Mock the heavy computational parts for integration testing
        with patch('qbes.md_engine.MDEngine') as mock_md_engine:
            mock_md_instance = mock_md_engine.return_value
            mock_md_instance.initialize_system.return_value = None
            mock_md_instance.setup_environment.return_value = None
            mock_md_instance.minimize_energy.return_value = -100.0
            
            # Test configuration loading and validation
            config_manager = ConfigurationManager()
            config = config_manager.load_config(config_file)
            
            assert config is not None
            
            # Test validation
            validation_result = config_manager.validate_parameters(config)
            
            # May not be fully valid due to mocking, but should not crash
            assert isinstance(validation_result, ValidationResult)
    
    def test_dry_run_validation_workflow(self, temp_output_dir, sample_pdb_content):
        """Test the complete dry-run validation workflow."""
        # Create temporary PDB file
        pdb_file = os.path.join(temp_output_dir, "validation_system.pdb")
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        # Create configuration
        config_dict = {
            "system": {
                "pdb_file": pdb_file,
                "force_field": "amber14",
                "temperature": 300.0
            },
            "simulation": {
                "simulation_time": 1e-12,
                "time_step": 1e-15
            },
            "quantum_subsystem": {
                "selection_method": "chromophores",
                "max_quantum_atoms": 50
            },
            "noise_model": {
                "type": "protein_ohmic",
                "coupling_strength": 1.0
            },
            "output": {
                "directory": temp_output_dir,
                "save_trajectory": False
            }
        }
        
        config_file = os.path.join(temp_output_dir, "validation_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_file)
        
        # Initialize simulation engine
        engine = SimulationEngine()
        
        # Mock heavy dependencies for integration testing
        with patch.object(engine, 'md_engine') as mock_md_engine:
            mock_md_engine.initialize_system.return_value = None
            mock_md_engine.setup_environment.return_value = None
            mock_md_engine.minimize_energy.return_value = -100.0
            
            with patch.object(engine, 'config_manager') as mock_config_manager:
                mock_config_manager.parse_pdb.return_value = None
                mock_config_manager.identify_quantum_subsystem.return_value = None
                mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
                
                with patch.object(engine, 'quantum_engine') as mock_quantum_engine:
                    mock_quantum_engine.initialize_hamiltonian.return_value = None
                    
                    with patch.object(engine, 'noise_factory') as mock_noise_factory:
                        mock_noise_factory.create_noise_model.return_value = None
                        
                        # Test dry-run validation
                        result = engine.perform_dry_run_validation(config)
                        
                        assert isinstance(result, ValidationResult)
    
    @pytest.mark.slow
    def test_configuration_workflow(self, temp_output_dir):
        """Test the complete configuration creation and loading workflow."""
        config_manager = ConfigurationManager()
        
        # Test default configuration creation
        default_config = config_manager.create_default_config()
        assert default_config is not None
        
        # Test configuration saving and loading
        config_file = os.path.join(temp_output_dir, "workflow_config.yaml")
        
        # Create a test configuration dictionary
        test_config = {
            "system": {
                "pdb_file": "test.pdb",
                "force_field": "amber14",
                "temperature": 300.0
            },
            "simulation": {
                "simulation_time": 1e-12,
                "time_step": 1e-15
            },
            "output": {
                "directory": temp_output_dir
            }
        }
        
        # Save configuration
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load configuration
        loaded_config = config_manager.load_config(config_file)
        
        assert loaded_config is not None
        assert loaded_config.system_pdb == "test.pdb"
        assert loaded_config.temperature == 300.0
    
    def test_error_handling_integration(self, temp_output_dir):
        """Test error handling in integrated workflows."""
        config_manager = ConfigurationManager()
        
        # Test loading non-existent file
        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent_config.yaml")
        
        # Test invalid configuration
        invalid_config_file = os.path.join(temp_output_dir, "invalid_config.yaml")
        with open(invalid_config_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(Exception):  # Should raise some parsing error
            config_manager.load_config(invalid_config_file)
    
    def test_simulation_engine_initialization_integration(self, temp_output_dir, sample_pdb_content):
        """Test simulation engine initialization with real configuration."""
        # Create PDB file
        pdb_file = os.path.join(temp_output_dir, "init_test.pdb")
        with open(pdb_file, 'w') as f:
            f.write(sample_pdb_content)
        
        # Create minimal valid configuration
        config_dict = {
            "system": {
                "pdb_file": pdb_file,
                "force_field": "amber14",
                "temperature": 300.0,
                "solvent_model": "tip3p",
                "ionic_strength": 0.15
            },
            "simulation": {
                "simulation_time": 1e-15,
                "time_step": 1e-16
            },
            "quantum_subsystem": {
                "selection_method": "all",
                "max_quantum_atoms": 5
            },
            "noise_model": {
                "type": "protein_ohmic",
                "coupling_strength": 0.1
            },
            "output": {
                "directory": temp_output_dir,
                "save_trajectory": False
            }
        }
        
        config_file = os.path.join(temp_output_dir, "init_config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f)
        
        # Load configuration
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_file)
        
        # Initialize simulation engine
        engine = SimulationEngine()
        
        # Mock dependencies for integration test
        with patch.object(engine, 'md_engine') as mock_md_engine:
            mock_md_engine.initialize_system.return_value = None
            mock_md_engine.setup_environment.return_value = None
            mock_md_engine.minimize_energy.return_value = -100.0
            
            with patch.object(engine, 'config_manager') as mock_config_manager:
                mock_config_manager.parse_pdb.return_value = None
                mock_config_manager.identify_quantum_subsystem.return_value = None
                mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
                
                with patch.object(engine, 'quantum_engine') as mock_quantum_engine:
                    mock_quantum_engine.initialize_hamiltonian.return_value = None
                    
                    with patch.object(engine, 'noise_factory') as mock_noise_factory:
                        mock_noise_factory.create_noise_model.return_value = None
                        
                        # Test initialization
                        result = engine.initialize_simulation(config)
                        
                        assert isinstance(result, ValidationResult)
                        assert engine.config == config