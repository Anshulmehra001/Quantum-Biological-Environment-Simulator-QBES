"""
Tests for QBES command-line interface.
"""

import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock

from qbes.cli import main, generate_config, validate, run, resume, monitor_sim, status
from qbes.core.data_models import SimulationConfig, ValidationResult


class TestCLI:
    """Test suite for CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_main_help(self):
        """Test main CLI help command."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Quantum Biological Environment Simulator' in result.output
        assert 'run' in result.output
        assert 'generate-config' in result.output
        assert 'validate' in result.output
    
    def test_generate_config_default(self):
        """Test default configuration generation."""
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        result = self.runner.invoke(generate_config, [config_file])
        assert result.exit_code == 0
        assert os.path.exists(config_file)
        
        # Verify config content
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'system' in config
        assert 'simulation' in config
        assert 'quantum_subsystem' in config
        assert 'noise_model' in config
        assert 'output' in config
        
        # Check specific values
        assert config['system']['pdb_file'] == 'system.pdb'
        assert config['simulation']['temperature'] == 300.0
        assert config['quantum_subsystem']['selection_method'] == 'chromophores'
    
    def test_generate_config_photosystem_template(self):
        """Test photosystem configuration template generation."""
        config_file = os.path.join(self.temp_dir, 'photosystem_config.yaml')
        
        result = self.runner.invoke(generate_config, [config_file, '--template', 'photosystem'])
        assert result.exit_code == 0
        assert os.path.exists(config_file)
        
        # Verify photosystem-specific content
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['quantum_subsystem']['custom_selection'] == 'resname CHL BCL'
        assert config['noise_model']['reorganization_energy'] == 35.0
        assert 'photosystem_output' in config['output']['directory']
    
    def test_generate_config_enzyme_template(self):
        """Test enzyme configuration template generation."""
        config_file = os.path.join(self.temp_dir, 'enzyme_config.yaml')
        
        result = self.runner.invoke(generate_config, [config_file, '--template', 'enzyme'])
        assert result.exit_code == 0
        assert os.path.exists(config_file)
        
        # Verify enzyme-specific content
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['simulation']['temperature'] == 310.0  # Body temperature
        assert config['quantum_subsystem']['selection_method'] == 'active_site'
        assert config['quantum_subsystem']['max_quantum_atoms'] == 50
    
    def test_generate_config_membrane_template(self):
        """Test membrane configuration template generation."""
        config_file = os.path.join(self.temp_dir, 'membrane_config.yaml')
        
        result = self.runner.invoke(generate_config, [config_file, '--template', 'membrane'])
        assert result.exit_code == 0
        assert os.path.exists(config_file)
        
        # Verify membrane-specific content
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['system']['force_field'] == 'charmm36'
        assert config['noise_model']['type'] == 'membrane_lipid'
        assert config['noise_model']['reorganization_energy'] == 25.0
    
    @patch('qbes.cli.ConfigurationManager')
    def test_validate_config_valid(self, mock_config_manager):
        """Test configuration validation with valid config."""
        # Create test config file
        config_file = os.path.join(self.temp_dir, 'valid_config.yaml')
        config_data = {
            'system': {'pdb_file': 'test.pdb'},
            'simulation': {'temperature': 300.0}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Mock configuration manager
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_config = Mock()
        mock_manager.load_config.return_value = mock_config
        
        mock_validation = ValidationResult(is_valid=True)
        mock_manager.validate_parameters.return_value = mock_validation
        
        result = self.runner.invoke(validate, [config_file])
        assert result.exit_code == 0
        assert '✓ Configuration is valid' in result.output
    
    @patch('qbes.cli.ConfigurationManager')
    def test_validate_config_invalid(self, mock_config_manager):
        """Test configuration validation with invalid config."""
        # Create test config file
        config_file = os.path.join(self.temp_dir, 'invalid_config.yaml')
        config_data = {'invalid': 'config'}
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Mock configuration manager
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_config = Mock()
        mock_manager.load_config.return_value = mock_config
        
        mock_validation = ValidationResult(is_valid=False)
        mock_validation.errors = ['Missing required parameter: system.pdb_file']
        mock_validation.warnings = ['Temperature not specified, using default']
        mock_manager.validate_parameters.return_value = mock_validation
        
        result = self.runner.invoke(validate, [config_file])
        assert result.exit_code == 1
        assert '✗ Configuration validation failed' in result.output
        assert 'Missing required parameter' in result.output
    
    @patch('qbes.cli.SimulationEngine')
    @patch('qbes.cli.ConfigurationManager')
    def test_run_simulation_dry_run(self, mock_config_manager, mock_engine):
        """Test dry run simulation."""
        # Create test config file
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        config_data = {
            'system': {'pdb_file': 'test.pdb'},
            'simulation': {'temperature': 300.0},
            'output': {'directory': self.temp_dir}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Mock configuration manager
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_config = Mock()
        mock_config.output_directory = self.temp_dir
        mock_manager.load_config.return_value = mock_config
        
        mock_validation = ValidationResult(is_valid=True)
        mock_manager.validate_parameters.return_value = mock_validation
        
        result = self.runner.invoke(run, [config_file, '--dry-run'])
        assert result.exit_code == 0
        assert 'Configuration is valid. Dry run complete.' in result.output
    
    @patch('qbes.cli.SimulationEngine')
    @patch('qbes.cli.ConfigurationManager')
    def test_run_simulation_success(self, mock_config_manager, mock_engine):
        """Test successful simulation run."""
        # Create test config file
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        config_data = {
            'system': {'pdb_file': 'test.pdb'},
            'simulation': {'temperature': 300.0},
            'output': {'directory': self.temp_dir}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Mock configuration manager
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        mock_config = Mock()
        mock_config.output_directory = self.temp_dir
        mock_manager.load_config.return_value = mock_config
        
        mock_validation = ValidationResult(is_valid=True)
        mock_manager.validate_parameters.return_value = mock_validation
        
        # Mock simulation engine
        mock_sim_engine = Mock()
        mock_engine.return_value = mock_sim_engine
        
        mock_init_result = ValidationResult(is_valid=True)
        mock_sim_engine.initialize_simulation.return_value = mock_init_result
        
        mock_results = Mock()
        mock_sim_engine.run_simulation.return_value = mock_results
        mock_sim_engine.is_running = False
        
        result = self.runner.invoke(run, [config_file])
        assert result.exit_code == 0
        assert 'Starting simulation...' in result.output
        assert 'Simulation completed successfully!' in result.output
    
    @patch('qbes.cli.SimulationEngine')
    def test_resume_simulation(self, mock_engine):
        """Test resuming simulation from checkpoint."""
        # Create dummy checkpoint file
        checkpoint_file = os.path.join(self.temp_dir, 'checkpoint.pkl')
        with open(checkpoint_file, 'wb') as f:
            f.write(b'dummy checkpoint data')
        
        # Mock simulation engine
        mock_sim_engine = Mock()
        mock_engine.return_value = mock_sim_engine
        
        mock_sim_engine.load_checkpoint.return_value = True
        mock_sim_engine.get_progress.return_value = 45.0
        mock_sim_engine.config.output_directory = self.temp_dir
        mock_sim_engine.is_running = False
        
        mock_results = Mock()
        mock_sim_engine.run_simulation.return_value = mock_results
        
        result = self.runner.invoke(resume, [checkpoint_file])
        assert result.exit_code == 0
        assert 'Resuming simulation from:' in result.output
        assert 'Resuming from 45.0% completion' in result.output
        assert 'Simulation completed successfully!' in result.output
    
    @patch('qbes.cli.SimulationEngine')
    def test_resume_simulation_failed_checkpoint(self, mock_engine):
        """Test resuming simulation with invalid checkpoint."""
        # Create dummy checkpoint file
        checkpoint_file = os.path.join(self.temp_dir, 'bad_checkpoint.pkl')
        with open(checkpoint_file, 'wb') as f:
            f.write(b'invalid checkpoint data')
        
        # Mock simulation engine
        mock_sim_engine = Mock()
        mock_engine.return_value = mock_sim_engine
        
        mock_sim_engine.load_checkpoint.return_value = False
        
        result = self.runner.invoke(resume, [checkpoint_file])
        assert result.exit_code == 1
        assert 'Failed to load checkpoint file' in result.output
    
    def test_status_not_started(self):
        """Test status command for simulation that hasn't started."""
        result = self.runner.invoke(status, [self.temp_dir])
        assert result.exit_code == 0
        assert 'Status: NOT STARTED ⚪' in result.output
        assert 'No simulation files found' in result.output
    
    def test_status_running(self):
        """Test status command for running simulation."""
        # Create intermediate results file
        intermediate_file = os.path.join(self.temp_dir, 'intermediate_001.json')
        intermediate_data = {
            'progress': 35.5,
            'step': 1000,
            'time': 1.0e-12
        }
        
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_data, f)
        
        result = self.runner.invoke(status, [self.temp_dir])
        assert result.exit_code == 0
        assert 'Status: RUNNING ⏳' in result.output
        assert 'Progress: 35.5%' in result.output
        assert 'Current step: 1000' in result.output
    
    def test_status_completed(self):
        """Test status command for completed simulation."""
        # Create results file
        results_file = os.path.join(self.temp_dir, 'simulation_results.json')
        results_data = {
            'simulation_time': 1.0e-12,
            'final_step': 5000
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f)
        
        result = self.runner.invoke(status, [self.temp_dir])
        assert result.exit_code == 0
        assert 'Status: COMPLETED ✓' in result.output
        assert 'Total simulation time: 1.00e-12 s' in result.output
        assert 'Final step: 5000' in result.output
    
    def test_status_failed(self):
        """Test status command for failed simulation."""
        # Create error log file
        error_file = os.path.join(self.temp_dir, 'error_001.log')
        error_content = """
        Error: Simulation failed
        Traceback (most recent call last):
        File "simulation.py", line 100, in run
        RuntimeError: Numerical instability detected
        """
        
        with open(error_file, 'w') as f:
            f.write(error_content)
        
        result = self.runner.invoke(status, [self.temp_dir])
        assert result.exit_code == 0
        assert 'Status: FAILED ✗' in result.output
        assert 'RuntimeError: Numerical instability detected' in result.output
    
    def test_status_paused(self):
        """Test status command for paused simulation."""
        # Create checkpoint file
        checkpoint_file = os.path.join(self.temp_dir, 'checkpoint_001.pkl')
        with open(checkpoint_file, 'wb') as f:
            f.write(b'checkpoint data')
        
        result = self.runner.invoke(status, [self.temp_dir])
        assert result.exit_code == 0
        assert 'Status: PAUSED ⏸' in result.output
        assert 'checkpoint_001.pkl' in result.output
    
    def test_status_nonexistent_directory(self):
        """Test status command with nonexistent directory."""
        nonexistent_dir = os.path.join(self.temp_dir, 'nonexistent')
        
        result = self.runner.invoke(status, [nonexistent_dir])
        assert result.exit_code == 1
        assert 'Output directory does not exist' in result.output
    
    @patch('qbes.cli.time.sleep')
    @patch('qbes.cli.click.echo')
    def test_monitor_simulation(self, mock_echo, mock_sleep):
        """Test simulation monitoring."""
        # Create intermediate results file
        intermediate_file = os.path.join(self.temp_dir, 'intermediate_001.json')
        intermediate_data = {
            'progress': 25.0,
            'step': 500,
            'time': 5.0e-13
        }
        
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_data, f)
        
        # Mock sleep to raise KeyboardInterrupt after first iteration
        mock_sleep.side_effect = KeyboardInterrupt()
        
        result = self.runner.invoke(monitor_sim, [self.temp_dir])
        assert result.exit_code == 0
        
        # Verify monitoring output was called
        mock_echo.assert_called()
    
    def test_monitor_nonexistent_directory(self):
        """Test monitoring with nonexistent directory."""
        nonexistent_dir = os.path.join(self.temp_dir, 'nonexistent')
        
        result = self.runner.invoke(monitor_sim, [nonexistent_dir])
        assert result.exit_code == 1
        assert 'Output directory does not exist' in result.output


class TestCLIErrorHandling:
    """Test error handling in CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_run_nonexistent_config(self):
        """Test running with nonexistent config file."""
        nonexistent_config = os.path.join(self.temp_dir, 'nonexistent.yaml')
        
        result = self.runner.invoke(run, [nonexistent_config])
        assert result.exit_code == 2  # Click file not found error
    
    def test_validate_nonexistent_config(self):
        """Test validating nonexistent config file."""
        nonexistent_config = os.path.join(self.temp_dir, 'nonexistent.yaml')
        
        result = self.runner.invoke(validate, [nonexistent_config])
        assert result.exit_code == 2  # Click file not found error
    
    def test_resume_nonexistent_checkpoint(self):
        """Test resuming with nonexistent checkpoint file."""
        nonexistent_checkpoint = os.path.join(self.temp_dir, 'nonexistent.pkl')
        
        result = self.runner.invoke(resume, [nonexistent_checkpoint])
        assert result.exit_code == 2  # Click file not found error
    
    @patch('qbes.cli.ConfigurationManager')
    def test_generate_config_write_error(self, mock_config_manager):
        """Test config generation with write permission error."""
        # Try to write to a directory that doesn't exist
        invalid_path = '/invalid/path/config.yaml'
        
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        mock_manager.generate_default_config.return_value = False
        
        result = self.runner.invoke(generate_config, [invalid_path])
        assert result.exit_code == 1
        assert 'Failed to generate configuration template' in result.output


if __name__ == '__main__':
    pytest.main([__file__])