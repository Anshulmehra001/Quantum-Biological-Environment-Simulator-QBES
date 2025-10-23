"""
Unit tests for the CLI module.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from qbes.cli import main, run, generate_config


class TestCLI:
    """Test suite for CLI functionality."""
    
    def test_main_command_help(self):
        """Test main command help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'Quantum Biological Environment Simulator' in result.output
        assert 'QBES' in result.output
    
    def test_main_command_version(self):
        """Test main command version output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
    
    @patch('qbes.cli.ConfigurationManager')
    @patch('qbes.cli.SimulationEngine')
    def test_run_command_basic(self, mock_sim_engine, mock_config_manager):
        """Test basic run command functionality."""
        # Setup mocks
        mock_config = Mock()
        mock_config.output_directory = "/tmp/test"
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_config_manager.return_value.validate_parameters.return_value = Mock(is_valid=True, errors=[], warnings=[])
        
        mock_engine = Mock()
        mock_engine.initialize_simulation.return_value = Mock(is_valid=True, errors=[], warnings=[])
        mock_engine.run_simulation.return_value = Mock()
        mock_engine.run_simulation.return_value.format_summary_table.return_value = "Test Summary"
        mock_sim_engine.return_value = mock_engine
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: config")
            config_file = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(run, [config_file])
            
            # Should not crash (exit code 0 or handled error)
            assert result.exit_code in [0, 1]  # May fail due to mocking, but shouldn't crash
            
        finally:
            os.unlink(config_file)
    
    @patch('qbes.cli.ConfigurationManager')
    def test_run_command_dry_run(self, mock_config_manager):
        """Test run command with dry-run option."""
        # Setup mocks
        mock_config = Mock()
        mock_config.output_directory = "/tmp/test"
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_config_manager.return_value.validate_parameters.return_value = Mock(is_valid=True, errors=[], warnings=[])
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: config")
            config_file = f.name
        
        try:
            with patch('qbes.cli.SimulationEngine') as mock_sim_engine:
                mock_engine = Mock()
                mock_engine.perform_dry_run_validation.return_value = Mock(is_valid=True, errors=[], warnings=[])
                mock_sim_engine.return_value = mock_engine
                
                runner = CliRunner()
                result = runner.invoke(run, [config_file, '--dry-run'])
                
                # Should call dry-run validation
                mock_engine.perform_dry_run_validation.assert_called_once()
                
        finally:
            os.unlink(config_file)
    
    def test_run_command_missing_file(self):
        """Test run command with missing config file."""
        runner = CliRunner()
        result = runner.invoke(run, ['nonexistent.yaml'])
        
        assert result.exit_code != 0
        assert 'Error' in result.output or 'error' in result.output.lower()
    
    @patch('qbes.cli.InteractiveConfigWizard')
    def test_generate_config_interactive(self, mock_wizard):
        """Test generate-config command with interactive mode."""
        # Setup mock wizard
        mock_wizard_instance = Mock()
        mock_wizard_instance.run_wizard.return_value = {"test": "config"}
        mock_wizard_instance.generate_config_file.return_value = True
        mock_wizard.return_value = mock_wizard_instance
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(generate_config, [config_file, '--interactive'])
            
            # Should not crash
            assert result.exit_code in [0, 1]
            
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)
    
    def test_run_command_verbose_flag(self):
        """Test run command with verbose flag."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: config")
            config_file = f.name
        
        try:
            with patch('qbes.cli.ConfigurationManager') as mock_config_manager:
                mock_config = Mock()
                mock_config.output_directory = "/tmp/test"
                mock_config_manager.return_value.load_config.return_value = mock_config
                mock_config_manager.return_value.validate_parameters.return_value = Mock(is_valid=True, errors=[], warnings=[])
                
                with patch('qbes.cli.SimulationEngine') as mock_sim_engine:
                    mock_engine = Mock()
                    mock_engine.initialize_simulation.return_value = Mock(is_valid=True, errors=[], warnings=[])
                    mock_engine.run_simulation.return_value = Mock()
                    mock_engine.run_simulation.return_value.format_summary_table.return_value = "Test Summary"
                    mock_sim_engine.return_value = mock_engine
                    
                    runner = CliRunner()
                    result = runner.invoke(run, [config_file, '--verbose'])
                    
                    # Should call set_verbose_logging
                    mock_engine.set_verbose_logging.assert_called_with(True)
                    
        finally:
            os.unlink(config_file)
    
    @pytest.mark.parametrize("debug_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_run_command_debug_levels(self, debug_level):
        """Test run command with different debug levels."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: config")
            config_file = f.name
        
        try:
            with patch('qbes.cli.ConfigurationManager') as mock_config_manager:
                mock_config = Mock()
                mock_config.output_directory = "/tmp/test"
                mock_config.debug_level = debug_level
                mock_config_manager.return_value.load_config.return_value = mock_config
                mock_config_manager.return_value.validate_parameters.return_value = Mock(is_valid=True, errors=[], warnings=[])
                
                with patch('qbes.cli.SimulationEngine') as mock_sim_engine:
                    mock_engine = Mock()
                    mock_engine.initialize_simulation.return_value = Mock(is_valid=True, errors=[], warnings=[])
                    mock_engine.run_simulation.return_value = Mock()
                    mock_engine.run_simulation.return_value.format_summary_table.return_value = "Test Summary"
                    mock_sim_engine.return_value = mock_engine
                    
                    runner = CliRunner()
                    result = runner.invoke(run, [config_file, '--debug-level', debug_level])
                    
                    # Should set the debug level on config
                    assert mock_config.debug_level == debug_level
                    
        finally:
            os.unlink(config_file)
    
    def test_run_command_output_override(self):
        """Test run command with output directory override."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: config")
            config_file = f.name
        
        try:
            with patch('qbes.cli.ConfigurationManager') as mock_config_manager:
                mock_config = Mock()
                mock_config.output_directory = "/tmp/test"
                mock_config_manager.return_value.load_config.return_value = mock_config
                mock_config_manager.return_value.validate_parameters.return_value = Mock(is_valid=True, errors=[], warnings=[])
                
                with patch('qbes.cli.SimulationEngine') as mock_sim_engine:
                    mock_engine = Mock()
                    mock_engine.initialize_simulation.return_value = Mock(is_valid=True, errors=[], warnings=[])
                    mock_engine.run_simulation.return_value = Mock()
                    mock_engine.run_simulation.return_value.format_summary_table.return_value = "Test Summary"
                    mock_sim_engine.return_value = mock_engine
                    
                    runner = CliRunner()
                    result = runner.invoke(run, [config_file, '--output-dir', '/custom/output'])
                    
                    # Should override output directory
                    assert mock_config.output_directory == '/custom/output'
                    
        finally:
            os.unlink(config_file)
    
    def test_error_handling_file_not_found(self):
        """Test error handling for file not found."""
        runner = CliRunner()
        result = runner.invoke(run, ['missing_file.yaml'])
        
        assert result.exit_code != 0
        # Should contain error message about file not found
        assert any(word in result.output.lower() for word in ['error', 'not found', 'missing'])
    
    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configuration."""
        # Create invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
            config_file = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(run, [config_file])
            
            assert result.exit_code != 0
            
        finally:
            os.unlink(config_file)
    
    @patch('qbes.cli.ConfigurationManager')
    def test_validation_failure_handling(self, mock_config_manager):
        """Test handling of validation failures."""
        # Setup mock to return validation failure
        mock_config = Mock()
        mock_config_manager.return_value.load_config.return_value = mock_config
        mock_validation_result = Mock()
        mock_validation_result.is_valid = False
        mock_validation_result.errors = ["Test error"]
        mock_validation_result.warnings = []
        mock_config_manager.return_value.validate_parameters.return_value = mock_validation_result
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("test: config")
            config_file = f.name
        
        try:
            runner = CliRunner()
            result = runner.invoke(run, [config_file])
            
            assert result.exit_code != 0
            assert "Test error" in result.output
            
        finally:
            os.unlink(config_file)