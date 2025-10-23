"""
Unit tests for dry-run mode functionality in QBES.

This module tests the comprehensive dry-run validation system that validates
all simulation setup steps without executing the actual time evolution.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from qbes.simulation_engine import SimulationEngine
from qbes.core.data_models import (
    SimulationConfig, ValidationResult, MolecularSystem, 
    QuantumSubsystem, Hamiltonian
)


class TestDryRunMode:
    """Test suite for dry-run mode functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = SimulationEngine()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock PDB file
        self.test_pdb = os.path.join(self.temp_dir, "test.pdb")
        with open(self.test_pdb, 'w') as f:
            f.write("ATOM      1  N   ALA A   1      20.154  16.967  14.421  1.00 20.00           N\n")
            f.write("ATOM      2  CA  ALA A   1      19.030  16.101  14.421  1.00 20.00           C\n")
            f.write("END\n")
        
        # Create basic configuration
        self.config = SimulationConfig(
            system_pdb=self.test_pdb,
            force_field="amber14",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            noise_model_type="ohmic",
            output_directory=os.path.join(self.temp_dir, "output"),
            quantum_subsystem_selection="resname ALA"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dry_run_validation_success(self):
        """Test successful dry-run validation with all components working."""
        # Mock all the components to return successful results
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]  # 100 atoms
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]  # 10 quantum atoms
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)  # 4x4 Hermitian matrix
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.return_value = Mock()
                                
                                # Run dry-run validation
                                result = self.engine.perform_dry_run_validation(self.config)
        
        # Verify successful validation
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Verify all components were called
        mock_validate.assert_called_once_with(self.config)
        mock_parse.assert_called_once_with(self.config.system_pdb)
        mock_quantum.assert_called_once()
        mock_hamiltonian.assert_called_once()
        mock_md.assert_called_once()
        mock_noise.assert_called_once()
    
    def test_dry_run_config_validation_failure(self):
        """Test dry-run validation with configuration validation failure."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            failed_validation = ValidationResult(is_valid=False)
            failed_validation.add_error("Invalid temperature")
            mock_validate.return_value = failed_validation
            
            result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert "Invalid temperature" in result.errors
    
    def test_dry_run_pdb_file_not_found(self):
        """Test dry-run validation with missing PDB file."""
        # Use non-existent PDB file
        self.config.system_pdb = "/nonexistent/file.pdb"
        
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("PDB file not found" in error for error in result.errors)
    
    def test_dry_run_pdb_parsing_failure(self):
        """Test dry-run validation with PDB parsing failure."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_parse.side_effect = Exception("Invalid PDB format")
                
                result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("PDB parsing failed" in error for error in result.errors)
    
    def test_dry_run_quantum_subsystem_failure(self):
        """Test dry-run validation with quantum subsystem identification failure."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_quantum.side_effect = Exception("No quantum atoms found")
                    
                    result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("Quantum subsystem identification failed" in error for error in result.errors)
    
    def test_dry_run_hamiltonian_construction_failure(self):
        """Test dry-run validation with Hamiltonian construction failure."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_hamiltonian.side_effect = Exception("Hamiltonian construction failed")
                        
                        result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("Hamiltonian construction failed" in error for error in result.errors)
    
    def test_dry_run_non_hermitian_hamiltonian_warning(self):
        """Test dry-run validation with non-Hermitian Hamiltonian warning."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        # Create non-Hermitian matrix
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.array([[1, 2], [3, 4]], dtype=complex)  # Non-Hermitian
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.return_value = Mock()
                                
                                result = self.engine.perform_dry_run_validation(self.config)
        
        assert result.is_valid  # Should still be valid, just with warning
        assert any("Hamiltonian is not Hermitian" in warning for warning in result.warnings)
    
    def test_dry_run_md_system_failure(self):
        """Test dry-run validation with MD system initialization failure."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.side_effect = Exception("MD initialization failed")
                            
                            result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("MD system initialization failed" in error for error in result.errors)
    
    def test_dry_run_noise_model_failure(self):
        """Test dry-run validation with noise model creation failure."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.side_effect = Exception("Unknown noise model type")
                                
                                result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("Noise model creation failed" in error for error in result.errors)
    
    def test_dry_run_output_directory_failure(self):
        """Test dry-run validation with output directory access failure."""
        # Set output directory to a location that can't be created (Windows-compatible)
        import platform
        if platform.system() == "Windows":
            self.config.output_directory = "C:\\Windows\\System32\\forbidden_directory"
        else:
            self.config.output_directory = "/root/forbidden_directory"
        
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.return_value = Mock()
                                
                                result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("Output directory validation failed" in error for error in result.errors)
    
    def test_dry_run_invalid_simulation_parameters(self):
        """Test dry-run validation with invalid simulation parameters."""
        # Set invalid time parameters
        self.config.simulation_time = -1e-12  # Negative time
        
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.return_value = Mock()
                                
                                result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("Invalid simulation parameters" in error for error in result.errors)
    
    def test_dry_run_large_step_count_warning(self):
        """Test dry-run validation with warning for very large step count."""
        # Set parameters that result in very large step count
        self.config.simulation_time = 1e-6  # 1 microsecond
        self.config.time_step = 1e-18  # 1 attosecond (very small)
        
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.return_value = Mock()
                                
                                result = self.engine.perform_dry_run_validation(self.config)
        
        assert result.is_valid  # Should still be valid, just with warning
        assert any("Very large number of steps" in warning for warning in result.warnings)
    
    def test_dry_run_summary_generation(self):
        """Test that dry-run generates comprehensive summary."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.return_value = ValidationResult(is_valid=True)
            
            with patch.object(self.engine.config_manager, 'parse_pdb') as mock_parse:
                mock_system = Mock(spec=MolecularSystem)
                mock_system.atoms = [Mock() for _ in range(100)]
                mock_parse.return_value = mock_system
                
                with patch.object(self.engine.config_manager, 'identify_quantum_subsystem') as mock_quantum:
                    mock_subsystem = Mock(spec=QuantumSubsystem)
                    mock_subsystem.atoms = [Mock() for _ in range(10)]
                    mock_quantum.return_value = mock_subsystem
                    
                    with patch.object(self.engine.quantum_engine, 'initialize_hamiltonian') as mock_hamiltonian:
                        mock_ham = Mock(spec=Hamiltonian)
                        mock_ham.matrix = np.eye(4, dtype=complex)
                        mock_hamiltonian.return_value = mock_ham
                        
                        with patch.object(self.engine.md_engine, 'initialize_system') as mock_md:
                            mock_md.return_value = Mock()
                            
                            with patch.object(self.engine.noise_factory, 'create_noise_model') as mock_noise:
                                mock_noise.return_value = Mock()
                                
                                with patch.object(self.engine, '_print_dry_run_summary') as mock_summary:
                                    result = self.engine.perform_dry_run_validation(self.config)
                                    
                                    # Verify summary was called for successful validation
                                    if result.is_valid:
                                        mock_summary.assert_called_once()
    
    def test_dry_run_unexpected_error_handling(self):
        """Test dry-run validation handles unexpected errors gracefully."""
        with patch.object(self.engine.config_manager, 'validate_parameters') as mock_validate:
            mock_validate.side_effect = RuntimeError("Unexpected system error")
            
            result = self.engine.perform_dry_run_validation(self.config)
        
        assert not result.is_valid
        assert any("Unexpected error during dry-run validation" in error for error in result.errors)


class TestDryRunCLIIntegration:
    """Test CLI integration with dry-run mode."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock PDB file
        self.test_pdb = os.path.join(self.temp_dir, "test.pdb")
        with open(self.test_pdb, 'w') as f:
            f.write("ATOM      1  N   ALA A   1      20.154  16.967  14.421  1.00 20.00           N\n")
            f.write("ATOM      2  CA  ALA A   1      19.030  16.101  14.421  1.00 20.00           C\n")
            f.write("END\n")
        
        # Create a mock config file
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        config_content = f"""
system:
  pdb_file: {self.test_pdb}
  force_field: amber14
  
simulation:
  temperature: 300.0
  simulation_time: 1e-12
  time_step: 1e-15
  
quantum_subsystem:
  selection_method: custom
  custom_selection: resname ALA
  
noise_model:
  type: ohmic
  
output:
  directory: {os.path.join(self.temp_dir, "output")}
"""
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Create basic configuration object for mocking
        self.config = SimulationConfig(
            system_pdb=self.test_pdb,
            force_field="amber14",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            noise_model_type="ohmic",
            output_directory=os.path.join(self.temp_dir, "output"),
            quantum_subsystem_selection="resname ALA"
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('qbes.cli.ConfigurationManager')
    @patch('qbes.cli.SimulationEngine')
    def test_cli_dry_run_success(self, mock_engine_class, mock_config_manager_class):
        """Test CLI dry-run command with successful validation."""
        from click.testing import CliRunner
        from qbes.cli import run
        
        # Mock the configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config_manager.load_config.return_value = self.config
        mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
        
        # Mock the engine instance and its methods
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock successful dry-run validation
        mock_validation = ValidationResult(is_valid=True)
        mock_engine.perform_dry_run_validation.return_value = mock_validation
        
        runner = CliRunner()
        result = runner.invoke(run, [self.config_file, '--dry-run'])
        
        assert result.exit_code == 0
        assert "Dry run complete" in result.output
        mock_engine.perform_dry_run_validation.assert_called_once()
    
    @patch('qbes.cli.ConfigurationManager')
    @patch('qbes.cli.SimulationEngine')
    def test_cli_dry_run_failure(self, mock_engine_class, mock_config_manager_class):
        """Test CLI dry-run command with validation failure."""
        from click.testing import CliRunner
        from qbes.cli import run
        
        # Mock the configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config_manager.load_config.return_value = self.config
        mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
        
        # Mock the engine instance and its methods
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock failed dry-run validation
        mock_validation = ValidationResult(is_valid=False)
        mock_validation.add_error("Test validation error")
        mock_engine.perform_dry_run_validation.return_value = mock_validation
        
        runner = CliRunner()
        result = runner.invoke(run, [self.config_file, '--dry-run'])
        
        assert result.exit_code == 1
        assert "Dry-run validation failed" in result.output
        assert "Test validation error" in result.output
        mock_engine.perform_dry_run_validation.assert_called_once()
    
    @patch('qbes.cli.ConfigurationManager')
    @patch('qbes.cli.SimulationEngine')
    def test_cli_dry_run_with_warnings(self, mock_engine_class, mock_config_manager_class):
        """Test CLI dry-run command with warnings in verbose mode."""
        from click.testing import CliRunner
        from qbes.cli import run
        
        # Mock the configuration manager
        mock_config_manager = Mock()
        mock_config_manager_class.return_value = mock_config_manager
        mock_config_manager.load_config.return_value = self.config
        mock_config_manager.validate_parameters.return_value = ValidationResult(is_valid=True)
        
        # Mock the engine instance and its methods
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock successful dry-run validation with warnings
        mock_validation = ValidationResult(is_valid=True)
        mock_validation.add_warning("Test warning message")
        mock_engine.perform_dry_run_validation.return_value = mock_validation
        
        runner = CliRunner()
        result = runner.invoke(run, [self.config_file, '--dry-run', '--verbose'])
        
        assert result.exit_code == 0
        assert "Dry run complete" in result.output
        assert "Test warning message" in result.output
        mock_engine.perform_dry_run_validation.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])