"""
Unit tests for QBES configuration manager.
"""

import pytest
import tempfile
import os
import yaml
import numpy as np
from pathlib import Path

from qbes.config_manager import ConfigurationManager
from qbes.core.data_models import SimulationConfig, MolecularSystem


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample configuration data
        self.sample_config_data = {
            'system': {
                'pdb_file': 'test.pdb',
                'force_field': 'amber14',
                'solvent_model': 'tip3p',
                'ionic_strength': 0.15
            },
            'simulation': {
                'temperature': 300.0,
                'simulation_time': 1.0e-12,
                'time_step': 1.0e-15
            },
            'quantum_subsystem': {
                'selection_method': 'chromophores',
                'custom_selection': '',
                'max_quantum_atoms': 100
            },
            'noise_model': {
                'type': 'protein_ohmic',
                'coupling_strength': 1.0,
                'cutoff_frequency': 1.0e13
            },
            'output': {
                'directory': './test_output',
                'save_trajectory': True,
                'save_checkpoints': True,
                'checkpoint_interval': 1000,
                'plot_format': 'png'
            }
        }
        
        # Sample PDB content
        self.sample_pdb_content = """ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.664  16.849  14.897  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.764  18.067  15.086  1.00 20.00           O  
ATOM      5  CB  ALA A   1      18.756  15.253  13.369  1.00 20.00           C  
CONECT    1    2
CONECT    2    1    3    5
CONECT    3    2    4
END
"""
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        # Create temporary PDB file first
        pdb_path = os.path.join(self.temp_dir, 'test.pdb')
        with open(pdb_path, 'w') as f:
            f.write(self.sample_pdb_content)
        
        # Update sample config to use the test PDB file
        config_data = self.sample_config_data.copy()
        config_data['system']['pdb_file'] = pdb_path
        
        # Create temporary config file
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        config = self.config_manager.load_config(config_path)
        
        # Verify configuration
        assert isinstance(config, SimulationConfig)
        assert config.temperature == 300.0
        assert config.system_pdb == pdb_path
        assert config.force_field == 'amber14'
        assert config.quantum_subsystem_selection == 'chromophores'
        assert config.noise_model_type == 'protein_ohmic'
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            self.config_manager.load_config('nonexistent.yaml')
    
    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        config_path = os.path.join(self.temp_dir, 'invalid.yaml')
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Error parsing YAML"):
            self.config_manager.load_config(config_path)
    
    def test_validate_valid_parameters(self):
        """Test validation of valid parameters."""
        config = SimulationConfig(
            system_pdb='test.pdb',
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection='chromophores',
            noise_model_type='protein_ohmic',
            output_directory='./output'
        )
        
        # Create dummy PDB file for validation
        pdb_path = os.path.join(self.temp_dir, 'test.pdb')
        with open(pdb_path, 'w') as f:
            f.write(self.sample_pdb_content)
        config.system_pdb = pdb_path
        
        result = self.config_manager.validate_parameters(config)
        assert result.is_valid
    
    def test_validate_invalid_temperature(self):
        """Test validation with invalid temperature."""
        # Test that SimulationConfig itself validates temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SimulationConfig(
                system_pdb='test.pdb',
                temperature=-100.0,  # Invalid negative temperature
                simulation_time=1e-12,
                time_step=1e-15,
                quantum_subsystem_selection='chromophores',
                noise_model_type='protein_ohmic',
                output_directory='./output'
            )
    
    def test_validate_invalid_time_step(self):
        """Test validation with invalid time step."""
        config = SimulationConfig(
            system_pdb='test.pdb',
            temperature=300.0,
            simulation_time=1e-15,  # Smaller than time step
            time_step=1e-12,
            quantum_subsystem_selection='chromophores',
            noise_model_type='protein_ohmic',
            output_directory='./output'
        )
        
        result = self.config_manager.validate_parameters(config)
        assert not result.is_valid
        assert any("Simulation time must be greater than time step" in error for error in result.errors)
    
    def test_validate_invalid_selection_method(self):
        """Test validation with invalid quantum subsystem selection."""
        config = SimulationConfig(
            system_pdb='test.pdb',
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection='invalid_method',
            noise_model_type='protein_ohmic',
            output_directory='./output'
        )
        
        result = self.config_manager.validate_parameters(config)
        assert not result.is_valid
        assert any("Invalid quantum subsystem selection" in error for error in result.errors)
    
    def test_parse_valid_pdb(self):
        """Test parsing a valid PDB file."""
        pdb_path = os.path.join(self.temp_dir, 'test.pdb')
        with open(pdb_path, 'w') as f:
            f.write(self.sample_pdb_content)
        
        system = self.config_manager.parse_pdb(pdb_path)
        
        assert isinstance(system, MolecularSystem)
        assert len(system.atoms) == 5  # 5 atoms in sample PDB
        assert system.system_name == 'test'
        assert len(system.bonds) > 0  # Should have connectivity
        assert 1 in system.residues  # Should have residue information
    
    def test_parse_nonexistent_pdb(self):
        """Test parsing a non-existent PDB file."""
        with pytest.raises(FileNotFoundError):
            self.config_manager.parse_pdb('nonexistent.pdb')
    
    def test_identify_quantum_subsystem_chromophores(self):
        """Test quantum subsystem identification with chromophore selection."""
        # Create a molecular system with aromatic residues
        pdb_content = """ATOM      1  CG  PHE A   1      20.154  16.967  14.365  1.00 20.00           C  
ATOM      2  CD1 PHE A   1      19.030  16.101  14.618  1.00 20.00           C  
ATOM      3  CD2 PHE A   1      17.664  16.849  14.897  1.00 20.00           C  
ATOM      4  CE1 PHE A   1      17.764  18.067  15.086  1.00 20.00           C  
ATOM      5  CE2 PHE A   1      18.756  15.253  13.369  1.00 20.00           C  
END
"""
        pdb_path = os.path.join(self.temp_dir, 'chromophore.pdb')
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)
        
        system = self.config_manager.parse_pdb(pdb_path)
        subsystem = self.config_manager.identify_quantum_subsystem(system, 'chromophores')
        
        assert len(subsystem.atoms) > 0
        assert subsystem.subsystem_id == 'chromophores_subsystem'
        assert len(subsystem.basis_states) > 0
        assert subsystem.coupling_matrix.shape[0] == len(subsystem.basis_states)
    
    def test_identify_quantum_subsystem_active_site(self):
        """Test quantum subsystem identification with active site selection."""
        # Create a molecular system with catalytic residues
        pdb_content = """ATOM      1  NE2 HIS A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  SG  CYS A   2      19.030  16.101  14.618  1.00 20.00           S  
ATOM      3  OD1 ASP A   3      17.664  16.849  14.897  1.00 20.00           O  
END
"""
        pdb_path = os.path.join(self.temp_dir, 'active_site.pdb')
        with open(pdb_path, 'w') as f:
            f.write(pdb_content)
        
        system = self.config_manager.parse_pdb(pdb_path)
        subsystem = self.config_manager.identify_quantum_subsystem(system, 'active_site')
        
        assert len(subsystem.atoms) > 0
        assert subsystem.subsystem_id == 'active_site_subsystem'
    
    def test_generate_default_config(self):
        """Test generation of default configuration file."""
        output_path = os.path.join(self.temp_dir, 'default_config.yaml')
        
        success = self.config_manager.generate_default_config(output_path)
        
        assert success
        assert os.path.exists(output_path)
        
        # Verify the generated config can be loaded
        with open(output_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        assert 'system' in config_data
        assert 'simulation' in config_data
        assert 'quantum_subsystem' in config_data
    
    def test_validate_file_paths_valid(self):
        """Test file path validation with valid paths."""
        # Create test PDB file
        pdb_path = os.path.join(self.temp_dir, 'test.pdb')
        with open(pdb_path, 'w') as f:
            f.write(self.sample_pdb_content)
        
        config = SimulationConfig(
            system_pdb=pdb_path,
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection='chromophores',
            noise_model_type='protein_ohmic',
            output_directory=os.path.join(self.temp_dir, 'output')
        )
        
        result = self.config_manager.validate_file_paths(config)
        assert result.is_valid
    
    def test_validate_file_paths_invalid_pdb(self):
        """Test file path validation with invalid PDB path."""
        config = SimulationConfig(
            system_pdb='nonexistent.pdb',
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection='chromophores',
            noise_model_type='protein_ohmic',
            output_directory='./output'
        )
        
        result = self.config_manager.validate_file_paths(config)
        assert not result.is_valid
        assert any("File does not exist" in error for error in result.errors)
    
    def test_get_atomic_mass(self):
        """Test atomic mass lookup."""
        assert self.config_manager._get_atomic_mass('C') == 12.011
        assert self.config_manager._get_atomic_mass('N') == 14.007
        assert self.config_manager._get_atomic_mass('O') == 15.999
        assert self.config_manager._get_atomic_mass('H') == 1.008
        # Test unknown element defaults to carbon
        assert self.config_manager._get_atomic_mass('X') == 12.011
    
    def test_select_chromophore_atoms(self):
        """Test chromophore atom selection."""
        # Create system with aromatic residues
        from qbes.core.data_models import Atom
        atoms = [
            Atom('C', np.array([0, 0, 0]), 0.0, 12.011, 1, 1, 'PHE'),
            Atom('C', np.array([1, 0, 0]), 0.0, 12.011, 2, 1, 'PHE'),
            Atom('C', np.array([2, 0, 0]), 0.0, 12.011, 3, 2, 'ALA'),
        ]
        system = MolecularSystem(atoms=atoms, bonds=[], residues={1: 'PHE', 2: 'ALA'})
        
        chromophore_atoms = self.config_manager._select_chromophore_atoms(system)
        
        # Should select PHE atoms
        assert len(chromophore_atoms) == 2
        assert all(atom.residue_name == 'PHE' for atom in chromophore_atoms)
    
    def test_select_active_site_atoms(self):
        """Test active site atom selection."""
        from qbes.core.data_models import Atom
        atoms = [
            Atom('N', np.array([0, 0, 0]), 0.0, 14.007, 1, 1, 'HIS'),
            Atom('S', np.array([1, 0, 0]), 0.0, 32.065, 2, 2, 'CYS'),
            Atom('C', np.array([2, 0, 0]), 0.0, 12.011, 3, 3, 'ALA'),
        ]
        system = MolecularSystem(atoms=atoms, bonds=[], residues={1: 'HIS', 2: 'CYS', 3: 'ALA'})
        
        active_site_atoms = self.config_manager._select_active_site_atoms(system)
        
        # Should select HIS and CYS atoms
        assert len(active_site_atoms) == 2
        assert all(atom.residue_name in ['HIS', 'CYS'] for atom in active_site_atoms)
    
    def test_validate_config_method_exists(self):
        """Test that validate_config method exists and works."""
        config = SimulationConfig(
            system_pdb='test.pdb',
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection='chromophores',
            noise_model_type='protein_ohmic',
            output_directory='./output'
        )
        
        # Test that validate_config method exists
        assert hasattr(self.config_manager, 'validate_config')
        
        # Test that it returns a ValidationResult
        result = self.config_manager.validate_config(config)
        from qbes.core.data_models import ValidationResult
        assert isinstance(result, ValidationResult)
    
    def test_load_config_with_validation_errors(self):
        """Test loading configuration with validation errors."""
        # Create config with invalid parameters
        invalid_config_data = {
            'system': {
                'pdb_file': 'test.pdb',
                'force_field': 'amber14'
            },
            'simulation': {
                'temperature': -100.0,  # Invalid negative temperature
                'simulation_time': 1e-15,
                'time_step': 1e-12  # Time step larger than simulation time
            },
            'quantum_subsystem': {
                'selection_method': 'invalid_method'  # Invalid selection method
            },
            'noise_model': {
                'type': 'invalid_noise'  # Invalid noise model
            }
        }
        
        config_path = os.path.join(self.temp_dir, 'invalid_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config_data, f)
        
        # Should raise ValueError due to validation errors
        with pytest.raises(ValueError, match="Invalid configuration parameters"):
            self.config_manager.load_config(config_path)
    
    def test_load_config_comprehensive_logging(self):
        """Test that load_config provides comprehensive logging."""
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.sample_config_data, f)
        
        # Create dummy PDB file
        pdb_path = os.path.join(self.temp_dir, 'test.pdb')
        with open(pdb_path, 'w') as f:
            f.write(self.sample_pdb_content)
        
        # Update config to use the test PDB file
        self.sample_config_data['system']['pdb_file'] = pdb_path
        with open(config_path, 'w') as f:
            yaml.dump(self.sample_config_data, f)
        
        # Load config and verify it works
        config = self.config_manager.load_config(config_path)
        assert isinstance(config, SimulationConfig)
        assert config.system_pdb == pdb_path