"""
Configuration management for QBES simulations.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from .core.interfaces import ConfigurationManagerInterface
from .core.data_models import (
    SimulationConfig, MolecularSystem, QuantumSubsystem, ValidationResult, 
    Atom, QuantumState
)
from .utils.validation import ValidationUtils


class ConfigurationManager(ConfigurationManagerInterface):
    """
    Handles configuration loading, validation, and PDB file parsing.
    
    This class implements the configuration management interface and provides
    methods for loading simulation parameters, validating inputs, and parsing
    molecular structure files.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.current_config = None
        self.molecular_system = None
        self.validation_utils = ValidationUtils()
    
    def load_config(self, config_path: str) -> SimulationConfig:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Extract configuration parameters from nested YAML structure
            system_config = config_data.get('system', {})
            sim_config = config_data.get('simulation', {})
            quantum_config = config_data.get('quantum_subsystem', {})
            noise_config = config_data.get('noise_model', {})
            output_config = config_data.get('output', {})
            
            # Create SimulationConfig object
            config = SimulationConfig(
                system_pdb=system_config.get('pdb_file', 'system.pdb'),
                temperature=sim_config.get('temperature', 300.0),
                simulation_time=sim_config.get('simulation_time', 1e-12),
                time_step=sim_config.get('time_step', 1e-15),
                quantum_subsystem_selection=quantum_config.get('selection_method', 'chromophores'),
                noise_model_type=noise_config.get('type', 'protein_ohmic'),
                output_directory=output_config.get('directory', './qbes_output'),
                force_field=system_config.get('force_field', 'amber14'),
                solvent_model=system_config.get('solvent_model', 'tip3p'),
                ionic_strength=system_config.get('ionic_strength', 0.15)
            )
            
            self.current_config = config
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def validate_parameters(self, config: SimulationConfig) -> ValidationResult:
        """Validate all configuration parameters against physical constraints."""
        result = ValidationResult(is_valid=True)
        
        # Validate basic physical parameters
        params = {
            'temperature': config.temperature,
            'time_step': config.time_step
        }
        physical_validation = self.validation_utils.validate_physical_parameters(params)
        result.errors.extend(physical_validation.errors)
        result.warnings.extend(physical_validation.warnings)
        if not physical_validation.is_valid:
            result.is_valid = False
        
        # Validate simulation time
        if config.simulation_time <= 0:
            result.add_error("Simulation time must be positive")
        elif config.simulation_time < config.time_step:
            result.add_error("Simulation time must be greater than time step")
        
        # Validate time step relative to simulation time
        if config.simulation_time / config.time_step > 1e8:
            result.add_warning("Very large number of time steps - simulation may be slow")
        
        # Validate file paths
        file_validation = self.validate_file_paths(config)
        result.errors.extend(file_validation.errors)
        result.warnings.extend(file_validation.warnings)
        if not file_validation.is_valid:
            result.is_valid = False
        
        # Validate selection criteria
        valid_selections = ['chromophores', 'active_site', 'custom', 'all']
        if config.quantum_subsystem_selection not in valid_selections:
            result.add_error(f"Invalid quantum subsystem selection: {config.quantum_subsystem_selection}")
        
        # Validate noise model type
        valid_noise_models = ['protein_ohmic', 'membrane', 'solvent_ionic', 'custom']
        if config.noise_model_type not in valid_noise_models:
            result.add_error(f"Invalid noise model type: {config.noise_model_type}")
        
        # Validate force field
        valid_force_fields = ['amber14', 'amber99sb', 'charmm36', 'opls']
        if config.force_field not in valid_force_fields:
            result.add_warning(f"Uncommon force field: {config.force_field}")
        
        # Validate ionic strength
        if config.ionic_strength < 0:
            result.add_error("Ionic strength cannot be negative")
        elif config.ionic_strength > 5.0:
            result.add_warning("Very high ionic strength (>5M)")
        
        return result
    
    def parse_pdb(self, pdb_path: str) -> MolecularSystem:
        """Parse PDB file and extract molecular structure."""
        try:
            atoms = []
            bonds = []
            residues = {}
            
            with open(pdb_path, 'r') as file:
                atom_id = 0
                for line in file:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        # Parse PDB ATOM/HETATM record
                        atom_id += 1
                        element = line[76:78].strip() or line[12:14].strip()[0]
                        
                        # Extract coordinates
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        position = np.array([x, y, z])
                        
                        # Extract other properties
                        residue_id = int(line[22:26])
                        residue_name = line[17:20].strip()
                        
                        # Estimate charge and mass based on element
                        charge = 0.0  # Simplified - would need force field parameters
                        mass = self._get_atomic_mass(element)
                        
                        atom = Atom(
                            element=element,
                            position=position,
                            charge=charge,
                            mass=mass,
                            atom_id=atom_id,
                            residue_id=residue_id,
                            residue_name=residue_name
                        )
                        atoms.append(atom)
                        residues[residue_id] = residue_name
                    
                    elif line.startswith('CONECT'):
                        # Parse connectivity information
                        parts = line.split()
                        if len(parts) > 2:
                            atom1 = int(parts[1])
                            for i in range(2, len(parts)):
                                atom2 = int(parts[i])
                                bonds.append((atom1 - 1, atom2 - 1))  # Convert to 0-based indexing
            
            # Calculate total charge
            total_charge = sum(atom.charge for atom in atoms)
            
            system = MolecularSystem(
                atoms=atoms,
                bonds=bonds,
                residues=residues,
                system_name=Path(pdb_path).stem,
                total_charge=total_charge
            )
            
            self.molecular_system = system
            return system
            
        except FileNotFoundError:
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        except Exception as e:
            raise RuntimeError(f"Error parsing PDB file: {e}")
    
    def identify_quantum_subsystem(self, system: MolecularSystem, 
                                  selection_criteria: str) -> QuantumSubsystem:
        """Identify and extract quantum subsystem from molecular structure."""
        quantum_atoms = []
        
        if selection_criteria == "chromophores":
            # Select atoms that are likely chromophores (aromatic systems)
            quantum_atoms = self._select_chromophore_atoms(system)
        elif selection_criteria == "active_site":
            # Select atoms in enzyme active sites (simplified heuristic)
            quantum_atoms = self._select_active_site_atoms(system)
        elif selection_criteria == "all":
            # Use all atoms (for small systems)
            quantum_atoms = system.atoms[:100]  # Limit to first 100 for computational feasibility
        else:
            # Default to first few atoms
            quantum_atoms = system.atoms[:10]
        
        # Create basic quantum states (ground and excited for each atom)
        basis_states = []
        basis_labels = []
        for i, atom in enumerate(quantum_atoms):
            # Create ground state for this atom
            coeffs_ground = np.zeros(len(quantum_atoms) * 2)
            coeffs_ground[2*i] = 1.0
            basis_labels.extend([f"{atom.element}_{i}_ground", f"{atom.element}_{i}_excited"])
        
        # Create two basis states: all ground and one excitation
        if quantum_atoms:
            # All ground state
            ground_coeffs = np.zeros(len(quantum_atoms) * 2)
            ground_coeffs[::2] = 1.0 / np.sqrt(len(quantum_atoms))
            basis_states.append(QuantumState(ground_coeffs, basis_labels))
            
            # Single excitation state
            excited_coeffs = np.zeros(len(quantum_atoms) * 2)
            excited_coeffs[1] = 1.0  # First atom excited
            basis_states.append(QuantumState(excited_coeffs, basis_labels))
        
        # Create coupling matrix (simplified)
        n_states = len(basis_states)
        coupling_matrix = np.zeros((n_states, n_states))
        if n_states > 1:
            coupling_matrix[0, 1] = coupling_matrix[1, 0] = 0.1  # Weak coupling
        
        subsystem = QuantumSubsystem(
            atoms=quantum_atoms,
            hamiltonian_parameters={"coupling": 0.1, "energy_gap": 2.0},
            coupling_matrix=coupling_matrix,
            basis_states=basis_states,
            subsystem_id=f"{selection_criteria}_subsystem"
        )
        
        return subsystem
    
    def generate_default_config(self, output_path: str) -> bool:
        """Generate default configuration file template."""
        try:
            default_config = {
                'system': {
                    'pdb_file': 'system.pdb',
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
                    'directory': './qbes_output',
                    'save_trajectory': True,
                    'save_checkpoints': True,
                    'checkpoint_interval': 1000,
                    'plot_format': 'png'
                }
            }
            
            with open(output_path, 'w') as file:
                yaml.dump(default_config, file, default_flow_style=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error generating default config: {e}")
            return False
    
    def validate_file_paths(self, config: SimulationConfig) -> ValidationResult:
        """Validate that all required files exist and are accessible."""
        result = ValidationResult(is_valid=True)
        
        # Validate PDB file
        if config.system_pdb:
            pdb_validation = self.validation_utils.validate_file_exists(config.system_pdb)
            if not pdb_validation.is_valid:
                result.errors.extend(pdb_validation.errors)
                result.is_valid = False
        
        # Validate output directory (create if doesn't exist)
        try:
            os.makedirs(config.output_directory, exist_ok=True)
        except Exception as e:
            result.add_error(f"Cannot create output directory: {e}")
        
        return result
    
    def _get_atomic_mass(self, element: str) -> float:
        """Get atomic mass for common elements."""
        masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
            'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'K': 39.098,
            'Ca': 40.078, 'Fe': 55.845, 'Zn': 65.38, 'Mg': 24.305
        }
        return masses.get(element, 12.011)  # Default to carbon mass
    
    def _select_chromophore_atoms(self, system: MolecularSystem) -> List[Atom]:
        """Select atoms that are likely part of chromophore systems."""
        chromophore_atoms = []
        
        # Look for aromatic residues and cofactors
        chromophore_residues = ['PHE', 'TRP', 'TYR', 'HIS', 'HEM', 'CHL', 'BCL']
        
        for atom in system.atoms:
            if atom.residue_name in chromophore_residues:
                chromophore_atoms.append(atom)
        
        # If no chromophores found, select carbon atoms (likely aromatic)
        if not chromophore_atoms:
            chromophore_atoms = [atom for atom in system.atoms if atom.element == 'C'][:20]
        
        return chromophore_atoms[:50]  # Limit to 50 atoms
    
    def _select_active_site_atoms(self, system: MolecularSystem) -> List[Atom]:
        """Select atoms likely to be in enzyme active sites."""
        active_site_atoms = []
        
        # Look for catalytic residues
        catalytic_residues = ['HIS', 'CYS', 'ASP', 'GLU', 'SER', 'THR']
        
        for atom in system.atoms:
            if atom.residue_name in catalytic_residues:
                active_site_atoms.append(atom)
        
        # If no catalytic residues found, select heteroatoms
        if not active_site_atoms:
            active_site_atoms = [atom for atom in system.atoms 
                               if atom.element in ['N', 'O', 'S', 'P']][:20]
        
        return active_site_atoms[:30]  # Limit to 30 atoms