"""
Tests for MD engine functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from qbes.md_engine import MDEngine
from qbes.core.data_models import MolecularSystem, Atom, SpectralDensity


class TestMDEngine:
    """Test suite for MDEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = None
        
    def teardown_method(self):
        """Clean up after tests."""
        if self.engine is not None:
            del self.engine
    
    @pytest.fixture
    def mock_openmm(self):
        """Mock OpenMM modules for testing without OpenMM dependency."""
        with patch.dict('sys.modules', {
            'openmm': MagicMock(),
            'openmm.app': MagicMock(),
            'openmm.unit': MagicMock()
        }):
            yield
    
    def test_engine_initialization_cpu_platform(self):
        """Test MD engine initialization with CPU platform."""
        try:
            engine = MDEngine(platform_name="CPU")
            assert engine is not None
            assert engine.system is None
            assert engine.topology is None
            assert engine.positions is None
            assert engine.simulation is None
            assert engine.trajectory_data is None
            assert engine.force_field is None
            
            # Check available force fields and water models
            assert 'amber14' in engine.available_force_fields
            assert 'tip3p' in engine.available_water_models
            
        except ImportError:
            # Skip test if OpenMM not available
            pytest.skip("OpenMM not available")
    
    def test_engine_initialization_fallback_platform(self):
        """Test MD engine initialization with fallback to CPU platform."""
        try:
            # Try to initialize with non-existent platform
            engine = MDEngine(platform_name="NonExistentPlatform")
            # Should fallback to CPU
            assert engine is not None
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_create_water_box(self):
        """Test creation of water box system."""
        try:
            engine = MDEngine()
            
            # Mock the OpenMM components
            with patch('qbes.md_engine.app') as mock_app:
                mock_topology = Mock()
                mock_modeller = Mock()
                mock_forcefield = Mock()
                
                mock_app.Topology.return_value = mock_topology
                mock_app.Modeller.return_value = mock_modeller
                mock_app.ForceField.return_value = mock_forcefield
                
                # Mock positions as a simple array
                mock_positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
                mock_modeller.positions = mock_positions
                mock_modeller.topology = mock_topology
                
                # Mock topology atoms
                mock_atoms = []
                for i in range(3):
                    mock_atom = Mock()
                    mock_atom.element.symbol = 'O' if i == 0 else 'H'
                    mock_atom.element.mass.value_in_unit.return_value = 15.999 if i == 0 else 1.008
                    mock_atom.residue.id = 0
                    mock_atom.residue.name = 'HOH'
                    mock_atoms.append(mock_atom)
                
                mock_topology.atoms.return_value = mock_atoms
                mock_topology.bonds.return_value = [(mock_atoms[0], mock_atoms[1]), (mock_atoms[0], mock_atoms[2])]
                
                # Mock system creation
                mock_system = Mock()
                mock_forcefield.createSystem.return_value = mock_system
                
                # Test water box creation
                molecular_system = engine.create_water_box(box_size=1.0, water_model="tip3p")
                
                assert isinstance(molecular_system, MolecularSystem)
                assert len(molecular_system.atoms) == 3
                assert molecular_system.atoms[0].element == 'O'
                assert molecular_system.atoms[1].element == 'H'
                assert molecular_system.atoms[2].element == 'H'
                
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_create_simple_protein_system(self):
        """Test creation of simple protein system."""
        try:
            engine = MDEngine()
            
            # Test simple protein creation
            molecular_system = engine.create_simple_protein_system("ALA-GLY")
            
            assert isinstance(molecular_system, MolecularSystem)
            assert len(molecular_system.atoms) == 8  # 4 atoms per residue * 2 residues
            assert len(molecular_system.residues) == 2
            assert molecular_system.residues[0] == "ALA"
            assert molecular_system.residues[1] == "GLY"
            
            # Check backbone atoms are present
            backbone_elements = [atom.element for atom in molecular_system.atoms]
            assert 'N' in backbone_elements
            assert 'C' in backbone_elements
            assert 'O' in backbone_elements
            
            # Check bonds are created
            assert len(molecular_system.bonds) > 0
            
            # Test with longer sequence
            longer_system = engine.create_simple_protein_system("ALA-GLY-ALA-VAL")
            assert len(longer_system.atoms) == 16  # 4 atoms per residue * 4 residues
            assert len(longer_system.residues) == 4
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_create_test_chromophore_system(self):
        """Test creation of chromophore system for quantum studies."""
        try:
            engine = MDEngine()
            
            # Test chromophore system creation
            molecular_system = engine.create_test_chromophore_system(n_chromophores=2)
            
            assert isinstance(molecular_system, MolecularSystem)
            assert len(molecular_system.residues) == 2
            
            # Check for Mg centers (one per chromophore)
            mg_atoms = [atom for atom in molecular_system.atoms if atom.element == 'Mg']
            assert len(mg_atoms) == 2
            
            # Check for N atoms (4 per chromophore)
            n_atoms = [atom for atom in molecular_system.atoms if atom.element == 'N']
            assert len(n_atoms) == 8  # 4 N per chromophore * 2 chromophores
            
            # Check bonds exist
            assert len(molecular_system.bonds) > 0
            
            # Test with different number of chromophores
            single_chrom = engine.create_test_chromophore_system(n_chromophores=1)
            assert len(single_chrom.residues) == 1
            
            triple_chrom = engine.create_test_chromophore_system(n_chromophores=3)
            assert len(triple_chrom.residues) == 3
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_extract_quantum_parameters(self):
        """Test extraction of quantum parameters from trajectory."""
        try:
            engine = MDEngine()
            
            # Create mock trajectory data
            n_frames = 10
            n_atoms = 5
            positions = np.random.rand(n_frames, n_atoms, 3) * 2.0  # Random positions in nm
            
            trajectory = {
                'positions': positions,
                'times': np.linspace(0, 1, n_frames),
                'energies': np.random.rand(n_frames) * 1000
            }
            
            quantum_atoms = [0, 1, 2]  # First 3 atoms
            
            # Extract parameters
            parameters = engine.extract_quantum_parameters(trajectory, quantum_atoms)
            
            # Check returned parameters
            assert 'distances' in parameters
            assert 'angles' in parameters
            assert 'dihedrals' in parameters
            assert 'coupling_fluctuations' in parameters
            
            # Check shapes
            assert parameters['distances'].shape == (n_frames, 3, 3)
            assert parameters['coupling_fluctuations'].shape == (n_frames, 3, 3)
            
            # Check that distances are symmetric and positive
            for frame in range(n_frames):
                dist_matrix = parameters['distances'][frame]
                assert np.allclose(dist_matrix, dist_matrix.T)  # Symmetric
                assert np.all(dist_matrix >= 0)  # Non-negative
                assert np.all(np.diag(dist_matrix) == 0)  # Zero diagonal
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_extract_quantum_parameters_invalid_atoms(self):
        """Test quantum parameter extraction with invalid atom indices."""
        try:
            engine = MDEngine()
            
            # Create mock trajectory data
            n_frames = 5
            n_atoms = 3
            positions = np.random.rand(n_frames, n_atoms, 3)
            
            trajectory = {
                'positions': positions,
                'times': np.linspace(0, 1, n_frames),
                'energies': np.random.rand(n_frames)
            }
            
            # Invalid quantum atoms (index out of range)
            quantum_atoms = [0, 1, 5]  # Index 5 doesn't exist
            
            with pytest.raises(ValueError, match="exceeds system size"):
                engine.extract_quantum_parameters(trajectory, quantum_atoms)
                
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_calculate_spectral_density(self):
        """Test spectral density calculation from fluctuations."""
        try:
            engine = MDEngine()
            
            # Create test fluctuation data (sinusoidal with known frequency)
            n_points = 1000
            time_step = 0.001  # ps
            times = np.arange(n_points) * time_step
            
            # Create signal with known frequency components
            freq1 = 100.0  # cm^-1 (converted to appropriate units)
            freq2 = 200.0  # cm^-1
            
            # Convert cm^-1 to Hz for signal generation
            c_light = 2.998e10  # cm/s
            freq1_hz = freq1 * c_light * 100  # Convert cm^-1 to Hz
            freq2_hz = freq2 * c_light * 100
            
            fluctuations = (np.sin(2 * np.pi * freq1_hz * times * 1e-12) + 
                          0.5 * np.sin(2 * np.pi * freq2_hz * times * 1e-12))
            
            # Calculate spectral density
            spectral_density = engine.calculate_spectral_density(fluctuations, time_step)
            
            assert isinstance(spectral_density, SpectralDensity)
            assert len(spectral_density.frequencies) > 0
            assert len(spectral_density.spectral_values) == len(spectral_density.frequencies)
            assert spectral_density.temperature == 300.0
            assert spectral_density.spectral_type == "md_derived"
            
            # Check that frequencies are positive
            assert np.all(spectral_density.frequencies >= 0)
            
            # Check that spectral values are non-negative
            assert np.all(spectral_density.spectral_values >= 0)
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_calculate_spectral_density_empty_input(self):
        """Test spectral density calculation with empty input."""
        try:
            engine = MDEngine()
            
            empty_fluctuations = np.array([])
            
            with pytest.raises(ValueError, match="cannot be empty"):
                engine.calculate_spectral_density(empty_fluctuations, 0.001)
                
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_analyze_parameter_fluctuations(self):
        """Test statistical analysis of parameter fluctuations."""
        try:
            engine = MDEngine()
            
            # Create test parameter data
            n_frames = 100
            n_atoms = 3
            
            # Create mock parameters with known statistics
            distances = np.random.normal(1.0, 0.1, (n_frames, n_atoms, n_atoms))
            angles = np.random.normal(np.pi/2, 0.2, (n_frames, n_atoms-2))
            coupling = np.random.exponential(50.0, (n_frames, n_atoms, n_atoms))
            
            parameters = {
                'distances': distances,
                'angles': angles,
                'coupling_fluctuations': coupling
            }
            
            # Analyze fluctuations
            analysis = engine.analyze_parameter_fluctuations(parameters, time_step=0.001)
            
            # Check that analysis contains expected keys
            assert 'distances' in analysis
            assert 'angles' in analysis
            assert 'coupling_fluctuations' in analysis
            
            # Check statistical measures
            for param_name, stats in analysis.items():
                assert 'mean' in stats
                assert 'std' in stats
                assert 'min' in stats
                assert 'max' in stats
                assert 'variance' in stats
                
                # Check that values are reasonable
                assert stats['std'] >= 0
                assert stats['variance'] >= 0
                assert stats['min'] <= stats['mean'] <= stats['max']
            
            # Check correlation time calculation
            if 'correlation_time_ps' in analysis['distances']:
                assert analysis['distances']['correlation_time_ps'] >= 0
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_extract_coupling_matrix_evolution(self):
        """Test extraction of coupling matrix time evolution."""
        try:
            engine = MDEngine()
            
            # Create mock trajectory data
            n_frames = 20
            n_atoms = 5
            positions = np.random.rand(n_frames, n_atoms, 3) * 2.0  # Random positions
            
            trajectory = {
                'positions': positions,
                'times': np.linspace(0, 1, n_frames),
                'energies': np.random.rand(n_frames) * 1000
            }
            
            quantum_atoms = [0, 1, 2]
            
            # Test exponential coupling model
            coupling_matrices = engine.extract_coupling_matrix_evolution(
                trajectory, quantum_atoms, coupling_model="exponential"
            )
            
            # Check shape
            assert coupling_matrices.shape == (n_frames, 3, 3)
            
            # Check that matrices are symmetric
            for frame in range(n_frames):
                matrix = coupling_matrices[frame]
                assert np.allclose(matrix, matrix.T)
                
                # Check diagonal is zero (no self-coupling)
                assert np.allclose(np.diag(matrix), 0.0)
                
                # Check all values are non-negative
                assert np.all(matrix >= 0)
            
            # Test power law coupling model
            coupling_matrices_pl = engine.extract_coupling_matrix_evolution(
                trajectory, quantum_atoms, coupling_model="power_law"
            )
            
            assert coupling_matrices_pl.shape == (n_frames, 3, 3)
            
            # Test invalid coupling model
            with pytest.raises(ValueError, match="Unknown coupling model"):
                engine.extract_coupling_matrix_evolution(
                    trajectory, quantum_atoms, coupling_model="invalid_model"
                )
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_autocorrelation_calculation(self):
        """Test autocorrelation function calculation."""
        try:
            engine = MDEngine()
            
            # Create test signal with known autocorrelation
            n_points = 100
            t = np.linspace(0, 10, n_points)
            
            # Exponentially decaying signal
            signal = np.exp(-t / 2.0) + 0.1 * np.random.randn(n_points)
            
            # Calculate autocorrelation
            autocorr = engine._calculate_autocorrelation(signal)
            
            # Check properties
            assert len(autocorr) == len(signal)
            assert autocorr[0] == 1.0  # Normalized to 1 at t=0
            assert np.all(autocorr <= 1.0)  # Should not exceed 1
            
            # For exponentially decaying signal, autocorr should also decay
            assert autocorr[10] < autocorr[0]
            assert autocorr[50] < autocorr[10]
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_correlation_time_calculation(self):
        """Test correlation time calculation."""
        try:
            engine = MDEngine()
            
            # Create autocorrelation that decays to 1/e
            n_points = 100
            time_step = 0.01
            
            # Exponential decay: exp(-t/tau) where tau is correlation time
            tau_expected = 2.0  # Expected correlation time
            autocorr = np.exp(-np.arange(n_points) * time_step / tau_expected)
            
            # Calculate correlation time
            tau_calculated = engine._find_correlation_time(autocorr, time_step)
            
            # Should be close to expected value
            assert abs(tau_calculated - tau_expected) < 0.1
            
            # Test with autocorr that never drops below 1/e
            constant_autocorr = np.ones(n_points) * 0.5  # Constant at 0.5
            tau_constant = engine._find_correlation_time(constant_autocorr, time_step)
            assert tau_constant >= 0  # Should return some reasonable value
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_validate_system_uninitialized(self):
        """Test system validation for uninitialized system."""
        try:
            engine = MDEngine()
            
            result = engine.validate_system()
            
            assert not result.is_valid
            assert "system not initialized" in str(result.errors).lower()
            assert "topology not loaded" in str(result.errors).lower()
            assert "positions not set" in str(result.errors).lower()
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    @patch('qbes.md_engine.HAS_OPENMM', False)
    def test_engine_initialization_no_openmm(self):
        """Test MD engine initialization when OpenMM is not available."""
        with pytest.raises(ImportError, match="OpenMM is required"):
            MDEngine()
    
    def test_force_field_validation(self):
        """Test force field validation."""
        try:
            engine = MDEngine()
            
            # Test with valid force field
            assert 'amber14' in engine.available_force_fields
            
            # Test with invalid force field - would be caught in initialize_system
            with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as f:
                # Write minimal PDB content
                f.write("ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N\n")
                f.write("ATOM      2  CA  ALA A   1      19.030  16.967  15.265  1.00 20.00           C\n")
                f.write("END\n")
                pdb_file = f.name
            
            try:
                with pytest.raises(ValueError, match="Unsupported force field"):
                    engine.initialize_system(pdb_file, "invalid_force_field")
            finally:
                os.unlink(pdb_file)
                
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_water_model_validation(self):
        """Test water model validation."""
        try:
            engine = MDEngine()
            
            # Test with valid water model
            assert 'tip3p' in engine.available_water_models
            
            # Test with invalid water model
            mock_system = Mock()
            with pytest.raises(ValueError, match="Unsupported solvent model"):
                engine.setup_environment(mock_system, "invalid_water_model", 0.15)
                
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_trajectory_data_structure(self):
        """Test that trajectory data has correct structure."""
        try:
            engine = MDEngine()
            
            # Mock a simple trajectory run
            with patch.object(engine, 'system', Mock()):
                with patch.object(engine, 'topology', Mock()):
                    with patch.object(engine, 'positions', Mock()):
                        with patch('qbes.md_engine.app.Simulation') as mock_sim_class:
                            mock_sim = Mock()
                            mock_sim_class.return_value = mock_sim
                            
                            # Mock state returns
                            mock_states = []
                            for i in range(5):  # 5 frames
                                mock_state = Mock()
                                mock_positions = np.random.rand(10, 3)  # 10 atoms
                                mock_state.getPositions.return_value.value_in_unit.return_value = mock_positions
                                mock_state.getPotentialEnergy.return_value.value_in_unit.return_value = 1000.0 + i
                                mock_states.append(mock_state)
                            
                            mock_sim.context.getState.side_effect = mock_states
                            
                            # Run trajectory
                            trajectory = engine.run_trajectory(duration=0.1, time_step=1.0, temperature=300.0)
                            
                            # Check structure
                            assert 'positions' in trajectory
                            assert 'times' in trajectory
                            assert 'energies' in trajectory
                            
                            assert len(trajectory['positions']) == len(trajectory['times'])
                            assert len(trajectory['times']) == len(trajectory['energies'])
                            
        except ImportError:
            pytest.skip("OpenMM not available")


class TestMDEngineUtilities:
    """Test MD engine utility methods that don't require OpenMM."""
    
    def test_simple_protein_system_without_openmm(self):
        """Test simple protein system creation without OpenMM dependency."""
        # Create engine instance without OpenMM
        with patch('qbes.md_engine.HAS_OPENMM', False):
            # This should work since create_simple_protein_system doesn't use OpenMM
            engine = MDEngine.__new__(MDEngine)  # Create without calling __init__
            
            # Initialize minimal attributes
            engine.system = None
            engine.topology = None
            engine.positions = None
            engine.simulation = None
            engine.trajectory_data = None
            engine.force_field = None
            
            # Test protein system creation
            molecular_system = engine.create_simple_protein_system("ALA-GLY")
            
            assert isinstance(molecular_system, MolecularSystem)
            assert len(molecular_system.atoms) == 8
            assert len(molecular_system.residues) == 2
            assert molecular_system.system_name == "peptide_ALA-GLY"
    
    def test_chromophore_system_without_openmm(self):
        """Test chromophore system creation without OpenMM dependency."""
        with patch('qbes.md_engine.HAS_OPENMM', False):
            engine = MDEngine.__new__(MDEngine)
            
            # Test chromophore system creation
            molecular_system = engine.create_test_chromophore_system(n_chromophores=2)
            
            assert isinstance(molecular_system, MolecularSystem)
            assert len(molecular_system.residues) == 2
            
            # Check for expected atoms
            mg_atoms = [atom for atom in molecular_system.atoms if atom.element == 'Mg']
            assert len(mg_atoms) == 2
    
    def test_parameter_analysis_without_openmm(self):
        """Test parameter analysis methods without OpenMM dependency."""
        with patch('qbes.md_engine.HAS_OPENMM', False):
            engine = MDEngine.__new__(MDEngine)
            
            # Create test data
            n_frames = 50
            distances = np.random.normal(1.0, 0.1, (n_frames, 3, 3))
            parameters = {'distances': distances}
            
            # Test analysis
            analysis = engine.analyze_parameter_fluctuations(parameters, 0.001)
            
            assert 'distances' in analysis
            assert 'mean' in analysis['distances']
            assert 'std' in analysis['distances']
    
    def test_coupling_matrix_without_openmm(self):
        """Test coupling matrix extraction without OpenMM dependency."""
        with patch('qbes.md_engine.HAS_OPENMM', False):
            engine = MDEngine.__new__(MDEngine)
            
            # Create mock trajectory
            n_frames = 10
            n_atoms = 4
            positions = np.random.rand(n_frames, n_atoms, 3) * 2.0
            
            trajectory = {
                'positions': positions,
                'times': np.linspace(0, 1, n_frames),
                'energies': np.random.rand(n_frames)
            }
            
            quantum_atoms = [0, 1, 2]
            
            # Test coupling matrix extraction
            coupling_matrices = engine.extract_coupling_matrix_evolution(
                trajectory, quantum_atoms, "exponential"
            )
            
            assert coupling_matrices.shape == (n_frames, 3, 3)
            
            # Check symmetry
            for frame in range(n_frames):
                matrix = coupling_matrices[frame]
                assert np.allclose(matrix, matrix.T)


class TestMDEngineIntegration:
    """Integration tests for MD engine with mock systems."""
    
    def test_full_workflow_water_box(self):
        """Test complete workflow with water box."""
        try:
            engine = MDEngine()
            
            # Create water box
            with patch('qbes.md_engine.app') as mock_app:
                # Mock the water box creation
                self._setup_water_box_mocks(mock_app)
                
                molecular_system = engine.create_water_box(box_size=1.0)
                
                # Validate system
                with patch.object(engine, 'system', Mock()):
                    with patch.object(engine, 'topology', Mock()):
                        with patch.object(engine, 'positions', Mock()):
                            result = engine.validate_system()
                            assert result.is_valid
                
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def test_complete_parameter_extraction_workflow(self):
        """Test complete parameter extraction and analysis workflow."""
        try:
            engine = MDEngine()
            
            # Create test system
            molecular_system = engine.create_test_chromophore_system(n_chromophores=2)
            
            # Create mock trajectory data
            n_frames = 50
            n_atoms = len(molecular_system.atoms)
            
            # Generate realistic trajectory with some structure
            positions = np.zeros((n_frames, n_atoms, 3))
            for frame in range(n_frames):
                for atom_idx, atom in enumerate(molecular_system.atoms):
                    # Add small fluctuations around initial positions
                    noise = np.random.normal(0, 0.05, 3)  # 0.05 nm fluctuations
                    positions[frame, atom_idx] = atom.position + noise
            
            trajectory = {
                'positions': positions,
                'times': np.linspace(0, 5.0, n_frames),  # 5 ps simulation
                'energies': np.random.normal(-1000, 50, n_frames)  # Realistic energies
            }
            
            # Extract quantum parameters for Mg atoms (chromophore centers)
            mg_indices = [i for i, atom in enumerate(molecular_system.atoms) 
                         if atom.element == 'Mg']
            
            parameters = engine.extract_quantum_parameters(trajectory, mg_indices)
            
            # Analyze fluctuations
            analysis = engine.analyze_parameter_fluctuations(parameters, 0.1)  # 0.1 ps time step
            
            # Extract coupling matrix evolution
            coupling_matrices = engine.extract_coupling_matrix_evolution(
                trajectory, mg_indices, "exponential"
            )
            
            # Verify results
            assert len(parameters) > 0
            assert len(analysis) > 0
            assert coupling_matrices.shape[0] == n_frames
            assert coupling_matrices.shape[1] == len(mg_indices)
            
            # Check that coupling varies over time (should not be constant)
            coupling_variance = np.var(coupling_matrices[:, 0, 1])  # Off-diagonal element
            assert coupling_variance > 0  # Should have some variation
            
        except ImportError:
            pytest.skip("OpenMM not available")
    
    def _setup_water_box_mocks(self, mock_app):
        """Helper to set up mocks for water box creation."""
        mock_topology = Mock()
        mock_modeller = Mock()
        mock_forcefield = Mock()
        
        mock_app.Topology.return_value = mock_topology
        mock_app.Modeller.return_value = mock_modeller
        mock_app.ForceField.return_value = mock_forcefield
        
        # Mock water molecule (3 atoms)
        mock_positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        mock_modeller.positions = mock_positions
        mock_modeller.topology = mock_topology
        
        # Mock atoms
        mock_atoms = []
        elements = ['O', 'H', 'H']
        masses = [15.999, 1.008, 1.008]
        
        for i, (element, mass) in enumerate(zip(elements, masses)):
            mock_atom = Mock()
            mock_atom.element.symbol = element
            mock_atom.element.mass.value_in_unit.return_value = mass
            mock_atom.residue.id = 0
            mock_atom.residue.name = 'HOH'
            mock_atoms.append(mock_atom)
        
        mock_topology.atoms.return_value = mock_atoms
        mock_topology.bonds.return_value = [(mock_atoms[0], mock_atoms[1]), (mock_atoms[0], mock_atoms[2])]
        
        # Mock system creation
        mock_system = Mock()
        mock_forcefield.createSystem.return_value = mock_system


if __name__ == "__main__":
    pytest.main([__file__])