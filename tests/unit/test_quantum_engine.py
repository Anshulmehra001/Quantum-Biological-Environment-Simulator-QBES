"""
Unit tests for the QuantumEngine class.
"""

import pytest
import numpy as np
from qbes.quantum_engine import QuantumEngine
from qbes.core.data_models import DensityMatrix, Hamiltonian, QuantumSubsystem


class TestQuantumEngine:
    """Test suite for QuantumEngine functionality."""
    
    def test_initialization(self, quantum_engine):
        """Test QuantumEngine initialization."""
        assert quantum_engine is not None
        assert quantum_engine.tolerance == 1e-12
    
    def test_create_two_level_hamiltonian(self, quantum_engine):
        """Test creation of two-level Hamiltonian."""
        hamiltonian = quantum_engine.create_two_level_hamiltonian(
            energy_gap=2.0, coupling=0.1
        )
        
        assert hamiltonian.matrix.shape == (2, 2)
        assert hamiltonian.matrix[0, 0] == 0.0
        assert hamiltonian.matrix[1, 1] == 2.0
        assert hamiltonian.matrix[0, 1] == 0.1
        assert hamiltonian.matrix[1, 0] == 0.1
        assert len(hamiltonian.basis_labels) == 2
    
    def test_initialize_ground_state(self, quantum_engine, simple_quantum_subsystem):
        """Test initialization of ground state."""
        state = quantum_engine.initialize_state(
            simple_quantum_subsystem, state_type="ground"
        )
        
        assert state.matrix.shape == (2, 2)
        assert np.isclose(state.matrix[0, 0], 1.0)
        assert np.isclose(state.matrix[1, 1], 0.0)
        assert np.isclose(np.trace(state.matrix), 1.0)
    
    def test_initialize_superposition_state(self, quantum_engine, simple_quantum_subsystem):
        """Test initialization of superposition state."""
        state = quantum_engine.initialize_state(
            simple_quantum_subsystem, state_type="superposition"
        )
        
        assert state.matrix.shape == (2, 2)
        assert np.isclose(state.matrix[0, 0], 0.5)
        assert np.isclose(state.matrix[1, 1], 0.5)
        assert np.isclose(np.trace(state.matrix), 1.0)
    
    def test_calculate_purity(self, quantum_engine, simple_density_matrix):
        """Test purity calculation."""
        purity = quantum_engine.calculate_purity(simple_density_matrix)
        
        # For the test matrix [[0.6, 0.2j], [-0.2j, 0.4]]
        # Purity = Tr(ρ²) should be calculated correctly
        expected_purity = np.real(np.trace(
            simple_density_matrix.matrix @ simple_density_matrix.matrix
        ))
        
        assert np.isclose(purity, expected_purity)
        assert 0 <= purity <= 1
    
    def test_calculate_von_neumann_entropy(self, quantum_engine, simple_density_matrix):
        """Test von Neumann entropy calculation."""
        entropy = quantum_engine.calculate_von_neumann_entropy(simple_density_matrix)
        
        assert entropy >= 0
        assert np.isfinite(entropy)
    
    def test_validate_quantum_state_valid(self, quantum_engine, simple_density_matrix):
        """Test validation of a valid quantum state."""
        result = quantum_engine.validate_quantum_state(simple_density_matrix)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_quantum_state_invalid_trace(self, quantum_engine):
        """Test validation of quantum state with invalid trace."""
        # Create matrix with trace != 1
        invalid_matrix = np.array([[0.8, 0.0], [0.0, 0.8]], dtype=complex)
        invalid_state = DensityMatrix(
            matrix=invalid_matrix,
            basis_labels=["ground", "excited"],
            time=0.0
        )
        
        result = quantum_engine.validate_quantum_state(invalid_state)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_evolve_state_basic(self, quantum_engine, simple_density_matrix, 
                               two_level_hamiltonian, simple_lindblad_operator):
        """Test basic state evolution."""
        evolved_state = quantum_engine.evolve_state(
            simple_density_matrix,
            time_step=0.01,
            hamiltonian=two_level_hamiltonian,
            lindblad_operators=[simple_lindblad_operator]
        )
        
        assert evolved_state.matrix.shape == simple_density_matrix.matrix.shape
        assert evolved_state.time == simple_density_matrix.time + 0.01
        assert np.isclose(np.trace(evolved_state.matrix), 1.0, rtol=1e-10)
    
    def test_calculate_observables(self, quantum_engine, simple_density_matrix, 
                                  two_level_hamiltonian):
        """Test calculation of quantum observables."""
        observables = quantum_engine.calculate_observables(
            simple_density_matrix, two_level_hamiltonian
        )
        
        assert 'purity' in observables
        assert 'von_neumann_entropy' in observables
        assert 'energy' in observables
        assert 'trace' in observables
        
        assert np.isclose(observables['trace'], 1.0)
        assert 0 <= observables['purity'] <= 1
        assert observables['von_neumann_entropy'] >= 0
    
    def test_fidelity_calculation(self, quantum_engine, simple_density_matrix):
        """Test quantum fidelity calculation."""
        # Fidelity with itself should be 1
        fidelity = quantum_engine.fidelity(simple_density_matrix, simple_density_matrix)
        
        assert np.isclose(fidelity, 1.0)
        assert 0 <= fidelity <= 1
    
    def test_create_thermal_state(self, quantum_engine, two_level_hamiltonian):
        """Test creation of thermal state."""
        thermal_state = quantum_engine.create_thermal_state(
            two_level_hamiltonian, temperature=300.0
        )
        
        assert thermal_state.matrix.shape == (2, 2)
        assert np.isclose(np.trace(thermal_state.matrix), 1.0)
        
        # Check that it's a valid density matrix
        validation = quantum_engine.validate_quantum_state(thermal_state)
        assert validation.is_valid
    
    def test_error_handling_invalid_input(self, quantum_engine):
        """Test error handling for invalid inputs."""
        with pytest.raises(TypeError):
            quantum_engine.calculate_purity("not a density matrix")
        
        with pytest.raises(ValueError):
            quantum_engine.create_two_level_hamiltonian(
                energy_gap=-1.0, coupling=0.1  # Negative energy gap might be invalid
            )
    
    @pytest.mark.parametrize("energy_gap,coupling", [
        (1.0, 0.1),
        (2.0, 0.5),
        (0.5, 0.01),
    ])
    def test_hamiltonian_creation_parameters(self, quantum_engine, energy_gap, coupling):
        """Test Hamiltonian creation with different parameters."""
        hamiltonian = quantum_engine.create_two_level_hamiltonian(
            energy_gap=energy_gap, coupling=coupling
        )
        
        assert hamiltonian.matrix[1, 1] == energy_gap
        assert hamiltonian.matrix[0, 1] == coupling
        assert hamiltonian.matrix[1, 0] == coupling