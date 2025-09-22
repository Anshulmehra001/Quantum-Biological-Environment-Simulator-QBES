"""
Unit tests for quantum state operations in the quantum engine.
"""

import pytest
import numpy as np
import math
from qbes.quantum_engine import QuantumEngine
from qbes.core.data_models import (
    DensityMatrix, QuantumState, Hamiltonian, ValidationResult
)


class TestQuantumStateOperations:
    """Test quantum state initialization and manipulation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
        self.tolerance = 1e-10
    
    def test_create_pure_state(self):
        """Test creation of pure quantum states."""
        # Test normalized state
        coeffs = np.array([1.0, 0.0], dtype=complex)
        labels = ["0", "1"]
        state = self.engine.create_pure_state(coeffs, labels)
        
        assert np.allclose(state.coefficients, coeffs)
        assert state.basis_labels == labels
        assert np.isclose(np.linalg.norm(state.coefficients), 1.0)
        
        # Test unnormalized state gets normalized
        coeffs_unnorm = np.array([2.0, 0.0], dtype=complex)
        state_norm = self.engine.create_pure_state(coeffs_unnorm, labels)
        assert np.isclose(np.linalg.norm(state_norm.coefficients), 1.0)
        assert np.allclose(state_norm.coefficients, [1.0, 0.0])
        
        # Test superposition state
        coeffs_super = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        state_super = self.engine.create_pure_state(coeffs_super, labels)
        assert np.isclose(np.linalg.norm(state_super.coefficients), 1.0)
        
        # Test zero state raises error
        with pytest.raises(ValueError, match="Cannot create state with zero norm"):
            self.engine.create_pure_state(np.array([0.0, 0.0]), labels)
    
    def test_pure_state_to_density_matrix(self):
        """Test conversion from pure state to density matrix."""
        # Test |0⟩ state
        coeffs = np.array([1.0, 0.0], dtype=complex)
        labels = ["0", "1"]
        pure_state = self.engine.create_pure_state(coeffs, labels)
        
        rho = self.engine.pure_state_to_density_matrix(pure_state, time=1.0)
        
        expected_matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert np.allclose(rho.matrix, expected_matrix)
        assert rho.basis_labels == labels
        assert rho.time == 1.0
        
        # Test superposition state |+⟩ = (|0⟩ + |1⟩)/√2
        coeffs_super = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        pure_super = self.engine.create_pure_state(coeffs_super, labels)
        rho_super = self.engine.pure_state_to_density_matrix(pure_super)
        
        expected_super = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(rho_super.matrix, expected_super)
    
    def test_create_maximally_mixed_state(self):
        """Test creation of maximally mixed states."""
        labels = ["0", "1"]
        mixed_state = self.engine.create_maximally_mixed_state(2, labels, time=2.0)
        
        expected_matrix = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert np.allclose(mixed_state.matrix, expected_matrix)
        assert mixed_state.basis_labels == labels
        assert mixed_state.time == 2.0
        
        # Test 3-dimensional case
        labels_3d = ["0", "1", "2"]
        mixed_3d = self.engine.create_maximally_mixed_state(3, labels_3d)
        expected_3d = np.eye(3) / 3
        assert np.allclose(mixed_3d.matrix, expected_3d)
        
        # Test dimension mismatch
        with pytest.raises(ValueError, match="Number of basis labels must match dimension"):
            self.engine.create_maximally_mixed_state(2, ["0"])
    
    def test_create_thermal_state(self):
        """Test creation of thermal (Gibbs) states."""
        # Simple 2-level system Hamiltonian
        H_matrix = np.array([[0.0, 0.0], [0.0, 1.0]])  # Energy gap of 1 eV
        labels = ["ground", "excited"]
        hamiltonian = Hamiltonian(matrix=H_matrix, basis_labels=labels)
        
        # Very high temperature limit (should approach maximally mixed)
        thermal_high_T = self.engine.create_thermal_state(hamiltonian, temperature=1000000)
        
        # At very high temperature, populations should be nearly equal
        pop_ground = thermal_high_T.matrix[0, 0]
        pop_excited = thermal_high_T.matrix[1, 1]
        assert abs(pop_ground - pop_excited) < 0.01
        
        # Low temperature limit (should be mostly ground state)
        thermal_low_T = self.engine.create_thermal_state(hamiltonian, temperature=1.0)
        
        # At low temperature, ground state should dominate
        pop_ground_low = thermal_low_T.matrix[0, 0]
        pop_excited_low = thermal_low_T.matrix[1, 1]
        assert pop_ground_low > 0.9
        assert pop_excited_low < 0.1
        
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be positive"):
            self.engine.create_thermal_state(hamiltonian, temperature=-1.0)
    
    def test_trace_density_matrix(self):
        """Test density matrix trace calculation."""
        # Pure state should have trace 1
        coeffs = np.array([1.0, 0.0], dtype=complex)
        labels = ["0", "1"]
        pure_state = self.engine.create_pure_state(coeffs, labels)
        rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        trace_val = self.engine.trace_density_matrix(rho)
        assert np.isclose(trace_val, 1.0)
        
        # Mixed state should also have trace 1
        mixed_state = self.engine.create_maximally_mixed_state(2, labels)
        trace_mixed = self.engine.trace_density_matrix(mixed_state)
        assert np.isclose(trace_mixed, 1.0)
    
    def test_partial_trace(self):
        """Test partial trace operations."""
        # Create a 2x2 tensor product state |00⟩
        labels = ["00", "01", "10", "11"]
        rho_matrix = np.zeros((4, 4))
        rho_matrix[0, 0] = 1.0  # |00⟩⟨00|
        
        bipartite_state = DensityMatrix(matrix=rho_matrix, basis_labels=labels)
        
        # Trace out second subsystem
        reduced_state = self.engine.partial_trace(
            bipartite_state, 
            subsystem_indices=[1], 
            subsystem_dimensions=[2, 2]
        )
        
        # Should get |0⟩⟨0| for first qubit
        expected_reduced = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert np.allclose(reduced_state.matrix, expected_reduced)
        
        # Test error cases
        with pytest.raises(ValueError, match="Need at least 2 subsystems"):
            self.engine.partial_trace(bipartite_state, [0], [4])
        
        with pytest.raises(ValueError, match="Matrix dimension doesn't match"):
            self.engine.partial_trace(bipartite_state, [0], [3, 3])
    
    def test_fidelity(self):
        """Test quantum fidelity calculation."""
        labels = ["0", "1"]
        
        # Fidelity between identical states should be 1
        state1 = self.engine.create_maximally_mixed_state(2, labels)
        fidelity_identical = self.engine.fidelity(state1, state1)
        assert np.isclose(fidelity_identical, 1.0)
        
        # Fidelity between orthogonal pure states should be 0
        coeffs1 = np.array([1.0, 0.0], dtype=complex)
        coeffs2 = np.array([0.0, 1.0], dtype=complex)
        
        pure1 = self.engine.create_pure_state(coeffs1, labels)
        pure2 = self.engine.create_pure_state(coeffs2, labels)
        
        rho1 = self.engine.pure_state_to_density_matrix(pure1)
        rho2 = self.engine.pure_state_to_density_matrix(pure2)
        
        fidelity_orthogonal = self.engine.fidelity(rho1, rho2)
        assert np.isclose(fidelity_orthogonal, 0.0, atol=1e-10)
        
        # Test dimension mismatch
        state_3d = self.engine.create_maximally_mixed_state(3, ["0", "1", "2"])
        with pytest.raises(ValueError, match="Density matrices must have same dimensions"):
            self.engine.fidelity(state1, state_3d)
    
    def test_purity_calculation(self):
        """Test purity calculation."""
        labels = ["0", "1"]
        
        # Pure state should have purity 1
        coeffs = np.array([1.0, 0.0], dtype=complex)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        rho_pure = self.engine.pure_state_to_density_matrix(pure_state)
        
        purity_pure = self.engine.calculate_purity(rho_pure)
        assert np.isclose(purity_pure, 1.0)
        
        # Maximally mixed state should have purity 1/d
        mixed_state = self.engine.create_maximally_mixed_state(2, labels)
        purity_mixed = self.engine.calculate_purity(mixed_state)
        assert np.isclose(purity_mixed, 0.5)
        
        # 3D maximally mixed state
        mixed_3d = self.engine.create_maximally_mixed_state(3, ["0", "1", "2"])
        purity_3d = self.engine.calculate_purity(mixed_3d)
        assert np.isclose(purity_3d, 1.0/3.0)
    
    def test_von_neumann_entropy(self):
        """Test von Neumann entropy calculation."""
        labels = ["0", "1"]
        
        # Pure state should have entropy 0
        coeffs = np.array([1.0, 0.0], dtype=complex)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        rho_pure = self.engine.pure_state_to_density_matrix(pure_state)
        
        entropy_pure = self.engine.calculate_von_neumann_entropy(rho_pure)
        assert np.isclose(entropy_pure, 0.0, atol=1e-10)
        
        # Maximally mixed 2D state should have entropy log2(2) = 1
        mixed_state = self.engine.create_maximally_mixed_state(2, labels)
        entropy_mixed = self.engine.calculate_von_neumann_entropy(mixed_state)
        assert np.isclose(entropy_mixed, 1.0)
        
        # Maximally mixed 3D state should have entropy log2(3)
        mixed_3d = self.engine.create_maximally_mixed_state(3, ["0", "1", "2"])
        entropy_3d = self.engine.calculate_von_neumann_entropy(mixed_3d)
        assert np.isclose(entropy_3d, np.log2(3))
    
    def test_linear_entropy(self):
        """Test linear entropy calculation."""
        labels = ["0", "1"]
        
        # Pure state should have linear entropy 0
        coeffs = np.array([1.0, 0.0], dtype=complex)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        rho_pure = self.engine.pure_state_to_density_matrix(pure_state)
        
        linear_entropy_pure = self.engine.calculate_linear_entropy(rho_pure)
        assert np.isclose(linear_entropy_pure, 0.0)
        
        # Maximally mixed state should have linear entropy (d-1)/d
        mixed_state = self.engine.create_maximally_mixed_state(2, labels)
        linear_entropy_mixed = self.engine.calculate_linear_entropy(mixed_state)
        assert np.isclose(linear_entropy_mixed, 0.5)
    
    def test_coherence_l1_norm(self):
        """Test l1-norm coherence measure."""
        labels = ["0", "1"]
        
        # Diagonal state should have zero coherence
        diagonal_matrix = np.array([[0.7, 0.0], [0.0, 0.3]])
        diagonal_state = DensityMatrix(matrix=diagonal_matrix, basis_labels=labels)
        
        coherence_diagonal = self.engine.calculate_coherence_l1_norm(diagonal_state)
        assert np.isclose(coherence_diagonal, 0.0)
        
        # Superposition state should have non-zero coherence
        coeffs_super = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        pure_super = self.engine.create_pure_state(coeffs_super, labels)
        rho_super = self.engine.pure_state_to_density_matrix(pure_super)
        
        coherence_super = self.engine.calculate_coherence_l1_norm(rho_super)
        assert coherence_super > 0.0
        assert np.isclose(coherence_super, 1.0)  # |+⟩ state has l1 coherence = 1
    
    def test_validate_quantum_state(self):
        """Test quantum state validation."""
        labels = ["0", "1"]
        
        # Valid pure state
        coeffs = np.array([1.0, 0.0], dtype=complex)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        rho_valid = self.engine.pure_state_to_density_matrix(pure_state)
        
        validation_result = self.engine.validate_quantum_state(rho_valid)
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        
        # Create invalid state by manually constructing it
        # We'll create a valid state first, then modify it to be invalid
        valid_mixed = self.engine.create_maximally_mixed_state(2, labels)
        
        # Manually modify the matrix to have wrong trace
        invalid_matrix = np.array([[0.5, 0.0], [0.0, 0.3]])  # Trace = 0.8
        valid_mixed.matrix = invalid_matrix  # Bypass validation
        
        validation_invalid = self.engine.validate_quantum_state(valid_mixed)
        assert not validation_invalid.is_valid
        assert any("trace" in error.lower() for error in validation_invalid.errors)
    
    def test_calculate_coherence_measures(self):
        """Test comprehensive coherence measures calculation."""
        labels = ["0", "1"]
        
        # Test with pure state
        coeffs = np.array([1.0, 0.0], dtype=complex)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        rho_pure = self.engine.pure_state_to_density_matrix(pure_state)
        
        coherence_metrics = self.engine.calculate_coherence_measures(rho_pure)
        
        assert np.isclose(coherence_metrics.purity, 1.0)
        assert np.isclose(coherence_metrics.von_neumann_entropy, 0.0, atol=1e-10)
        assert coherence_metrics.coherence_lifetime > 0
        
        # Test with mixed state
        mixed_state = self.engine.create_maximally_mixed_state(2, labels)
        coherence_mixed = self.engine.calculate_coherence_measures(mixed_state)
        
        assert np.isclose(coherence_mixed.purity, 0.5)
        assert np.isclose(coherence_mixed.von_neumann_entropy, 1.0)


class TestAnalyticalSolutions:
    """Test against known analytical solutions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
    
    def test_two_level_system_thermal_state(self):
        """Test thermal state for two-level system against analytical solution."""
        # Two-level system with energy gap ΔE
        delta_E = 1.0  # eV
        H_matrix = np.array([[0.0, 0.0], [0.0, delta_E]])
        labels = ["ground", "excited"]
        hamiltonian = Hamiltonian(matrix=H_matrix, basis_labels=labels)
        
        # Test at specific temperature
        temperature = 300.0  # K
        k_B = 8.617333e-5  # eV/K
        beta = 1.0 / (k_B * temperature)
        
        # Analytical solution for populations
        exp_factor = np.exp(-beta * delta_E)
        Z = 1 + exp_factor  # Partition function
        p_ground_analytical = 1.0 / Z
        p_excited_analytical = exp_factor / Z
        
        # Numerical solution
        thermal_state = self.engine.create_thermal_state(hamiltonian, temperature)
        p_ground_numerical = thermal_state.matrix[0, 0]
        p_excited_numerical = thermal_state.matrix[1, 1]
        
        assert np.isclose(p_ground_numerical, p_ground_analytical, rtol=1e-10)
        assert np.isclose(p_excited_numerical, p_excited_analytical, rtol=1e-10)
    
    def test_harmonic_oscillator_coherent_state(self):
        """Test coherent state properties for harmonic oscillator."""
        # Create a truncated harmonic oscillator basis
        n_levels = 4
        labels = [f"n={i}" for i in range(n_levels)]
        
        # Create coherent state |α⟩ with α = 1
        alpha = 1.0
        coeffs = np.array([
            np.exp(-0.5 * abs(alpha)**2) * (alpha**n) / np.sqrt(math.factorial(n))
            for n in range(n_levels)
        ], dtype=complex)
        
        coherent_state = self.engine.create_pure_state(coeffs, labels)
        rho_coherent = self.engine.pure_state_to_density_matrix(coherent_state)
        
        # Coherent states should be pure
        purity = self.engine.calculate_purity(rho_coherent)
        assert np.isclose(purity, 1.0, rtol=1e-10)
        
        # Should have zero von Neumann entropy
        entropy = self.engine.calculate_von_neumann_entropy(rho_coherent)
        assert np.isclose(entropy, 0.0, atol=1e-10)
    
    def test_bell_state_properties(self):
        """Test properties of Bell states."""
        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        labels = ["00", "01", "10", "11"]
        coeffs = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2)
        
        bell_state = self.engine.create_pure_state(coeffs, labels)
        rho_bell = self.engine.pure_state_to_density_matrix(bell_state)
        
        # Bell state should be pure
        purity = self.engine.calculate_purity(rho_bell)
        assert np.isclose(purity, 1.0, rtol=1e-10)
        
        # Should have maximum coherence in computational basis
        coherence = self.engine.calculate_coherence_l1_norm(rho_bell)
        assert coherence > 0.0
        
        # Test partial trace - should give maximally mixed single-qubit states
        reduced_state = self.engine.partial_trace(
            rho_bell, 
            subsystem_indices=[1], 
            subsystem_dimensions=[2, 2]
        )
        
        # Reduced state should be maximally mixed
        expected_reduced = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert np.allclose(reduced_state.matrix, expected_reduced, rtol=1e-10)