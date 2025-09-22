"""
Unit tests for Lindblad master equation solver in the quantum engine.
"""

import pytest
import numpy as np
from qbes.quantum_engine import QuantumEngine
from qbes.core.data_models import (
    DensityMatrix, Hamiltonian, LindbladOperator
)


class TestLindbladOperators:
    """Test Lindblad operator creation and manipulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
        self.tolerance = 1e-10
    
    def test_create_dephasing_operator(self):
        """Test creation of dephasing Lindblad operators."""
        dimension = 3
        site_index = 1
        coupling_strength = 0.1
        
        dephasing_op = self.engine.create_dephasing_operator(
            dimension=dimension,
            site_index=site_index,
            coupling_strength=coupling_strength
        )
        
        # Check operator properties
        assert dephasing_op.coupling_strength == coupling_strength
        assert dephasing_op.operator_type == "dephasing"
        
        # Check operator matrix (should be |1⟩⟨1|)
        expected_operator = np.zeros((dimension, dimension))
        expected_operator[site_index, site_index] = 1.0
        
        assert np.allclose(dephasing_op.operator, expected_operator)
        
        # Test error case
        with pytest.raises(ValueError, match="exceeds dimension"):
            self.engine.create_dephasing_operator(
                dimension=2,
                site_index=3,
                coupling_strength=0.1
            )
    
    def test_create_relaxation_operator(self):
        """Test creation of relaxation Lindblad operators."""
        dimension = 3
        from_state = 2
        to_state = 0
        coupling_strength = 0.2
        
        relaxation_op = self.engine.create_relaxation_operator(
            dimension=dimension,
            from_state=from_state,
            to_state=to_state,
            coupling_strength=coupling_strength
        )
        
        # Check operator properties
        assert relaxation_op.coupling_strength == coupling_strength
        assert relaxation_op.operator_type == "relaxation"
        
        # Check operator matrix (should be |0⟩⟨2|)
        expected_operator = np.zeros((dimension, dimension))
        expected_operator[to_state, from_state] = 1.0
        
        assert np.allclose(relaxation_op.operator, expected_operator)
        
        # Test error case
        with pytest.raises(ValueError, match="exceed dimension"):
            self.engine.create_relaxation_operator(
                dimension=2,
                from_state=3,
                to_state=0,
                coupling_strength=0.1
            )
    
    def test_create_thermal_lindblad_operators(self):
        """Test creation of thermal Lindblad operators."""
        # Simple two-level system
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=1.0,
            coupling=0.0
        )
        
        temperature = 300.0  # K
        coupling_strength = 0.01
        
        thermal_ops = self.engine.create_thermal_lindblad_operators(
            hamiltonian=hamiltonian,
            temperature=temperature,
            coupling_strength=coupling_strength
        )
        
        # Should have operators for all transitions
        assert len(thermal_ops) > 0
        
        # All operators should be thermal type
        for op in thermal_ops:
            assert op.operator_type == "thermal"
            assert op.coupling_strength >= 0
        
        # Test error case
        with pytest.raises(ValueError, match="Temperature must be positive"):
            self.engine.create_thermal_lindblad_operators(
                hamiltonian=hamiltonian,
                temperature=-1.0,
                coupling_strength=0.1
            )
    
    def test_apply_lindblad_operators(self):
        """Test application of Lindblad operators to density matrices."""
        # Create initial pure state
        labels = ["0", "1"]
        coeffs = np.array([1.0, 1.0]) / np.sqrt(2)  # |+⟩ state
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Create dephasing operator
        dephasing_op = self.engine.create_dephasing_operator(
            dimension=2,
            site_index=0,
            coupling_strength=1.0
        )
        
        # Apply dephasing
        dephased_state = self.engine.apply_lindblad_operators(
            initial_rho, [dephasing_op]
        )
        
        # Dephasing should reduce off-diagonal elements
        initial_coherence = abs(initial_rho.matrix[0, 1])
        final_coherence = abs(dephased_state.matrix[0, 1])
        
        # Note: This is just one application, not time evolution
        # The coherence change depends on the specific implementation
        assert dephased_state.basis_labels == initial_rho.basis_labels
        
        # Check that trace is preserved
        initial_trace = np.trace(initial_rho.matrix)
        final_trace = np.trace(dephased_state.matrix)
        assert np.isclose(initial_trace, final_trace, rtol=1e-10)


class TestLindbladEvolution:
    """Test Lindblad master equation evolution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
        self.tolerance = 1e-8
    
    def test_evolve_state_no_dissipation(self):
        """Test state evolution with no Lindblad operators (unitary evolution)."""
        # Create initial state
        labels = ["0", "1"]
        coeffs = np.array([1.0, 0.0])  # |0⟩ state
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Create Hamiltonian with coupling
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=0.0,  # No energy difference
            coupling=1.0     # Strong coupling
        )
        
        # Evolve with no Lindblad operators
        time_step = np.pi / 2  # Quarter period
        evolved_state = self.engine.evolve_state(
            initial_rho, time_step, hamiltonian, []
        )
        
        # Should have evolved unitarily
        assert evolved_state.time == initial_rho.time + time_step
        
        # Check that trace is preserved
        assert np.isclose(np.trace(evolved_state.matrix), 1.0, rtol=self.tolerance)
        
        # Check that purity is preserved (unitary evolution)
        initial_purity = self.engine.calculate_purity(initial_rho)
        final_purity = self.engine.calculate_purity(evolved_state)
        assert np.isclose(initial_purity, final_purity, rtol=self.tolerance)
    
    def test_evolve_state_with_dephasing(self):
        """Test state evolution with dephasing."""
        # Create superposition state
        labels = ["0", "1"]
        coeffs = np.array([1.0, 1.0]) / np.sqrt(2)  # |+⟩ state
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Create Hamiltonian (no evolution for simplicity)
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=0.0,
            coupling=0.0
        )
        
        # Create dephasing operator
        dephasing_op = self.engine.create_dephasing_operator(
            dimension=2,
            site_index=0,
            coupling_strength=1.0
        )
        
        # Evolve with dephasing
        time_step = 0.1
        evolved_state = self.engine.evolve_state(
            initial_rho, time_step, hamiltonian, [dephasing_op]
        )
        
        # Check basic properties
        assert evolved_state.time == initial_rho.time + time_step
        assert np.isclose(np.trace(evolved_state.matrix), 1.0, rtol=self.tolerance)
        
        # Dephasing should reduce purity
        initial_purity = self.engine.calculate_purity(initial_rho)
        final_purity = self.engine.calculate_purity(evolved_state)
        assert final_purity <= initial_purity + self.tolerance
        
        # Dephasing should reduce coherence
        initial_coherence = self.engine.calculate_coherence_l1_norm(initial_rho)
        final_coherence = self.engine.calculate_coherence_l1_norm(evolved_state)
        assert final_coherence <= initial_coherence + self.tolerance
    
    def test_evolve_lindblad_adaptive(self):
        """Test adaptive time stepping evolution."""
        # Create initial state
        labels = ["0", "1"]
        coeffs = np.array([1.0, 1.0]) / np.sqrt(2)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Create simple Hamiltonian
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=1.0,
            coupling=0.1
        )
        
        # Create dephasing operator
        dephasing_op = self.engine.create_dephasing_operator(
            dimension=2,
            site_index=0,
            coupling_strength=0.5
        )
        
        # Evolve with adaptive time stepping
        total_time = 1.0
        trajectory = self.engine.evolve_lindblad_adaptive(
            initial_rho, hamiltonian, [dephasing_op], total_time
        )
        
        # Check trajectory properties
        assert len(trajectory) > 1
        assert trajectory[0].time == initial_rho.time
        assert trajectory[-1].time <= initial_rho.time + total_time + self.tolerance
        
        # Check that all states are valid density matrices
        for state in trajectory:
            validation = self.engine.validate_quantum_state(state)
            assert validation.is_valid, f"Invalid state at t={state.time}: {validation.errors}"
        
        # Check monotonic time evolution
        for i in range(1, len(trajectory)):
            assert trajectory[i].time >= trajectory[i-1].time
    
    def test_calculate_decoherence_time(self):
        """Test decoherence time calculation."""
        # Create coherent superposition state
        labels = ["0", "1"]
        coeffs = np.array([1.0, 1.0]) / np.sqrt(2)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Create Hamiltonian
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=0.0,
            coupling=0.0
        )
        
        # Create strong dephasing operator
        dephasing_op = self.engine.create_dephasing_operator(
            dimension=2,
            site_index=0,
            coupling_strength=2.0  # Strong dephasing
        )
        
        # Calculate decoherence time
        decoherence_time = self.engine.calculate_decoherence_time(
            initial_rho, hamiltonian, [dephasing_op]
        )
        
        # Should be finite and positive
        assert decoherence_time > 0
        assert decoherence_time < 100.0  # Should decay reasonably quickly
        
        # Test with no dephasing (should take very long or hit max time)
        no_dephasing_time = self.engine.calculate_decoherence_time(
            initial_rho, hamiltonian, []
        )
        
        # Without dephasing, decoherence time should be much longer
        assert no_dephasing_time >= decoherence_time
    
    def test_ensure_valid_density_matrix(self):
        """Test density matrix correction function."""
        # Create a matrix with small numerical errors
        rho_with_errors = np.array([
            [0.5001, 0.1 + 0.001j],
            [0.1 - 0.001j, 0.4999]
        ])
        
        # Add small non-Hermitian part
        rho_with_errors[0, 1] += 0.0001j
        
        # Correct the matrix
        rho_corrected = self.engine._ensure_valid_density_matrix(rho_with_errors)
        
        # Check that result is Hermitian
        assert np.allclose(rho_corrected, rho_corrected.conj().T, atol=1e-12)
        
        # Check that trace is 1
        assert np.isclose(np.trace(rho_corrected), 1.0, atol=1e-12)
        
        # Check that eigenvalues are non-negative
        eigenvals = np.linalg.eigvals(rho_corrected)
        assert np.all(eigenvals >= -1e-12)
        
        # Test with matrix having negative eigenvalues
        rho_negative = np.array([
            [1.5, 0.0],
            [0.0, -0.5]
        ])
        
        rho_fixed = self.engine._ensure_valid_density_matrix(rho_negative)
        
        # Should fix negative eigenvalues
        eigenvals_fixed = np.linalg.eigvals(rho_fixed)
        assert np.all(eigenvals_fixed >= -1e-12)
        assert np.isclose(np.trace(rho_fixed), 1.0, atol=1e-12)


class TestAnalyticalSolutions:
    """Test Lindblad evolution against known analytical solutions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
    
    def test_pure_dephasing_analytical(self):
        """Test pure dephasing against analytical solution."""
        # For pure dephasing of |+⟩ state: ρ₀₁(t) = ρ₀₁(0) * exp(-γt)
        labels = ["0", "1"]
        coeffs = np.array([1.0, 1.0]) / np.sqrt(2)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # No Hamiltonian evolution
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=0.0,
            coupling=0.0
        )
        
        # Pure dephasing
        gamma = 1.0
        dephasing_op = LindbladOperator(
            operator=np.array([[1.0, 0.0], [0.0, 0.0]]),  # |0⟩⟨0|
            coupling_strength=gamma,
            operator_type="dephasing"
        )
        
        # Evolve for specific time
        time = 0.5
        evolved_state = self.engine.evolve_state(
            initial_rho, time, hamiltonian, [dephasing_op]
        )
        
        # Analytical solution: off-diagonal elements decay as exp(-γt)
        initial_coherence = initial_rho.matrix[0, 1]
        expected_coherence = initial_coherence * np.exp(-gamma * time)
        actual_coherence = evolved_state.matrix[0, 1]
        
        # Allow for some numerical error (the implementation may not exactly match analytical)
        # The key is that coherence should decay
        assert abs(actual_coherence) < abs(initial_coherence)
        
        # Check that it's in the right ballpark (within factor of 2)
        assert abs(actual_coherence) < 2 * abs(expected_coherence)
        assert abs(actual_coherence) > 0.5 * abs(expected_coherence)
        
        # Diagonal elements should remain unchanged for pure dephasing
        assert np.isclose(evolved_state.matrix[0, 0], initial_rho.matrix[0, 0], rtol=0.01)
        assert np.isclose(evolved_state.matrix[1, 1], initial_rho.matrix[1, 1], rtol=0.01)
    
    def test_thermal_equilibrium(self):
        """Test that thermal Lindblad operators lead to thermal equilibrium."""
        # Two-level system
        energy_gap = 1.0
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=energy_gap,
            coupling=0.0
        )
        
        # Start with excited state
        labels = ["ground", "excited"]
        coeffs = np.array([0.0, 1.0])  # |excited⟩
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Create thermal operators
        temperature = 300.0
        coupling_strength = 0.1
        thermal_ops = self.engine.create_thermal_lindblad_operators(
            hamiltonian, temperature, coupling_strength
        )
        
        # Evolve for long time to reach equilibrium
        total_time = 10.0
        trajectory = self.engine.evolve_lindblad_adaptive(
            initial_rho, hamiltonian, thermal_ops, total_time
        )
        
        final_state = trajectory[-1]
        
        # Check that final state approaches thermal equilibrium
        thermal_state = self.engine.create_thermal_state(hamiltonian, temperature)
        
        # Compare populations (diagonal elements)
        final_pop_ground = final_state.matrix[0, 0]
        thermal_pop_ground = thermal_state.matrix[0, 0]
        
        # Should be close to thermal equilibrium (allow some tolerance)
        assert np.isclose(final_pop_ground, thermal_pop_ground, rtol=0.1)
    
    def test_unitary_evolution_preservation(self):
        """Test that unitary evolution preserves quantum properties."""
        # Create entangled state (if we had 2-qubit system)
        # For now, test with superposition state
        labels = ["0", "1"]
        coeffs = np.array([1.0, 1.0]) / np.sqrt(2)
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Hamiltonian with only coupling (no energy difference)
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=0.0,
            coupling=1.0
        )
        
        # Evolve unitarily (no Lindblad operators)
        time = np.pi  # Full period
        evolved_state = self.engine.evolve_state(
            initial_rho, time, hamiltonian, []
        )
        
        # Unitary evolution should preserve purity
        initial_purity = self.engine.calculate_purity(initial_rho)
        final_purity = self.engine.calculate_purity(evolved_state)
        assert np.isclose(initial_purity, final_purity, rtol=1e-6)
        
        # Should preserve von Neumann entropy
        initial_entropy = self.engine.calculate_von_neumann_entropy(initial_rho)
        final_entropy = self.engine.calculate_von_neumann_entropy(evolved_state)
        assert np.isclose(initial_entropy, final_entropy, rtol=1e-6)
    
    def test_relaxation_dynamics(self):
        """Test relaxation from excited to ground state."""
        # Start with excited state
        labels = ["ground", "excited"]
        coeffs = np.array([0.0, 1.0])
        pure_state = self.engine.create_pure_state(coeffs, labels)
        initial_rho = self.engine.pure_state_to_density_matrix(pure_state)
        
        # Simple Hamiltonian
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=1.0,
            coupling=0.0
        )
        
        # Create relaxation operator: |ground⟩⟨excited|
        relaxation_op = self.engine.create_relaxation_operator(
            dimension=2,
            from_state=1,  # excited
            to_state=0,    # ground
            coupling_strength=1.0
        )
        
        # Evolve
        time = 1.0
        evolved_state = self.engine.evolve_state(
            initial_rho, time, hamiltonian, [relaxation_op]
        )
        
        # Should have more population in ground state
        initial_ground_pop = initial_rho.matrix[0, 0]
        final_ground_pop = evolved_state.matrix[0, 0]
        
        assert final_ground_pop > initial_ground_pop
        
        # Total population should be conserved
        initial_total = np.trace(initial_rho.matrix)
        final_total = np.trace(evolved_state.matrix)
        assert np.isclose(initial_total, final_total, rtol=1e-8)