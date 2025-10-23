"""
Unit tests for new QuantumEngine methods: initialize_state and calculate_observables.
"""

import unittest
import numpy as np
from qbes.quantum_engine import QuantumEngine
from qbes.core.data_models import QuantumSubsystem, Hamiltonian, DensityMatrix


class TestQuantumEngineNewMethods(unittest.TestCase):
    """Test the newly added QuantumEngine methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
        
        # Import required classes
        from qbes.core.data_models import Atom, QuantumState
        
        # Create atoms for the quantum subsystem
        atom1 = Atom(
            element="C",
            position=np.array([0.0, 0.0, 0.0]),
            charge=0.0,
            mass=12.0,
            atom_id=0
        )
        atom2 = Atom(
            element="N",
            position=np.array([1.0, 0.0, 0.0]),
            charge=0.0,
            mass=14.0,
            atom_id=1
        )
        
        # Create quantum states
        ground_state = QuantumState(
            coefficients=np.array([1.0, 0.0], dtype=complex),
            basis_labels=["ground", "excited"]
        )
        excited_state = QuantumState(
            coefficients=np.array([0.0, 1.0], dtype=complex),
            basis_labels=["ground", "excited"]
        )
        
        # Create a simple 2-level quantum subsystem
        self.quantum_subsystem = QuantumSubsystem(
            atoms=[atom1, atom2],
            hamiltonian_parameters={"coupling": 0.1, "energy_gap": 1.0},
            coupling_matrix=np.array([[0.0, 0.1], [0.1, 1.0]], dtype=complex),
            basis_states=[ground_state, excited_state],
            subsystem_id="test_system"
        )
        
        # Create a simple Hamiltonian
        self.hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 0.1], [0.1, 1.0]], dtype=complex),
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
    
    def test_initialize_state_ground(self):
        """Test ground state initialization."""
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Check that it's a valid density matrix
        self.assertIsInstance(state, DensityMatrix)
        self.assertEqual(state.matrix.shape, (2, 2))
        
        # Check that it's in ground state (population in first state)
        expected_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        np.testing.assert_allclose(state.matrix, expected_matrix, rtol=1e-10)
        
        # Check trace is 1
        self.assertAlmostEqual(np.trace(state.matrix), 1.0, places=10)
    
    def test_initialize_state_mixed(self):
        """Test maximally mixed state initialization."""
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="mixed")
        
        # Check that it's a valid density matrix
        self.assertIsInstance(state, DensityMatrix)
        self.assertEqual(state.matrix.shape, (2, 2))
        
        # Check that it's maximally mixed (identity/dimension)
        expected_matrix = np.eye(2) / 2
        np.testing.assert_allclose(state.matrix, expected_matrix, rtol=1e-10)
        
        # Check trace is 1
        self.assertAlmostEqual(np.trace(state.matrix), 1.0, places=10)
    
    def test_initialize_state_superposition(self):
        """Test equal superposition state initialization."""
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="superposition")
        
        # Check that it's a valid density matrix
        self.assertIsInstance(state, DensityMatrix)
        self.assertEqual(state.matrix.shape, (2, 2))
        
        # Check that it's equal superposition
        expected_matrix = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        np.testing.assert_allclose(state.matrix, expected_matrix, rtol=1e-10)
        
        # Check trace is 1
        self.assertAlmostEqual(np.trace(state.matrix), 1.0, places=10)
    
    def test_initialize_state_thermal(self):
        """Test thermal state initialization."""
        temperature = 300.0  # Kelvin
        state = self.engine.initialize_state(
            self.quantum_subsystem, 
            state_type="thermal", 
            temperature=temperature
        )
        
        # Check that it's a valid density matrix
        self.assertIsInstance(state, DensityMatrix)
        self.assertEqual(state.matrix.shape, (2, 2))
        
        # Check trace is 1
        self.assertAlmostEqual(np.trace(state.matrix), 1.0, places=10)
        
        # Check that it's Hermitian
        np.testing.assert_allclose(state.matrix, state.matrix.conj().T, rtol=1e-10)
        
        # Check that eigenvalues are non-negative
        eigenvals = np.linalg.eigvals(state.matrix)
        self.assertTrue(np.all(eigenvals >= -1e-10))
    
    def test_initialize_state_invalid_type(self):
        """Test error handling for invalid state type."""
        with self.assertRaises(ValueError):
            self.engine.initialize_state(self.quantum_subsystem, state_type="invalid")
    
    def test_initialize_state_thermal_no_temperature(self):
        """Test error handling for thermal state without temperature."""
        with self.assertRaises(ValueError):
            self.engine.initialize_state(self.quantum_subsystem, state_type="thermal")
    
    def test_calculate_observables_ground_state(self):
        """Test observable calculation for ground state."""
        # Create ground state
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Calculate observables
        observables = self.engine.calculate_observables(state, self.hamiltonian)
        
        # Check that all expected observables are present
        expected_keys = [
            'purity', 'von_neumann_entropy', 'linear_entropy', 'coherence_l1_norm',
            'trace', 'energy', 'population_0', 'population_1', 'coherence_0_1',
            'max_coherence', 'total_coherence'
        ]
        for key in expected_keys:
            self.assertIn(key, observables)
        
        # Check specific values for ground state
        self.assertAlmostEqual(observables['purity'], 1.0, places=10)  # Pure state
        self.assertAlmostEqual(observables['von_neumann_entropy'], 0.0, places=10)  # Pure state
        self.assertAlmostEqual(observables['linear_entropy'], 0.0, places=10)  # Pure state
        self.assertAlmostEqual(observables['trace'], 1.0, places=10)  # Valid density matrix
        self.assertAlmostEqual(observables['population_0'], 1.0, places=10)  # Ground state
        self.assertAlmostEqual(observables['population_1'], 0.0, places=10)  # Ground state
        self.assertAlmostEqual(observables['coherence_0_1'], 0.0, places=10)  # No coherence
        self.assertAlmostEqual(observables['energy'], 0.0, places=10)  # Ground state energy
    
    def test_calculate_observables_superposition_state(self):
        """Test observable calculation for superposition state."""
        # Create superposition state
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="superposition")
        
        # Calculate observables
        observables = self.engine.calculate_observables(state, self.hamiltonian)
        
        # Check specific values for superposition state
        self.assertAlmostEqual(observables['purity'], 1.0, places=10)  # Pure state
        self.assertAlmostEqual(observables['trace'], 1.0, places=10)  # Valid density matrix
        self.assertAlmostEqual(observables['population_0'], 0.5, places=10)  # Equal superposition
        self.assertAlmostEqual(observables['population_1'], 0.5, places=10)  # Equal superposition
        self.assertAlmostEqual(observables['coherence_0_1'], 0.5, places=10)  # Maximum coherence
        
        # Energy should be average of diagonal elements
        expected_energy = 0.5 * (0.0 + 1.0) + 2 * 0.5 * 0.1  # <H> for superposition
        self.assertAlmostEqual(observables['energy'], expected_energy, places=10)
    
    def test_calculate_observables_without_hamiltonian(self):
        """Test observable calculation without Hamiltonian."""
        # Create ground state
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Calculate observables without Hamiltonian
        observables = self.engine.calculate_observables(state)
        
        # Check that energy is not calculated
        self.assertNotIn('energy', observables)
        
        # Check that other observables are still present
        self.assertIn('purity', observables)
        self.assertIn('population_0', observables)
    
    def test_calculate_expectation_value(self):
        """Test expectation value calculation."""
        # Create superposition state
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="superposition")
        
        # Calculate expectation value of Hamiltonian
        expectation = self.engine.calculate_expectation_value(state, self.hamiltonian.matrix)
        
        # For equal superposition: <H> = 0.5 * 0 + 0.5 * 1 + 2 * 0.5 * 0.1 = 0.6
        expected_value = 0.6
        self.assertAlmostEqual(np.real(expectation), expected_value, places=10)
    
    def test_calculate_expectation_value_dimension_mismatch(self):
        """Test error handling for dimension mismatch in expectation value."""
        # Create ground state
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Try to calculate expectation with wrong-sized operator
        wrong_operator = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        
        with self.assertRaises(ValueError):
            self.engine.calculate_expectation_value(state, wrong_operator)
    
    def test_validation_benchmark_state_preparation(self):
        """Test state preparation for validation benchmarks."""
        # Test ground state for analytical benchmarks
        ground_state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Verify it's a pure ground state (for analytical validation)
        self.assertAlmostEqual(ground_state.matrix[0, 0], 1.0, places=10)
        self.assertAlmostEqual(ground_state.matrix[1, 1], 0.0, places=10)
        self.assertAlmostEqual(np.trace(ground_state.matrix), 1.0, places=10)
        
        # Test superposition state for coherence benchmarks
        superposition_state = self.engine.initialize_state(self.quantum_subsystem, state_type="superposition")
        
        # Verify equal superposition (for coherence validation)
        self.assertAlmostEqual(superposition_state.matrix[0, 0], 0.5, places=10)
        self.assertAlmostEqual(superposition_state.matrix[1, 1], 0.5, places=10)
        self.assertAlmostEqual(abs(superposition_state.matrix[0, 1]), 0.5, places=10)
    
    def test_validation_observable_accuracy(self):
        """Test observable calculation accuracy for validation purposes."""
        # Create known state for validation
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Calculate observables with high precision
        observables = self.engine.calculate_observables(state, self.hamiltonian)
        
        # Verify high-precision calculations for validation
        self.assertAlmostEqual(observables['purity'], 1.0, places=12)  # Pure state
        self.assertAlmostEqual(observables['trace'], 1.0, places=12)   # Normalized
        self.assertAlmostEqual(observables['population_0'], 1.0, places=12)  # Ground state
        
        # Test energy calculation precision
        expected_ground_energy = 0.0  # Ground state of our test Hamiltonian
        self.assertAlmostEqual(observables['energy'], expected_ground_energy, places=10)
    
    def test_validation_coherence_measures(self):
        """Test coherence measures for validation benchmarks."""
        # Create superposition state for coherence testing
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="superposition")
        
        # Calculate observables
        observables = self.engine.calculate_observables(state, self.hamiltonian)
        
        # Verify coherence measures for validation
        self.assertAlmostEqual(observables['coherence_0_1'], 0.5, places=10)  # Maximum coherence
        self.assertAlmostEqual(observables['max_coherence'], 0.5, places=10)
        
        # Test coherence L1 norm
        expected_l1_norm = 1.0  # For equal superposition
        self.assertAlmostEqual(observables['coherence_l1_norm'], expected_l1_norm, places=10)
    
    def test_validation_thermal_state_accuracy(self):
        """Test thermal state preparation accuracy for validation."""
        # Test thermal state at different temperatures
        temperatures = [77.0, 300.0, 500.0]  # Liquid nitrogen, room temp, elevated
        
        for temp in temperatures:
            with self.subTest(temperature=temp):
                thermal_state = self.engine.initialize_state(
                    self.quantum_subsystem, 
                    state_type="thermal", 
                    temperature=temp
                )
                
                # Verify thermal state properties
                self.assertAlmostEqual(np.trace(thermal_state.matrix), 1.0, places=10)
                
                # Check Hermiticity
                hermitian_diff = thermal_state.matrix - thermal_state.matrix.conj().T
                self.assertLess(np.max(np.abs(hermitian_diff)), 1e-10)
                
                # Check positive semidefinite
                eigenvals = np.linalg.eigvals(thermal_state.matrix)
                self.assertTrue(np.all(eigenvals >= -1e-10))
    
    def test_validation_hamiltonian_consistency(self):
        """Test Hamiltonian consistency for validation benchmarks."""
        # Verify Hamiltonian properties required for validation
        H = self.hamiltonian.matrix
        
        # Check Hermiticity
        hermitian_diff = H - H.conj().T
        self.assertLess(np.max(np.abs(hermitian_diff)), 1e-12)
        
        # Check eigenvalue structure for two-level system
        eigenvals, eigenvecs = np.linalg.eigh(H)
        
        # Verify ground state energy
        ground_energy = np.min(eigenvals)
        self.assertAlmostEqual(ground_energy, 0.0, places=10)
        
        # Verify energy gap
        energy_gap = np.max(eigenvals) - np.min(eigenvals)
        expected_gap = np.sqrt(1.0 + 4 * 0.1**2)  # For our test Hamiltonian
        self.assertAlmostEqual(energy_gap, expected_gap, places=10)
    
    def test_validation_numerical_precision(self):
        """Test numerical precision requirements for validation."""
        # Test with high-precision calculations
        state = self.engine.initialize_state(self.quantum_subsystem, state_type="ground")
        
        # Multiple calculations should be consistent
        observables1 = self.engine.calculate_observables(state, self.hamiltonian)
        observables2 = self.engine.calculate_observables(state, self.hamiltonian)
        
        # Verify reproducibility (important for validation)
        for key in ['purity', 'trace', 'energy', 'population_0']:
            self.assertAlmostEqual(
                observables1[key], observables2[key], 
                places=12, 
                msg=f"Observable {key} not reproducible"
            )
    
    def test_validation_edge_cases(self):
        """Test edge cases important for validation robustness."""
        # Test with minimal coupling (near-decoupled limit)
        minimal_coupling_subsystem = QuantumSubsystem(
            atoms=self.quantum_subsystem.atoms,
            hamiltonian_parameters={"coupling": 1e-6, "energy_gap": 1.0},
            coupling_matrix=np.array([[0.0, 1e-6], [1e-6, 1.0]], dtype=complex),
            basis_states=self.quantum_subsystem.basis_states,
            subsystem_id="minimal_coupling"
        )
        
        minimal_hamiltonian = Hamiltonian(
            matrix=np.array([[0.0, 1e-6], [1e-6, 1.0]], dtype=complex),
            basis_labels=["ground", "excited"],
            time_dependent=False
        )
        
        # Test state initialization with minimal coupling
        state = self.engine.initialize_state(minimal_coupling_subsystem, state_type="ground")
        observables = self.engine.calculate_observables(state, minimal_hamiltonian)
        
        # Should still maintain numerical accuracy
        self.assertAlmostEqual(observables['trace'], 1.0, places=10)
        self.assertAlmostEqual(observables['purity'], 1.0, places=10)


if __name__ == '__main__':
    unittest.main()