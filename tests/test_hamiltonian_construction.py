"""
Unit tests for Hamiltonian construction methods in the quantum engine.
"""

import pytest
import numpy as np
from qbes.quantum_engine import QuantumEngine
from qbes.core.data_models import (
    Hamiltonian, QuantumSubsystem, QuantumState, Atom
)


class TestHamiltonianConstruction:
    """Test Hamiltonian construction methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
        self.tolerance = 1e-10
    
    def test_create_two_level_hamiltonian(self):
        """Test creation of two-level system Hamiltonians."""
        # Simple two-level system
        energy_gap = 1.0  # eV
        coupling = 0.1    # eV
        
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=energy_gap,
            coupling=coupling
        )
        
        expected_matrix = np.array([
            [0.0, coupling],
            [coupling, energy_gap]
        ], dtype=complex)
        
        assert np.allclose(hamiltonian.matrix, expected_matrix)
        assert hamiltonian.basis_labels == ["ground", "excited"]
        assert not hamiltonian.time_dependent
        
        # Test with custom labels
        custom_labels = ["0", "1"]
        hamiltonian_custom = self.engine.create_two_level_hamiltonian(
            energy_gap=energy_gap,
            coupling=coupling,
            basis_labels=custom_labels
        )
        
        assert hamiltonian_custom.basis_labels == custom_labels
        
        # Test error cases
        with pytest.raises(ValueError, match="exactly 2 basis labels"):
            self.engine.create_two_level_hamiltonian(
                energy_gap=1.0,
                basis_labels=["0", "1", "2"]
            )
    
    def test_create_harmonic_oscillator_hamiltonian(self):
        """Test creation of harmonic oscillator Hamiltonians."""
        frequency = 0.1  # eV
        n_levels = 4
        
        hamiltonian = self.engine.create_harmonic_oscillator_hamiltonian(
            frequency=frequency,
            n_levels=n_levels
        )
        
        # Check diagonal elements: E_n = ℏω(n + 1/2)
        for n in range(n_levels):
            expected_energy = frequency * (n + 0.5)
            assert np.isclose(hamiltonian.matrix[n, n], expected_energy)
        
        # Check off-diagonal elements are zero
        for i in range(n_levels):
            for j in range(n_levels):
                if i != j:
                    assert np.isclose(hamiltonian.matrix[i, j], 0.0)
        
        # Check basis labels
        expected_labels = [f"n={i}" for i in range(n_levels)]
        assert hamiltonian.basis_labels == expected_labels
        
        # Test with custom labels
        custom_labels = ["ground", "first", "second", "third"]
        hamiltonian_custom = self.engine.create_harmonic_oscillator_hamiltonian(
            frequency=frequency,
            n_levels=n_levels,
            basis_labels=custom_labels
        )
        
        assert hamiltonian_custom.basis_labels == custom_labels
        
        # Test error cases
        with pytest.raises(ValueError, match="at least 2 levels"):
            self.engine.create_harmonic_oscillator_hamiltonian(frequency=1.0, n_levels=1)
        
        with pytest.raises(ValueError, match="must match n_levels"):
            self.engine.create_harmonic_oscillator_hamiltonian(
                frequency=1.0,
                n_levels=3,
                basis_labels=["0", "1"]
            )
    
    def test_create_multi_chromophore_hamiltonian(self):
        """Test creation of multi-chromophore system Hamiltonians."""
        # Three-chromophore system
        site_energies = [0.0, 0.1, 0.2]  # eV
        coupling_matrix = np.array([
            [0.0, 0.05, 0.02],
            [0.05, 0.0, 0.03],
            [0.02, 0.03, 0.0]
        ])
        
        hamiltonian = self.engine.create_multi_chromophore_hamiltonian(
            site_energies=site_energies,
            coupling_matrix=coupling_matrix
        )
        
        # Check diagonal elements (site energies)
        for i, energy in enumerate(site_energies):
            assert np.isclose(hamiltonian.matrix[i, i], energy)
        
        # Check off-diagonal elements (couplings)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert np.isclose(hamiltonian.matrix[i, j], coupling_matrix[i, j])
        
        # Check Hermiticity
        assert np.allclose(hamiltonian.matrix, hamiltonian.matrix.conj().T)
        
        # Check basis labels
        expected_labels = ["site_0", "site_1", "site_2"]
        assert hamiltonian.basis_labels == expected_labels
        
        # Test with custom labels
        custom_labels = ["Chl_a", "Chl_b", "Car"]
        hamiltonian_custom = self.engine.create_multi_chromophore_hamiltonian(
            site_energies=site_energies,
            coupling_matrix=coupling_matrix,
            basis_labels=custom_labels
        )
        
        assert hamiltonian_custom.basis_labels == custom_labels
        
        # Test error cases
        with pytest.raises(ValueError, match="doesn't match number of sites"):
            self.engine.create_multi_chromophore_hamiltonian(
                site_energies=[0.0, 0.1],
                coupling_matrix=np.eye(3)
            )
        
        with pytest.raises(ValueError, match="must match number of sites"):
            self.engine.create_multi_chromophore_hamiltonian(
                site_energies=site_energies,
                coupling_matrix=coupling_matrix,
                basis_labels=["site_0", "site_1"]
            )
    
    def test_create_time_dependent_hamiltonian(self):
        """Test creation of time-dependent Hamiltonians."""
        # Base Hamiltonian (two-level system)
        base_hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=1.0,
            coupling=0.0
        )
        
        # Time-dependent perturbation: oscillating field
        perturbation = np.array([[0.0, 0.1], [0.1, 0.0]], dtype=complex)
        
        def oscillating_field(t):
            return np.cos(2.0 * t)  # Oscillating at frequency 2
        
        time_dependent_terms = [(perturbation, oscillating_field)]
        
        hamiltonian_func = self.engine.create_time_dependent_hamiltonian(
            base_hamiltonian=base_hamiltonian,
            time_dependent_terms=time_dependent_terms
        )
        
        # Test at t=0
        H_t0 = hamiltonian_func(0.0)
        expected_t0 = base_hamiltonian.matrix + perturbation * oscillating_field(0.0)
        assert np.allclose(H_t0, expected_t0)
        
        # Test at t=π/4 (cos(π/2) = 0)
        H_t_pi4 = hamiltonian_func(np.pi/4)
        expected_t_pi4 = base_hamiltonian.matrix + perturbation * oscillating_field(np.pi/4)
        assert np.allclose(H_t_pi4, expected_t_pi4)
        
        # Test error case: mismatched operator size
        wrong_size_operator = np.array([[1.0]], dtype=complex)
        wrong_terms = [(wrong_size_operator, oscillating_field)]
        
        wrong_hamiltonian_func = self.engine.create_time_dependent_hamiltonian(
            base_hamiltonian=base_hamiltonian,
            time_dependent_terms=wrong_terms
        )
        
        with pytest.raises(ValueError, match="doesn't match base Hamiltonian"):
            wrong_hamiltonian_func(0.0)
    
    def test_calculate_coupling_from_dipole_interaction(self):
        """Test dipole-dipole coupling calculation."""
        # Two chromophores separated along x-axis
        positions = np.array([
            [0.0, 0.0, 0.0],  # First chromophore at origin
            [10.0, 0.0, 0.0]  # Second chromophore 10 Å away
        ])
        
        # Parallel dipole moments along z-axis
        dipole_moments = np.array([
            [0.0, 0.0, 1.0],  # 1 Debye in z-direction
            [0.0, 0.0, 1.0]   # 1 Debye in z-direction
        ])
        
        coupling_matrix = self.engine.calculate_coupling_from_dipole_interaction(
            positions=positions,
            dipole_moments=dipole_moments,
            dielectric_constant=1.0
        )
        
        # Check symmetry
        assert np.allclose(coupling_matrix, coupling_matrix.T)
        
        # Check diagonal elements are zero
        assert np.allclose(np.diag(coupling_matrix), 0.0)
        
        # Check coupling strength (should be positive for parallel dipoles)
        coupling_strength = coupling_matrix[0, 1]
        assert coupling_strength > 0
        
        # Test with perpendicular dipoles (should give different coupling)
        dipole_moments_perp = np.array([
            [0.0, 0.0, 1.0],  # z-direction
            [0.0, 1.0, 0.0]   # y-direction
        ])
        
        coupling_matrix_perp = self.engine.calculate_coupling_from_dipole_interaction(
            positions=positions,
            dipole_moments=dipole_moments_perp,
            dielectric_constant=1.0
        )
        
        # Perpendicular dipoles should give different coupling
        assert not np.isclose(coupling_matrix[0, 1], coupling_matrix_perp[0, 1])
        
        # Test error case: mismatched array shapes
        with pytest.raises(ValueError, match="must match positions"):
            self.engine.calculate_coupling_from_dipole_interaction(
                positions=positions,
                dipole_moments=np.array([[1.0, 0.0, 0.0]]),  # Wrong shape
                dielectric_constant=1.0
            )
    
    def test_diagonalize_hamiltonian(self):
        """Test Hamiltonian diagonalization."""
        # Create a simple 2x2 Hamiltonian
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=1.0,
            coupling=0.1
        )
        
        eigenvalues, eigenvectors = self.engine.diagonalize_hamiltonian(hamiltonian)
        
        # Check that eigenvalues are sorted
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])
        
        # Check that eigenvectors are orthonormal
        assert np.allclose(eigenvectors.conj().T @ eigenvectors, np.eye(2))
        
        # Check that H * v_i = λ_i * v_i
        for i in range(2):
            Hv = hamiltonian.matrix @ eigenvectors[:, i]
            lambda_v = eigenvalues[i] * eigenvectors[:, i]
            assert np.allclose(Hv, lambda_v)
        
        # Test with harmonic oscillator (should have known eigenvalues)
        ho_hamiltonian = self.engine.create_harmonic_oscillator_hamiltonian(
            frequency=0.1,
            n_levels=3
        )
        
        ho_eigenvalues, ho_eigenvectors = self.engine.diagonalize_hamiltonian(ho_hamiltonian)
        
        # Harmonic oscillator eigenvalues should be ℏω(n + 1/2)
        expected_eigenvalues = [0.1 * (n + 0.5) for n in range(3)]
        assert np.allclose(ho_eigenvalues, expected_eigenvalues)
    
    def test_transform_hamiltonian_basis(self):
        """Test basis transformation of Hamiltonians."""
        # Create original Hamiltonian
        original_hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=1.0,
            coupling=0.1
        )
        
        # Create unitary transformation (rotation by π/4)
        angle = np.pi / 4
        transformation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=complex)
        
        new_basis_labels = ["rotated_0", "rotated_1"]
        
        transformed_hamiltonian = self.engine.transform_hamiltonian_basis(
            hamiltonian=original_hamiltonian,
            transformation_matrix=transformation_matrix,
            new_basis_labels=new_basis_labels
        )
        
        # Check that transformation preserves eigenvalues
        orig_eigenvals, _ = self.engine.diagonalize_hamiltonian(original_hamiltonian)
        trans_eigenvals, _ = self.engine.diagonalize_hamiltonian(transformed_hamiltonian)
        
        assert np.allclose(np.sort(orig_eigenvals), np.sort(trans_eigenvals))
        
        # Check new basis labels
        assert transformed_hamiltonian.basis_labels == new_basis_labels
        
        # Check that transformation is unitary (preserves trace)
        orig_trace = np.trace(original_hamiltonian.matrix)
        trans_trace = np.trace(transformed_hamiltonian.matrix)
        assert np.isclose(orig_trace, trans_trace)
        
        # Test error cases
        with pytest.raises(ValueError, match="doesn't match Hamiltonian"):
            self.engine.transform_hamiltonian_basis(
                hamiltonian=original_hamiltonian,
                transformation_matrix=np.eye(3),  # Wrong size
                new_basis_labels=["0", "1"]
            )
        
        with pytest.raises(ValueError, match="doesn't match Hamiltonian size"):
            self.engine.transform_hamiltonian_basis(
                hamiltonian=original_hamiltonian,
                transformation_matrix=transformation_matrix,
                new_basis_labels=["0"]  # Wrong number of labels
            )
        
        # Test non-unitary matrix
        non_unitary = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=complex)
        with pytest.raises(ValueError, match="not unitary"):
            self.engine.transform_hamiltonian_basis(
                hamiltonian=original_hamiltonian,
                transformation_matrix=non_unitary,
                new_basis_labels=new_basis_labels
            )
    
    def test_initialize_hamiltonian_from_subsystem(self):
        """Test Hamiltonian initialization from QuantumSubsystem."""
        # Create a simple quantum subsystem
        atoms = [
            Atom(element="C", position=np.array([0.0, 0.0, 0.0]), 
                 charge=0.0, mass=12.0, atom_id=1),
            Atom(element="N", position=np.array([1.0, 0.0, 0.0]), 
                 charge=0.0, mass=14.0, atom_id=2)
        ]
        
        # Create basis states
        basis_states = [
            QuantumState(coefficients=np.array([1.0, 0.0]), basis_labels=["ground", "excited"]),
            QuantumState(coefficients=np.array([0.0, 1.0]), basis_labels=["ground", "excited"])
        ]
        
        # Coupling matrix
        coupling_matrix = np.array([
            [0.0, 0.1],
            [0.1, 1.0]
        ])
        
        subsystem = QuantumSubsystem(
            atoms=atoms,
            hamiltonian_parameters={"coupling_strength": 0.1},
            coupling_matrix=coupling_matrix,
            basis_states=basis_states
        )
        
        hamiltonian = self.engine.initialize_hamiltonian(subsystem)
        
        # Check that Hamiltonian uses the coupling matrix
        assert np.allclose(hamiltonian.matrix, coupling_matrix)
        
        # Check basis labels
        expected_labels = ["state_0", "state_1"]
        assert hamiltonian.basis_labels == expected_labels
        
        # Test error case: mismatched sizes
        # Create a valid subsystem first, then modify it to be invalid
        wrong_coupling = np.array([[1.0]])  # Wrong size
        
        # Create a copy of the valid subsystem and modify its coupling matrix
        subsystem_copy = QuantumSubsystem(
            atoms=atoms,
            hamiltonian_parameters={},
            coupling_matrix=coupling_matrix.copy(),  # Start with valid matrix
            basis_states=basis_states
        )
        
        # Manually set the wrong coupling matrix to bypass validation
        subsystem_copy.coupling_matrix = wrong_coupling
        
        with pytest.raises(ValueError, match="doesn't match number of basis states"):
            self.engine.initialize_hamiltonian(subsystem_copy)


class TestAnalyticalSolutions:
    """Test Hamiltonian construction against known analytical solutions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = QuantumEngine()
    
    def test_two_level_system_eigenvalues(self):
        """Test two-level system eigenvalues against analytical solution."""
        energy_gap = 1.0
        coupling = 0.2
        
        hamiltonian = self.engine.create_two_level_hamiltonian(
            energy_gap=energy_gap,
            coupling=coupling
        )
        
        eigenvalues, _ = self.engine.diagonalize_hamiltonian(hamiltonian)
        
        # Analytical eigenvalues for 2x2 matrix [[0, V], [V, Δ]]
        # λ = (Δ ± √(Δ² + 4V²)) / 2
        delta = energy_gap
        V = coupling
        
        lambda_minus = (delta - np.sqrt(delta**2 + 4*V**2)) / 2
        lambda_plus = (delta + np.sqrt(delta**2 + 4*V**2)) / 2
        
        expected_eigenvalues = np.array([lambda_minus, lambda_plus])
        
        assert np.allclose(eigenvalues, expected_eigenvalues, rtol=1e-10)
    
    def test_harmonic_oscillator_energy_levels(self):
        """Test harmonic oscillator energy levels."""
        frequency = 0.05  # eV
        n_levels = 5
        
        hamiltonian = self.engine.create_harmonic_oscillator_hamiltonian(
            frequency=frequency,
            n_levels=n_levels
        )
        
        eigenvalues, _ = self.engine.diagonalize_hamiltonian(hamiltonian)
        
        # Analytical eigenvalues: E_n = ℏω(n + 1/2)
        expected_eigenvalues = np.array([frequency * (n + 0.5) for n in range(n_levels)])
        
        assert np.allclose(eigenvalues, expected_eigenvalues, rtol=1e-10)
    
    def test_three_level_system_symmetry(self):
        """Test symmetric three-level system."""
        # Symmetric three-level system with equal couplings
        site_energies = [0.0, 0.0, 0.0]  # All sites at same energy
        coupling_strength = 0.1
        coupling_matrix = np.array([
            [0.0, coupling_strength, coupling_strength],
            [coupling_strength, 0.0, coupling_strength],
            [coupling_strength, coupling_strength, 0.0]
        ])
        
        hamiltonian = self.engine.create_multi_chromophore_hamiltonian(
            site_energies=site_energies,
            coupling_matrix=coupling_matrix
        )
        
        eigenvalues, _ = self.engine.diagonalize_hamiltonian(hamiltonian)
        
        # For symmetric system, expect one eigenvalue at 2V and two at -V
        expected_eigenvalues = np.sort([-coupling_strength, -coupling_strength, 2*coupling_strength])
        
        assert np.allclose(np.sort(eigenvalues), expected_eigenvalues, rtol=1e-10)
    
    def test_dipole_coupling_distance_dependence(self):
        """Test that dipole coupling follows 1/r³ dependence."""
        # Two identical chromophores at different distances
        dipole_moments = np.array([
            [0.0, 0.0, 1.0],  # Both along z-axis
            [0.0, 0.0, 1.0]
        ])
        
        distances = [5.0, 10.0, 20.0]  # Å
        couplings = []
        
        for distance in distances:
            positions = np.array([
                [0.0, 0.0, 0.0],
                [distance, 0.0, 0.0]
            ])
            
            coupling_matrix = self.engine.calculate_coupling_from_dipole_interaction(
                positions=positions,
                dipole_moments=dipole_moments,
                dielectric_constant=1.0
            )
            
            couplings.append(coupling_matrix[0, 1])
        
        # Check 1/r³ scaling
        for i in range(1, len(distances)):
            ratio_expected = (distances[0] / distances[i])**3
            ratio_actual = couplings[i] / couplings[0]
            assert np.isclose(ratio_actual, ratio_expected, rtol=0.01)