"""
Quantum mechanical calculations and state evolution.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.linalg import expm, logm, sqrtm
from scipy.integrate import solve_ivp

from .core.interfaces import QuantumEngineInterface
from .core.data_models import (
    QuantumSubsystem, DensityMatrix, Hamiltonian, LindbladOperator,
    CoherenceMetrics, ValidationResult, QuantumState
)


class QuantumEngine(QuantumEngineInterface):
    """
    Handles quantum mechanical calculations using open quantum systems theory.
    
    This class implements quantum state evolution using the Lindblad master equation
    and provides methods for calculating quantum coherence measures.
    """
    
    def __init__(self):
        """Initialize the quantum engine."""
        self.current_hamiltonian = None
        self.lindblad_operators = []
        self.tolerance = 1e-12  # Numerical tolerance for validations
    
    def initialize_hamiltonian(self, system: QuantumSubsystem) -> Hamiltonian:
        """Construct the system Hamiltonian from molecular structure."""
        # Use the coupling matrix from the quantum subsystem
        # This is a simplified implementation - in practice would involve
        # more complex electronic structure calculations
        
        if system.coupling_matrix.shape[0] != len(system.basis_states):
            raise ValueError("Coupling matrix size doesn't match number of basis states")
        
        # For now, use the coupling matrix directly as the Hamiltonian
        # In a full implementation, this would include:
        # - Electronic structure calculations
        # - Vibronic coupling terms
        # - Environmental field effects
        
        return Hamiltonian(
            matrix=system.coupling_matrix.copy(),
            basis_labels=[f"state_{i}" for i in range(len(system.basis_states))],
            time_dependent=False
        )
    
    def evolve_state(self, initial_state: DensityMatrix, time_step: float, 
                    hamiltonian: Hamiltonian, 
                    lindblad_operators: List[LindbladOperator]) -> DensityMatrix:
        """Evolve quantum state by one time step using Lindblad master equation."""
        # Use the Lindblad master equation solver
        final_time = initial_state.time + time_step
        
        # For time-independent case, use simple integration
        if not hamiltonian.time_dependent:
            evolved_state = self._evolve_lindblad_fixed_step(
                initial_state, hamiltonian, lindblad_operators, time_step
            )
        else:
            # For time-dependent case, would need more sophisticated integration
            raise NotImplementedError("Time-dependent Hamiltonian evolution not yet implemented")
        
        # Update time
        evolved_state.time = final_time
        return evolved_state
    
    def calculate_coherence_measures(self, state: DensityMatrix) -> CoherenceMetrics:
        """Calculate various quantum coherence measures from density matrix."""
        purity = self.calculate_purity(state)
        von_neumann_entropy = self.calculate_von_neumann_entropy(state)
        
        # For now, set placeholder values for measures that require more complex calculations
        # These will be implemented in later tasks
        coherence_lifetime = 1.0  # Will be calculated from time evolution
        quantum_discord = 0.0     # Requires bipartite system analysis
        entanglement_measure = 0.0  # Requires entanglement calculation
        
        return CoherenceMetrics(
            coherence_lifetime=coherence_lifetime,
            quantum_discord=quantum_discord,
            entanglement_measure=entanglement_measure,
            purity=purity,
            von_neumann_entropy=von_neumann_entropy
        )
    
    def apply_lindblad_operators(self, state: DensityMatrix, 
                                operators: List[LindbladOperator]) -> DensityMatrix:
        """Apply Lindblad operators to represent environmental decoherence."""
        # Apply the dissipative part of the Lindblad equation
        # L[ρ] = Σᵢ γᵢ (LᵢρLᵢ† - ½{Lᵢ†Lᵢ, ρ})
        
        rho = state.matrix.copy().astype(complex)
        
        for lindblad_op in operators:
            L = lindblad_op.operator
            gamma = lindblad_op.coupling_strength
            
            # Calculate Lᵢ†Lᵢ
            L_dagger_L = L.conj().T @ L
            
            # Apply dissipator: γ(LρL† - ½{L†L, ρ})
            dissipator = gamma * (
                L @ rho @ L.conj().T - 
                0.5 * (L_dagger_L @ rho + rho @ L_dagger_L)
            )
            
            rho += dissipator
        
        return DensityMatrix(
            matrix=rho,
            basis_labels=state.basis_labels,
            time=state.time
        )
    
    def validate_quantum_state(self, state: DensityMatrix) -> ValidationResult:
        """Validate that quantum state satisfies physical constraints."""
        result = ValidationResult(is_valid=True)
        
        # Check trace = 1
        trace_val = self.trace_density_matrix(state)
        if not np.isclose(trace_val, 1.0, rtol=self.tolerance):
            result.add_error(f"Density matrix trace is {trace_val}, should be 1.0")
        
        # Check Hermiticity
        if not np.allclose(state.matrix, state.matrix.conj().T, rtol=self.tolerance):
            result.add_error("Density matrix is not Hermitian")
        
        # Check positive semidefinite
        eigenvals = np.linalg.eigvals(state.matrix)
        if np.any(eigenvals < -self.tolerance):
            result.add_error(f"Density matrix has negative eigenvalues: {eigenvals[eigenvals < 0]}")
        
        # Check purity bounds
        purity = self.calculate_purity(state)
        dimension = state.matrix.shape[0]
        min_purity = 1.0 / dimension
        if purity < min_purity - self.tolerance or purity > 1.0 + self.tolerance:
            result.add_warning(f"Purity {purity} outside expected range [{min_purity}, 1.0]")
        
        return result
    
    def calculate_expectation_value(self, state: DensityMatrix, 
                                   operator: np.ndarray) -> complex:
        """Calculate expectation value of an operator."""
        # Placeholder implementation
        raise NotImplementedError("Expectation value calculation not yet implemented")
    
    # ===== QUANTUM STATE INITIALIZATION AND MANIPULATION =====
    
    def create_pure_state(self, coefficients: np.ndarray, 
                         basis_labels: List[str]) -> QuantumState:
        """
        Create a pure quantum state from coefficients.
        
        Args:
            coefficients: Complex amplitudes for each basis state
            basis_labels: Labels for the basis states
            
        Returns:
            QuantumState object representing the pure state
        """
        # Normalize the coefficients
        norm = np.linalg.norm(coefficients)
        if norm == 0:
            raise ValueError("Cannot create state with zero norm")
        
        normalized_coeffs = coefficients / norm
        return QuantumState(
            coefficients=normalized_coeffs,
            basis_labels=basis_labels
        )
    
    def pure_state_to_density_matrix(self, state: QuantumState, 
                                   time: float = 0.0) -> DensityMatrix:
        """
        Convert a pure state to its density matrix representation.
        
        Args:
            state: Pure quantum state
            time: Time stamp for the density matrix
            
        Returns:
            DensityMatrix representation of the pure state
        """
        psi = state.coefficients.reshape(-1, 1)
        rho = psi @ psi.conj().T
        
        return DensityMatrix(
            matrix=rho,
            basis_labels=state.basis_labels,
            time=time
        )
    
    def create_maximally_mixed_state(self, dimension: int, 
                                   basis_labels: List[str],
                                   time: float = 0.0) -> DensityMatrix:
        """
        Create a maximally mixed state (identity matrix / dimension).
        
        Args:
            dimension: Dimension of the Hilbert space
            basis_labels: Labels for the basis states
            time: Time stamp for the density matrix
            
        Returns:
            DensityMatrix representing maximally mixed state
        """
        if len(basis_labels) != dimension:
            raise ValueError("Number of basis labels must match dimension")
        
        rho = np.eye(dimension) / dimension
        
        return DensityMatrix(
            matrix=rho,
            basis_labels=basis_labels,
            time=time
        )
    
    def create_thermal_state(self, hamiltonian: Hamiltonian, 
                           temperature: float,
                           time: float = 0.0) -> DensityMatrix:
        """
        Create a thermal (Gibbs) state at given temperature.
        
        Args:
            hamiltonian: System Hamiltonian
            temperature: Temperature in Kelvin
            time: Time stamp for the density matrix
            
        Returns:
            DensityMatrix representing thermal state
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Boltzmann constant in eV/K
        k_B = 8.617333e-5
        beta = 1.0 / (k_B * temperature)
        
        # Calculate thermal state: rho = exp(-beta * H) / Z
        exp_neg_beta_H = expm(-beta * hamiltonian.matrix)
        partition_function = np.trace(exp_neg_beta_H)
        
        if partition_function == 0:
            raise ValueError("Partition function is zero - invalid Hamiltonian or temperature")
        
        rho = exp_neg_beta_H / partition_function
        
        return DensityMatrix(
            matrix=rho,
            basis_labels=hamiltonian.basis_labels,
            time=time
        )
    
    # ===== DENSITY MATRIX OPERATIONS =====
    
    def trace_density_matrix(self, state: DensityMatrix) -> float:
        """
        Calculate the trace of a density matrix.
        
        Args:
            state: Density matrix
            
        Returns:
            Trace value (should be 1 for valid density matrices)
        """
        return np.real(np.trace(state.matrix))
    
    def partial_trace(self, state: DensityMatrix, 
                     subsystem_indices: List[int],
                     subsystem_dimensions: List[int]) -> DensityMatrix:
        """
        Calculate partial trace over specified subsystems.
        
        Args:
            state: Bipartite or multipartite density matrix
            subsystem_indices: Indices of subsystems to trace out
            subsystem_dimensions: Dimensions of each subsystem
            
        Returns:
            Reduced density matrix after partial trace
        """
        if len(subsystem_dimensions) < 2:
            raise ValueError("Need at least 2 subsystems for partial trace")
        
        total_dim = np.prod(subsystem_dimensions)
        if state.matrix.shape[0] != total_dim:
            raise ValueError("Matrix dimension doesn't match subsystem dimensions")
        
        # Reshape density matrix to tensor form
        shape = subsystem_dimensions + subsystem_dimensions
        rho_tensor = state.matrix.reshape(shape)
        
        # Trace out specified subsystems
        for idx in sorted(subsystem_indices, reverse=True):
            # Sum over diagonal elements of subsystem idx
            rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + len(subsystem_dimensions))
            # Remove traced subsystem from dimensions list
            subsystem_dimensions.pop(idx)
        
        # Reshape back to matrix form
        remaining_dim = np.prod(subsystem_dimensions)
        reduced_matrix = rho_tensor.reshape(remaining_dim, remaining_dim)
        
        # Create new basis labels for reduced system
        reduced_labels = [f"reduced_{i}" for i in range(remaining_dim)]
        
        return DensityMatrix(
            matrix=reduced_matrix,
            basis_labels=reduced_labels,
            time=state.time
        )
    
    def matrix_power(self, state: DensityMatrix, power: float) -> np.ndarray:
        """
        Calculate matrix power of density matrix.
        
        Args:
            state: Density matrix
            power: Power to raise matrix to
            
        Returns:
            Matrix raised to the specified power
        """
        eigenvals, eigenvecs = np.linalg.eigh(state.matrix)
        
        # Handle numerical precision issues
        eigenvals = np.maximum(eigenvals, 0)
        
        # Calculate power
        powered_eigenvals = np.power(eigenvals, power)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(powered_eigenvals) @ eigenvecs.conj().T
    
    def fidelity(self, state1: DensityMatrix, state2: DensityMatrix) -> float:
        """
        Calculate quantum fidelity between two density matrices.
        
        Args:
            state1: First density matrix
            state2: Second density matrix
            
        Returns:
            Fidelity value between 0 and 1
        """
        if state1.matrix.shape != state2.matrix.shape:
            raise ValueError("Density matrices must have same dimensions")
        
        # F = Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))
        sqrt_rho1 = sqrtm(state1.matrix)
        intermediate = sqrt_rho1 @ state2.matrix @ sqrt_rho1
        sqrt_intermediate = sqrtm(intermediate)
        
        fidelity_val = np.real(np.trace(sqrt_intermediate))
        
        # Ensure fidelity is in valid range [0, 1]
        return np.clip(fidelity_val, 0.0, 1.0)
    
    # ===== QUANTUM COHERENCE MEASURES =====
    
    def calculate_purity(self, state: DensityMatrix) -> float:
        """
        Calculate purity of a quantum state: Tr(rho^2).
        
        Args:
            state: Density matrix
            
        Returns:
            Purity value between 1/d and 1 (d = dimension)
        """
        rho_squared = state.matrix @ state.matrix
        purity = np.real(np.trace(rho_squared))
        
        # Ensure purity is in valid range
        dimension = state.matrix.shape[0]
        return np.clip(purity, 1.0/dimension, 1.0)
    
    def calculate_von_neumann_entropy(self, state: DensityMatrix) -> float:
        """
        Calculate von Neumann entropy: -Tr(rho * log(rho)).
        
        Args:
            state: Density matrix
            
        Returns:
            von Neumann entropy (non-negative)
        """
        eigenvals = np.linalg.eigvals(state.matrix)
        
        # Remove zero eigenvalues to avoid log(0)
        eigenvals = eigenvals[eigenvals > self.tolerance]
        
        if len(eigenvals) == 0:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return np.real(entropy)
    
    def calculate_linear_entropy(self, state: DensityMatrix) -> float:
        """
        Calculate linear entropy: 1 - Tr(rho^2).
        
        Args:
            state: Density matrix
            
        Returns:
            Linear entropy between 0 and (d-1)/d
        """
        purity = self.calculate_purity(state)
        return 1.0 - purity
    
    def calculate_coherence_l1_norm(self, state: DensityMatrix) -> float:
        """
        Calculate l1-norm coherence measure.
        
        Args:
            state: Density matrix
            
        Returns:
            l1-norm coherence (sum of absolute values of off-diagonal elements)
        """
        # Extract off-diagonal elements
        rho = state.matrix
        off_diagonal = rho - np.diag(np.diag(rho))
        
        # Calculate l1 norm
        coherence = np.sum(np.abs(off_diagonal))
        return np.real(coherence)
    
    def calculate_relative_entropy_coherence(self, state: DensityMatrix) -> float:
        """
        Calculate relative entropy of coherence.
        
        Args:
            state: Density matrix
            
        Returns:
            Relative entropy coherence measure
        """
        # Create incoherent state (diagonal part only)
        diagonal_state = DensityMatrix(
            matrix=np.diag(np.diag(state.matrix)),
            basis_labels=state.basis_labels,
            time=state.time
        )
        
        # Calculate relative entropy S(rho_diag) - S(rho)
        entropy_original = self.calculate_von_neumann_entropy(state)
        entropy_diagonal = self.calculate_von_neumann_entropy(diagonal_state)
        
        return entropy_diagonal - entropy_original
    
    # ===== HAMILTONIAN CONSTRUCTION =====
    
    def create_two_level_hamiltonian(self, energy_gap: float, 
                                   coupling: float = 0.0,
                                   basis_labels: Optional[List[str]] = None) -> Hamiltonian:
        """
        Create a two-level system Hamiltonian.
        
        Args:
            energy_gap: Energy difference between levels (in eV)
            coupling: Off-diagonal coupling strength (in eV)
            basis_labels: Labels for the two levels
            
        Returns:
            Hamiltonian for the two-level system
        """
        if basis_labels is None:
            basis_labels = ["ground", "excited"]
        
        if len(basis_labels) != 2:
            raise ValueError("Two-level system requires exactly 2 basis labels")
        
        # Construct Hamiltonian matrix
        H_matrix = np.array([
            [0.0, coupling],
            [coupling, energy_gap]
        ], dtype=complex)
        
        return Hamiltonian(
            matrix=H_matrix,
            basis_labels=basis_labels,
            time_dependent=False
        )
    
    def create_harmonic_oscillator_hamiltonian(self, frequency: float, 
                                             n_levels: int,
                                             basis_labels: Optional[List[str]] = None) -> Hamiltonian:
        """
        Create a truncated harmonic oscillator Hamiltonian.
        
        Args:
            frequency: Oscillator frequency (in eV)
            n_levels: Number of energy levels to include
            basis_labels: Labels for the energy levels
            
        Returns:
            Hamiltonian for the harmonic oscillator
        """
        if n_levels < 2:
            raise ValueError("Need at least 2 levels for harmonic oscillator")
        
        if basis_labels is None:
            basis_labels = [f"n={i}" for i in range(n_levels)]
        
        if len(basis_labels) != n_levels:
            raise ValueError(f"Number of basis labels ({len(basis_labels)}) must match n_levels ({n_levels})")
        
        # Diagonal Hamiltonian: H_nn = ℏω(n + 1/2)
        # Setting ℏ = 1 for simplicity
        H_matrix = np.zeros((n_levels, n_levels), dtype=complex)
        for n in range(n_levels):
            H_matrix[n, n] = frequency * (n + 0.5)
        
        return Hamiltonian(
            matrix=H_matrix,
            basis_labels=basis_labels,
            time_dependent=False
        )
    
    def create_multi_chromophore_hamiltonian(self, site_energies: List[float],
                                           coupling_matrix: np.ndarray,
                                           basis_labels: Optional[List[str]] = None) -> Hamiltonian:
        """
        Create Hamiltonian for multi-chromophore system (e.g., photosynthetic complex).
        
        Args:
            site_energies: On-site energies for each chromophore (in eV)
            coupling_matrix: Inter-chromophore coupling matrix (in eV)
            basis_labels: Labels for the chromophore sites
            
        Returns:
            Hamiltonian for the multi-chromophore system
        """
        n_sites = len(site_energies)
        
        if coupling_matrix.shape != (n_sites, n_sites):
            raise ValueError(f"Coupling matrix shape {coupling_matrix.shape} doesn't match number of sites {n_sites}")
        
        if basis_labels is None:
            basis_labels = [f"site_{i}" for i in range(n_sites)]
        
        if len(basis_labels) != n_sites:
            raise ValueError(f"Number of basis labels ({len(basis_labels)}) must match number of sites ({n_sites})")
        
        # Construct Hamiltonian: H_ij = ε_i δ_ij + V_ij (1 - δ_ij)
        H_matrix = coupling_matrix.copy().astype(complex)
        
        # Set diagonal elements to site energies
        for i in range(n_sites):
            H_matrix[i, i] = site_energies[i]
        
        # Ensure Hermiticity
        H_matrix = (H_matrix + H_matrix.conj().T) / 2
        
        return Hamiltonian(
            matrix=H_matrix,
            basis_labels=basis_labels,
            time_dependent=False
        )
    
    def create_time_dependent_hamiltonian(self, base_hamiltonian: Hamiltonian,
                                        time_dependent_terms: List[Tuple[np.ndarray, callable]]) -> callable:
        """
        Create a time-dependent Hamiltonian H(t) = H_0 + Σ_i f_i(t) * H_i.
        
        Args:
            base_hamiltonian: Time-independent part H_0
            time_dependent_terms: List of (H_i, f_i(t)) pairs where H_i are operators
                                and f_i(t) are time-dependent functions
                                
        Returns:
            Function that returns Hamiltonian matrix at given time
        """
        def hamiltonian_at_time(t: float) -> np.ndarray:
            """Return Hamiltonian matrix at time t."""
            H_t = base_hamiltonian.matrix.copy()
            
            for operator, time_function in time_dependent_terms:
                if operator.shape != base_hamiltonian.matrix.shape:
                    raise ValueError("Time-dependent operator shape doesn't match base Hamiltonian")
                
                H_t += time_function(t) * operator
            
            return H_t
        
        return hamiltonian_at_time
    
    def calculate_coupling_from_dipole_interaction(self, positions: np.ndarray,
                                                 dipole_moments: np.ndarray,
                                                 dielectric_constant: float = 1.0) -> np.ndarray:
        """
        Calculate dipole-dipole coupling matrix between chromophores.
        
        Args:
            positions: Array of chromophore positions (N x 3)
            dipole_moments: Array of transition dipole moments (N x 3)
            dielectric_constant: Relative dielectric constant of medium
            
        Returns:
            Coupling matrix (N x N) in eV
        """
        n_chromophores = positions.shape[0]
        
        if dipole_moments.shape != (n_chromophores, 3):
            raise ValueError("Dipole moments array shape must match positions")
        
        coupling_matrix = np.zeros((n_chromophores, n_chromophores))
        
        # Physical constants (in appropriate units)
        # Coulomb constant in eV·Å / e^2
        k_e = 14.3996  # eV·Å / e^2
        
        for i in range(n_chromophores):
            for j in range(i + 1, n_chromophores):
                # Distance vector
                r_ij = positions[j] - positions[i]
                r_distance = np.linalg.norm(r_ij)
                
                if r_distance == 0:
                    continue  # Skip self-interaction
                
                r_unit = r_ij / r_distance
                
                # Dipole-dipole interaction
                mu_i = dipole_moments[i]
                mu_j = dipole_moments[j]
                
                # V_ij = (k_e / ε) * [μ_i·μ_j / r^3 - 3(μ_i·r̂)(μ_j·r̂) / r^3]
                dot_product = np.dot(mu_i, mu_j)
                mu_i_dot_r = np.dot(mu_i, r_unit)
                mu_j_dot_r = np.dot(mu_j, r_unit)
                
                coupling = (k_e / dielectric_constant) * (
                    dot_product / r_distance**3 - 
                    3 * mu_i_dot_r * mu_j_dot_r / r_distance**3
                )
                
                coupling_matrix[i, j] = coupling
                coupling_matrix[j, i] = coupling  # Symmetric
        
        return coupling_matrix
    
    def diagonalize_hamiltonian(self, hamiltonian: Hamiltonian) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize Hamiltonian to find eigenvalues and eigenvectors.
        
        Args:
            hamiltonian: Hamiltonian to diagonalize
            
        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvectors[:, i] 
            is the eigenvector for eigenvalues[i]
        """
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.matrix)
        
        # Sort by eigenvalue (ascending)
        sort_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        return eigenvalues, eigenvectors
    
    def transform_hamiltonian_basis(self, hamiltonian: Hamiltonian,
                                  transformation_matrix: np.ndarray,
                                  new_basis_labels: List[str]) -> Hamiltonian:
        """
        Transform Hamiltonian to a new basis using unitary transformation.
        
        Args:
            hamiltonian: Original Hamiltonian
            transformation_matrix: Unitary transformation matrix U
            new_basis_labels: Labels for the new basis
            
        Returns:
            Transformed Hamiltonian H' = U† H U
        """
        if transformation_matrix.shape[0] != hamiltonian.matrix.shape[0]:
            raise ValueError("Transformation matrix size doesn't match Hamiltonian")
        
        if len(new_basis_labels) != hamiltonian.matrix.shape[0]:
            raise ValueError("Number of new basis labels doesn't match Hamiltonian size")
        
        # Check if transformation matrix is unitary
        identity_check = transformation_matrix @ transformation_matrix.conj().T
        if not np.allclose(identity_check, np.eye(transformation_matrix.shape[0])):
            raise ValueError("Transformation matrix is not unitary")
        
        # Transform: H' = U† H U
        transformed_matrix = (transformation_matrix.conj().T @ 
                            hamiltonian.matrix @ 
                            transformation_matrix)
        
        return Hamiltonian(
            matrix=transformed_matrix,
            basis_labels=new_basis_labels,
            time_dependent=hamiltonian.time_dependent
        )
    
    # ===== LINDBLAD MASTER EQUATION SOLVER =====
    
    def create_dephasing_operator(self, dimension: int, 
                                site_index: int,
                                coupling_strength: float) -> LindbladOperator:
        """
        Create a dephasing Lindblad operator for a specific site.
        
        Args:
            dimension: Dimension of the Hilbert space
            site_index: Index of the site to dephase (0-based)
            coupling_strength: Dephasing rate
            
        Returns:
            LindbladOperator for pure dephasing
        """
        if site_index >= dimension:
            raise ValueError(f"Site index {site_index} exceeds dimension {dimension}")
        
        # Create |i⟩⟨i| operator for pure dephasing
        operator = np.zeros((dimension, dimension), dtype=complex)
        operator[site_index, site_index] = 1.0
        
        return LindbladOperator(
            operator=operator,
            coupling_strength=coupling_strength,
            operator_type="dephasing"
        )
    
    def create_relaxation_operator(self, dimension: int,
                                 from_state: int,
                                 to_state: int,
                                 coupling_strength: float) -> LindbladOperator:
        """
        Create a relaxation Lindblad operator between two states.
        
        Args:
            dimension: Dimension of the Hilbert space
            from_state: Index of the initial state
            to_state: Index of the final state
            coupling_strength: Relaxation rate
            
        Returns:
            LindbladOperator for relaxation process
        """
        if from_state >= dimension or to_state >= dimension:
            raise ValueError("State indices exceed dimension")
        
        # Create |to⟩⟨from| operator for relaxation
        operator = np.zeros((dimension, dimension), dtype=complex)
        operator[to_state, from_state] = 1.0
        
        return LindbladOperator(
            operator=operator,
            coupling_strength=coupling_strength,
            operator_type="relaxation"
        )
    
    def create_thermal_lindblad_operators(self, hamiltonian: Hamiltonian,
                                        temperature: float,
                                        coupling_strength: float) -> List[LindbladOperator]:
        """
        Create Lindblad operators for thermal relaxation.
        
        Args:
            hamiltonian: System Hamiltonian
            temperature: Temperature in Kelvin
            coupling_strength: Overall coupling strength to thermal bath
            
        Returns:
            List of Lindblad operators for thermal processes
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Diagonalize Hamiltonian to get energy eigenstates
        eigenvalues, eigenvectors = self.diagonalize_hamiltonian(hamiltonian)
        dimension = len(eigenvalues)
        
        # Boltzmann constant in eV/K
        k_B = 8.617333e-5
        beta = 1.0 / (k_B * temperature)
        
        operators = []
        
        # Create relaxation operators between all pairs of states
        for i in range(dimension):
            for j in range(dimension):
                if i != j:
                    # Energy difference
                    energy_diff = eigenvalues[j] - eigenvalues[i]
                    
                    # Transition rate (detailed balance)
                    if energy_diff > 0:
                        # Absorption: rate ∝ n(ω) where n is Bose-Einstein distribution
                        rate = coupling_strength * np.exp(-beta * energy_diff)
                    else:
                        # Emission: rate ∝ (n(ω) + 1)
                        rate = coupling_strength * (1 + np.exp(beta * energy_diff))
                    
                    # Create transition operator in energy eigenbasis
                    transition_op = np.zeros((dimension, dimension), dtype=complex)
                    transition_op[j, i] = 1.0
                    
                    # Transform back to original basis
                    transition_op_original = (eigenvectors @ transition_op @ eigenvectors.conj().T)
                    
                    operators.append(LindbladOperator(
                        operator=transition_op_original,
                        coupling_strength=np.sqrt(rate),
                        operator_type="thermal"
                    ))
        
        return operators
    
    def _evolve_lindblad_fixed_step(self, initial_state: DensityMatrix,
                                  hamiltonian: Hamiltonian,
                                  lindblad_operators: List[LindbladOperator],
                                  time_step: float) -> DensityMatrix:
        """
        Evolve state using fixed time step integration of Lindblad equation.
        
        Args:
            initial_state: Initial density matrix
            hamiltonian: System Hamiltonian
            lindblad_operators: List of Lindblad operators
            time_step: Integration time step
            
        Returns:
            Evolved density matrix
        """
        # Lindblad master equation: dρ/dt = -i[H, ρ] + Σᵢ Lᵢ[ρ]
        # where Lᵢ[ρ] = γᵢ(LᵢρLᵢ† - ½{Lᵢ†Lᵢ, ρ})
        
        def lindblad_rhs(t, rho_vec):
            """Right-hand side of Lindblad equation (vectorized)."""
            # Reshape vector back to matrix
            dim = int(np.sqrt(len(rho_vec)))
            rho = rho_vec.reshape((dim, dim))
            
            # Unitary evolution: -i[H, ρ]
            commutator = -1j * (hamiltonian.matrix @ rho - rho @ hamiltonian.matrix)
            
            # Dissipative evolution
            dissipator = np.zeros_like(rho, dtype=complex)
            for lindblad_op in lindblad_operators:
                L = lindblad_op.operator
                gamma = lindblad_op.coupling_strength
                L_dagger_L = L.conj().T @ L
                
                dissipator += gamma * (
                    L @ rho @ L.conj().T - 
                    0.5 * (L_dagger_L @ rho + rho @ L_dagger_L)
                )
            
            # Total evolution
            drho_dt = commutator + dissipator
            
            # Return as vector
            return drho_dt.flatten()
        
        # Convert initial state to vector (ensure complex)
        rho_initial = initial_state.matrix.astype(complex).flatten()
        
        # Integrate using scipy's solve_ivp
        solution = solve_ivp(
            lindblad_rhs,
            [0, time_step],
            rho_initial,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        
        # Extract final state
        rho_final = solution.y[:, -1].reshape(initial_state.matrix.shape)
        
        # Ensure the result is a valid density matrix
        rho_final = self._ensure_valid_density_matrix(rho_final)
        
        return DensityMatrix(
            matrix=rho_final,
            basis_labels=initial_state.basis_labels,
            time=initial_state.time + time_step
        )
    
    def evolve_lindblad_adaptive(self, initial_state: DensityMatrix,
                               hamiltonian: Hamiltonian,
                               lindblad_operators: List[LindbladOperator],
                               total_time: float,
                               rtol: float = 1e-6,
                               atol: float = 1e-8) -> List[DensityMatrix]:
        """
        Evolve state using adaptive time stepping.
        
        Args:
            initial_state: Initial density matrix
            hamiltonian: System Hamiltonian
            lindblad_operators: List of Lindblad operators
            total_time: Total evolution time
            rtol: Relative tolerance for integration
            atol: Absolute tolerance for integration
            
        Returns:
            List of density matrices at integration time points
        """
        def lindblad_rhs(t, rho_vec):
            """Right-hand side of Lindblad equation (vectorized)."""
            dim = int(np.sqrt(len(rho_vec)))
            rho = rho_vec.reshape((dim, dim))
            
            # Unitary evolution
            commutator = -1j * (hamiltonian.matrix @ rho - rho @ hamiltonian.matrix)
            
            # Dissipative evolution
            dissipator = np.zeros_like(rho, dtype=complex)
            for lindblad_op in lindblad_operators:
                L = lindblad_op.operator
                gamma = lindblad_op.coupling_strength
                L_dagger_L = L.conj().T @ L
                
                dissipator += gamma * (
                    L @ rho @ L.conj().T - 
                    0.5 * (L_dagger_L @ rho + rho @ L_dagger_L)
                )
            
            drho_dt = commutator + dissipator
            return drho_dt.flatten()
        
        # Convert initial state to vector (ensure complex)
        rho_initial = initial_state.matrix.astype(complex).flatten()
        
        # Integrate with adaptive time stepping
        solution = solve_ivp(
            lindblad_rhs,
            [0, total_time],
            rho_initial,
            method='DOP853',  # High-order adaptive method
            rtol=rtol,
            atol=atol,
            dense_output=True
        )
        
        if not solution.success:
            raise RuntimeError(f"Integration failed: {solution.message}")
        
        # Convert solution back to density matrices
        trajectory = []
        for i, t in enumerate(solution.t):
            rho_t = solution.y[:, i].reshape(initial_state.matrix.shape)
            rho_t = self._ensure_valid_density_matrix(rho_t)
            
            trajectory.append(DensityMatrix(
                matrix=rho_t,
                basis_labels=initial_state.basis_labels,
                time=initial_state.time + t
            ))
        
        return trajectory
    
    def _ensure_valid_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """
        Ensure numerical result is a valid density matrix.
        
        Args:
            rho: Density matrix that may have numerical errors
            
        Returns:
            Corrected density matrix
        """
        # Ensure Hermiticity
        rho = (rho + rho.conj().T) / 2
        
        # Ensure trace = 1
        trace = np.trace(rho)
        if abs(trace) > self.tolerance:
            rho = rho / trace
        
        # Ensure positive semidefinite by eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        
        # Set negative eigenvalues to zero
        eigenvals = np.maximum(eigenvals, 0)
        
        # Renormalize
        if np.sum(eigenvals) > 0:
            eigenvals = eigenvals / np.sum(eigenvals)
        
        # Reconstruct matrix
        rho_corrected = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
        
        return rho_corrected
    
    def calculate_decoherence_time(self, initial_state: DensityMatrix,
                                 hamiltonian: Hamiltonian,
                                 lindblad_operators: List[LindbladOperator],
                                 coherence_threshold: float = 1/np.e) -> float:
        """
        Calculate decoherence time by evolving state until coherence drops below threshold.
        
        Args:
            initial_state: Initial density matrix
            hamiltonian: System Hamiltonian
            lindblad_operators: List of Lindblad operators
            coherence_threshold: Threshold for coherence decay (default: 1/e)
            
        Returns:
            Decoherence time
        """
        # Calculate initial coherence
        initial_coherence = self.calculate_coherence_l1_norm(initial_state)
        
        if initial_coherence == 0:
            return 0.0  # Already incoherent
        
        # Target coherence
        target_coherence = initial_coherence * coherence_threshold
        
        # Evolve state and monitor coherence
        current_state = initial_state
        time_step = 0.01  # Small time step
        max_time = 100.0  # Maximum time to search
        
        while current_state.time < max_time:
            current_state = self.evolve_state(
                current_state, time_step, hamiltonian, lindblad_operators
            )
            
            current_coherence = self.calculate_coherence_l1_norm(current_state)
            
            if current_coherence <= target_coherence:
                return current_state.time - initial_state.time
        
        # If coherence hasn't decayed enough, return maximum time
        return max_time