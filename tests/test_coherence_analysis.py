"""
Tests for quantum coherence analysis functionality.
"""

import pytest
import numpy as np
from qbes.analysis import ResultsAnalyzer
from qbes.core.data_models import DensityMatrix, CoherenceMetrics


class TestCoherenceAnalysis:
    """Test quantum coherence analysis methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ResultsAnalyzer()
        
        # Create test basis labels
        self.basis_2 = ['|0>', '|1>']
        self.basis_4 = ['|00>', '|01>', '|10>', '|11>']
    
    def create_pure_state_dm(self, coeffs: np.ndarray, basis: list, time: float = 0.0) -> DensityMatrix:
        """Create density matrix from pure state coefficients."""
        # Normalize coefficients
        coeffs = coeffs / np.linalg.norm(coeffs)
        # Create density matrix |ψ><ψ|
        rho = np.outer(coeffs, coeffs.conj())
        return DensityMatrix(matrix=rho, basis_labels=basis, time=time)
    
    def create_mixed_state_dm(self, eigenvals: np.ndarray, basis: list, time: float = 0.0) -> DensityMatrix:
        """Create mixed state density matrix with given eigenvalues."""
        n = len(eigenvals)
        # Normalize eigenvalues
        eigenvals = eigenvals / np.sum(eigenvals)
        
        # Create random unitary matrix for eigenvectors
        np.random.seed(42)  # For reproducibility
        U = self._random_unitary(n)
        
        # Construct density matrix
        rho = U @ np.diag(eigenvals) @ U.conj().T
        return DensityMatrix(matrix=rho, basis_labels=basis, time=time)
    
    def _random_unitary(self, n: int) -> np.ndarray:
        """Generate random unitary matrix using QR decomposition."""
        # Generate random complex matrix
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        # QR decomposition
        Q, R = np.linalg.qr(A)
        # Make it properly unitary
        D = np.diag(R) / np.abs(np.diag(R))
        return Q @ np.diag(D)
    
    def test_coherence_lifetime_pure_state(self):
        """Test coherence lifetime calculation for pure state evolution."""
        # Create superposition state that decoheres over time
        times = np.linspace(0, 5, 11)
        trajectory = []
        
        for t in times:
            # Exponential decoherence: |+> -> |0> with rate γ = 0.5
            gamma = 0.5
            coherence_factor = np.exp(-gamma * t)
            
            # Density matrix elements
            rho = np.array([
                [0.5, 0.5 * coherence_factor],
                [0.5 * coherence_factor, 0.5]
            ], dtype=complex)
            
            dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=t)
            trajectory.append(dm)
        
        # Calculate coherence lifetime
        lifetime = self.analyzer.calculate_coherence_lifetime(trajectory)
        
        # Should be approximately 1/γ = 2.0, but allow for numerical fitting differences
        # The coherence decays as |ρ₀₁|² = (0.5)² * exp(-2γt) = 0.25 * exp(-t)
        # So the effective decay rate is γ_eff = 1.0, giving lifetime = 1.0
        assert abs(lifetime - 1.0) < 0.3, f"Expected lifetime ~1.0, got {lifetime}"
    
    def test_coherence_lifetime_no_decay(self):
        """Test coherence lifetime for state with no decoherence."""
        # Create constant superposition state
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        
        trajectory = []
        for t in range(5):
            dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=float(t))
            trajectory.append(dm)
        
        lifetime = self.analyzer.calculate_coherence_lifetime(trajectory)
        
        # Should be infinite (no decay)
        assert lifetime == np.inf or lifetime > 100
    
    def test_coherence_lifetime_edge_cases(self):
        """Test edge cases for coherence lifetime calculation."""
        # Empty trajectory
        with pytest.raises(ValueError):
            self.analyzer.calculate_coherence_lifetime([])
        
        # Single state
        rho = np.eye(2) / 2
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=0.0)
        with pytest.raises(ValueError):
            self.analyzer.calculate_coherence_lifetime([dm])
    
    def test_quantum_discord_separable_state(self):
        """Test quantum discord for separable (classical) state."""
        # Create separable state |00><00|
        rho = np.zeros((4, 4))
        rho[0, 0] = 1.0
        
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_4, time=0.0)
        discord = self.analyzer.measure_quantum_discord(dm)
        
        # Separable state should have zero discord
        assert abs(discord) < 1e-10, f"Expected discord ~0, got {discord}"
    
    def test_quantum_discord_bell_state(self):
        """Test quantum discord for maximally entangled Bell state."""
        # Create Bell state |Φ+> = (|00> + |11>)/√2
        coeffs = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        dm = self.create_pure_state_dm(coeffs, self.basis_4)
        
        discord = self.analyzer.measure_quantum_discord(dm)
        
        # Bell state should have significant discord
        assert discord > 0.5, f"Expected discord > 0.5, got {discord}"
    
    def test_quantum_discord_invalid_dimensions(self):
        """Test quantum discord with invalid system dimensions."""
        # Create 3x3 system (not bipartite)
        rho = np.eye(3) / 3
        dm = DensityMatrix(matrix=rho, basis_labels=['|0>', '|1>', '|2>'], time=0.0)
        
        with pytest.raises(ValueError):
            self.analyzer.measure_quantum_discord(dm)
    
    def test_entanglement_measure_separable(self):
        """Test entanglement measure for separable state."""
        # Product state |0>⊗|0>
        rho = np.zeros((4, 4))
        rho[0, 0] = 1.0
        
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_4, time=0.0)
        entanglement = self.analyzer.calculate_entanglement_measure(dm)
        
        assert abs(entanglement) < 1e-10, f"Expected entanglement ~0, got {entanglement}"
    
    def test_entanglement_measure_bell_state(self):
        """Test entanglement measure for Bell state."""
        # Bell state |Φ+>
        coeffs = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        dm = self.create_pure_state_dm(coeffs, self.basis_4)
        
        entanglement = self.analyzer.calculate_entanglement_measure(dm)
        
        # Bell state should be maximally entangled
        assert entanglement > 0.8, f"Expected entanglement > 0.8, got {entanglement}"
    
    def test_purity_pure_state(self):
        """Test purity calculation for pure state."""
        # Pure state |0>
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=0.0)
        
        purity = self.analyzer.calculate_purity(dm)
        
        assert abs(purity - 1.0) < 1e-10, f"Expected purity = 1, got {purity}"
    
    def test_purity_maximally_mixed(self):
        """Test purity calculation for maximally mixed state."""
        # Maximally mixed state
        rho = np.eye(2) / 2
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=0.0)
        
        purity = self.analyzer.calculate_purity(dm)
        
        assert abs(purity - 0.5) < 1e-10, f"Expected purity = 0.5, got {purity}"
    
    def test_von_neumann_entropy_pure_state(self):
        """Test von Neumann entropy for pure state."""
        # Pure state should have zero entropy
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=0.0)
        
        entropy = self.analyzer.calculate_von_neumann_entropy(dm)
        
        assert abs(entropy) < 1e-10, f"Expected entropy = 0, got {entropy}"
    
    def test_von_neumann_entropy_maximally_mixed(self):
        """Test von Neumann entropy for maximally mixed state."""
        # Maximally mixed 2-level system should have entropy = 1
        rho = np.eye(2) / 2
        dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=0.0)
        
        entropy = self.analyzer.calculate_von_neumann_entropy(dm)
        
        assert abs(entropy - 1.0) < 1e-10, f"Expected entropy = 1, got {entropy}"
    
    def test_decoherence_statistics(self):
        """Test statistical analysis of decoherence processes."""
        # Create trajectory with known decoherence
        times = np.linspace(0, 2, 21)
        trajectory = []
        
        for t in times:
            # Exponential decoherence
            gamma = 1.0
            coherence_factor = np.exp(-gamma * t)
            
            rho = np.array([
                [0.5, 0.5 * coherence_factor],
                [0.5 * coherence_factor, 0.5]
            ], dtype=complex)
            
            dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=t)
            trajectory.append(dm)
        
        stats = self.analyzer.analyze_decoherence_statistics(trajectory)
        
        # Check that we get reasonable statistics
        assert 'purity_decay_rate' in stats
        assert 'entropy_growth_rate' in stats
        assert 'coherence_decay_rate' in stats
        assert stats['coherence_decay_rate'] > 0
        assert stats['final_coherence'] < stats['initial_coherence']
    
    def test_generate_coherence_metrics(self):
        """Test comprehensive coherence metrics generation."""
        # Create simple decoherence trajectory
        times = [0.0, 1.0, 2.0]
        trajectory = []
        
        for i, t in enumerate(times):
            coherence = 0.5 * np.exp(-0.5 * t)
            rho = np.array([
                [0.5, coherence],
                [coherence, 0.5]
            ], dtype=complex)
            
            dm = DensityMatrix(matrix=rho, basis_labels=self.basis_2, time=t)
            trajectory.append(dm)
        
        metrics = self.analyzer.generate_coherence_metrics(trajectory)
        
        # Check that we get a valid CoherenceMetrics object
        assert isinstance(metrics, CoherenceMetrics)
        assert metrics.coherence_lifetime > 0
        assert 0 <= metrics.purity <= 1
        assert metrics.von_neumann_entropy >= 0
    
    def test_generate_coherence_metrics_empty_trajectory(self):
        """Test coherence metrics with empty trajectory."""
        with pytest.raises(ValueError):
            self.analyzer.generate_coherence_metrics([])


class TestQuantumInformationHelpers:
    """Test helper methods for quantum information calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ResultsAnalyzer()
    
    def test_partial_trace_product_state(self):
        """Test partial trace for product state."""
        # Create product state |0>⊗|1>
        rho = np.zeros((4, 4))
        rho[1, 1] = 1.0  # |01><01|
        
        # Trace over subsystem B
        rho_A = self.analyzer._partial_trace(rho, 2, 2, 'B')
        expected_A = np.array([[1.0, 0.0], [0.0, 0.0]])  # |0><0|
        
        assert np.allclose(rho_A, expected_A), "Partial trace over B failed"
        
        # Trace over subsystem A
        rho_B = self.analyzer._partial_trace(rho, 2, 2, 'A')
        expected_B = np.array([[0.0, 0.0], [0.0, 1.0]])  # |1><1|
        
        assert np.allclose(rho_B, expected_B), "Partial trace over A failed"
    
    def test_partial_trace_entangled_state(self):
        """Test partial trace for entangled state."""
        # Bell state |Φ+> = (|00> + |11>)/√2
        rho = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5]
        ])
        
        # Both reduced states should be maximally mixed
        rho_A = self.analyzer._partial_trace(rho, 2, 2, 'B')
        rho_B = self.analyzer._partial_trace(rho, 2, 2, 'A')
        
        expected_mixed = np.eye(2) / 2
        
        assert np.allclose(rho_A, expected_mixed), "Reduced state A not maximally mixed"
        assert np.allclose(rho_B, expected_mixed), "Reduced state B not maximally mixed"
    
    def test_von_neumann_entropy_calculation(self):
        """Test von Neumann entropy calculation."""
        # Test known cases
        
        # Pure state: entropy = 0
        rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]])
        entropy_pure = self.analyzer._von_neumann_entropy(rho_pure)
        assert abs(entropy_pure) < 1e-10
        
        # Maximally mixed: entropy = log2(d)
        rho_mixed = np.eye(2) / 2
        entropy_mixed = self.analyzer._von_neumann_entropy(rho_mixed)
        assert abs(entropy_mixed - 1.0) < 1e-10
        
        # 3-level maximally mixed: entropy = log2(3)
        rho_3 = np.eye(3) / 3
        entropy_3 = self.analyzer._von_neumann_entropy(rho_3)
        expected_3 = np.log2(3)
        assert abs(entropy_3 - expected_3) < 1e-10
    
    def test_concurrence_calculation(self):
        """Test concurrence calculation for two-qubit states."""
        # Separable state: concurrence = 0
        rho_sep = np.zeros((4, 4))
        rho_sep[0, 0] = 1.0  # |00><00|
        concurrence_sep = self.analyzer._calculate_concurrence(rho_sep)
        assert abs(concurrence_sep) < 1e-10
        
        # Bell state: concurrence = 1
        rho_bell = np.array([
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.5]
        ])
        concurrence_bell = self.analyzer._calculate_concurrence(rho_bell)
        assert abs(concurrence_bell - 1.0) < 1e-10
    
    def test_fit_decay_rate(self):
        """Test exponential decay rate fitting."""
        # Create exponential decay data
        times = np.linspace(0, 5, 11)
        true_rate = 0.5
        values = np.exp(-true_rate * times)
        
        fitted_rate = self.analyzer._fit_decay_rate(times, values)
        
        assert abs(fitted_rate - true_rate) < 0.1, f"Expected rate {true_rate}, got {fitted_rate}"
    
    def test_fit_growth_rate(self):
        """Test linear growth rate fitting."""
        # Create linear growth data
        times = np.linspace(0, 5, 11)
        true_rate = 0.3
        values = true_rate * times + 1.0
        
        fitted_rate = self.analyzer._fit_growth_rate(times, values)
        
        assert abs(fitted_rate - true_rate) < 0.1, f"Expected rate {true_rate}, got {fitted_rate}"


if __name__ == "__main__":
    pytest.main([__file__])