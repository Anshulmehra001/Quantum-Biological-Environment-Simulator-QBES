"""
Results analysis and validation tools.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import linalg, optimize, stats
try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

from .core.interfaces import AnalysisInterface
from .core.data_models import (
    DensityMatrix, SimulationResults, ValidationResult, StatisticalSummary,
    CoherenceMetrics
)

__all__ = ['CoherenceAnalyzer', 'StatisticalAnalyzer', 'ResultsAnalyzer']


class CoherenceAnalyzer:
    """Analyzer for quantum coherence measures."""
    
    def __init__(self):
        self.analyzer = ResultsAnalyzer()
    
    def analyze_coherence_evolution(self, state_trajectory):
        """Analyze coherence evolution over time."""
        return self.analyzer.calculate_coherence_lifetime(state_trajectory)
    
    def calculate_coherence_lifetime(self, state_trajectory):
        """Calculate quantum coherence lifetime from state evolution."""
        return self.analyzer.calculate_coherence_lifetime(state_trajectory)
    
    def analyze_decoherence(self, state_trajectory):
        """Analyze decoherence processes in the quantum system."""
        if not state_trajectory:
            return {"decoherence_rate": 0.0, "coherence_time": float('inf')}
        
        # Calculate decoherence rate from coherence decay
        coherence_lifetime = self.calculate_coherence_lifetime(state_trajectory)
        decoherence_rate = 1.0 / coherence_lifetime if coherence_lifetime > 0 else 0.0
        
        return {
            "decoherence_rate": decoherence_rate,
            "coherence_time": coherence_lifetime,
            "decay_type": "exponential"  # Assume exponential decay
        }
    
    def calculate_coherence_measures(self, state):
        """Calculate various coherence measures."""
        return self.analyzer.generate_coherence_metrics([state])


class StatisticalAnalyzer:
    """Analyzer for statistical analysis of simulation results."""
    
    def __init__(self):
        self.analyzer = ResultsAnalyzer()
    
    def calculate_statistics(self, data_series):
        """Calculate statistical measures for a data series.
        
        Args:
            data_series: List or array of numerical data
            
        Returns:
            Dictionary containing statistical measures
        """
        if not data_series:
            return {"mean": 0.0, "std": 0.0, "variance": 0.0, "count": 0}
        
        data_array = np.array(data_series)
        
        return {
            "mean": np.mean(data_array),
            "std": np.std(data_array),
            "variance": np.var(data_array),
            "median": np.median(data_array),
            "min": np.min(data_array),
            "max": np.max(data_array),
            "count": len(data_array)
        }
    
    def generate_confidence_intervals(self, data_series, confidence_level=0.95):
        """Generate confidence intervals for statistical measures.
        
        Args:
            data_series: List or array of numerical data
            confidence_level: Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary containing confidence intervals
        """
        if not data_series:
            return {"mean_ci": (0.0, 0.0), "confidence_level": confidence_level}
        
        data_array = np.array(data_series)
        n = len(data_array)
        
        if n < 2:
            mean_val = np.mean(data_array)
            return {
                "mean_ci": (mean_val, mean_val),
                "confidence_level": confidence_level,
                "sample_size": n
            }
        
        # Calculate confidence interval for the mean
        mean_val = np.mean(data_array)
        std_err = stats.sem(data_array)  # Standard error of the mean
        
        # Use t-distribution for small samples
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin_error = t_critical * std_err
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        
        return {
            "mean_ci": (ci_lower, ci_upper),
            "confidence_level": confidence_level,
            "sample_size": n,
            "standard_error": std_err,
            "margin_of_error": margin_error
        }


class ResultsAnalyzer(AnalysisInterface):
    """
    Provides comprehensive analysis and validation of simulation results.
    
    This class implements statistical analysis, physical validation, and
    uncertainty quantification for quantum biological simulations.
    """
    
    def __init__(self):
        """Initialize the results analyzer."""
        self.analysis_cache = {}
    
    def calculate_coherence_lifetime(self, state_trajectory: List[DensityMatrix]) -> float:
        """
        Calculate quantum coherence lifetime from state evolution.
        
        Uses the decay of off-diagonal elements in the density matrix to determine
        the characteristic decoherence time scale.
        
        Args:
            state_trajectory: List of density matrices over time
            
        Returns:
            Coherence lifetime in simulation time units
        """
        if len(state_trajectory) < 2:
            raise ValueError("Need at least 2 time points to calculate coherence lifetime")
        
        # Extract time points and coherence measures
        times = np.array([state.time for state in state_trajectory])
        coherences = []
        
        for state in state_trajectory:
            # Calculate coherence as sum of squared off-diagonal elements
            rho = state.matrix
            n = rho.shape[0]
            off_diag_sum = 0.0
            
            for i in range(n):
                for j in range(i+1, n):
                    off_diag_sum += abs(rho[i, j])**2
            
            # For 2x2 case, multiply by 2 to account for both off-diagonal elements
            if n == 2:
                off_diag_sum *= 2
            
            coherences.append(off_diag_sum)
        
        coherences = np.array(coherences)
        
        # Fit exponential decay: C(t) = C0 * exp(-t/tau)
        if coherences[0] == 0:
            return 0.0
        
        # Normalize by initial coherence
        normalized_coherence = coherences / coherences[0]
        
        # Fit exponential decay
        try:
            def exp_decay(t, tau):
                return np.exp(-t / tau)
            
            # Only fit if we have decay
            if normalized_coherence[-1] < normalized_coherence[0]:
                # Use relative time (subtract initial time)
                rel_times = times - times[0]
                popt, _ = optimize.curve_fit(exp_decay, rel_times, normalized_coherence, 
                                           bounds=(0, np.inf), maxfev=1000)
                return popt[0]  # Return tau (lifetime)
            else:
                # No significant decay observed
                return np.inf
                
        except (RuntimeError, ValueError):
            # Fallback: calculate 1/e time manually
            target = 1.0 / np.e
            idx = np.where(normalized_coherence <= target)[0]
            if len(idx) > 0:
                return times[idx[0]] - times[0]
            else:
                return np.inf
    
    def measure_quantum_discord(self, bipartite_state: DensityMatrix) -> float:
        """
        Calculate quantum discord for bipartite quantum state.
        
        Quantum discord measures quantum correlations beyond entanglement.
        For a bipartite state ρ_AB, discord is defined as:
        D(A|B) = I(A:B) - J(A|B)
        where I is mutual information and J is classical correlation.
        
        Args:
            bipartite_state: Density matrix of bipartite quantum system
            
        Returns:
            Quantum discord value
        """
        rho = bipartite_state.matrix
        n = rho.shape[0]
        
        # Assume equal subsystem dimensions for simplicity
        if not np.log2(n).is_integer():
            raise ValueError("System dimension must be power of 2 for bipartite analysis")
        
        dim_A = dim_B = int(np.sqrt(n))
        
        if dim_A * dim_B != n:
            raise ValueError("Cannot partition system into equal subsystems")
        
        # Calculate reduced density matrices
        rho_A = self._partial_trace(rho, dim_A, dim_B, trace_over='B')
        rho_B = self._partial_trace(rho, dim_A, dim_B, trace_over='A')
        
        # Calculate von Neumann entropies
        S_A = self._von_neumann_entropy(rho_A)
        S_B = self._von_neumann_entropy(rho_B)
        S_AB = self._von_neumann_entropy(rho)
        
        # Mutual information
        mutual_info = S_A + S_B - S_AB
        
        # Classical correlation (maximum over all measurements on B)
        # Simplified calculation using computational basis measurements
        classical_corr = self._calculate_classical_correlation(rho, dim_A, dim_B)
        
        # Quantum discord
        discord = mutual_info - classical_corr
        
        return max(0.0, discord)  # Discord should be non-negative
    
    def validate_energy_conservation(self, energy_trajectory: List[float]) -> ValidationResult:
        """
        Validate energy conservation throughout simulation.
        
        Checks that the total energy remains approximately constant within
        acceptable numerical tolerances.
        
        Args:
            energy_trajectory: List of energy values over time
            
        Returns:
            ValidationResult with conservation status and diagnostics
        """
        result = ValidationResult(is_valid=True)
        
        if len(energy_trajectory) < 2:
            result.add_error("Need at least 2 energy points for conservation check")
            return result
        
        energies = np.array(energy_trajectory)
        
        # Calculate energy statistics
        initial_energy = energies[0]
        final_energy = energies[-1]
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        max_deviation = np.max(np.abs(energies - initial_energy))
        
        # Define tolerance based on initial energy magnitude
        if abs(initial_energy) > 1e-12:
            relative_tolerance = 1e-6  # 0.0001% relative error
            absolute_tolerance = abs(initial_energy) * relative_tolerance
        else:
            absolute_tolerance = 1e-12  # Absolute tolerance for near-zero energies
        
        # Check conservation criteria
        if max_deviation > absolute_tolerance:
            result.add_error(f"Energy not conserved: max deviation {max_deviation:.2e} "
                           f"exceeds tolerance {absolute_tolerance:.2e}")
        
        # Check for systematic drift
        energy_drift = abs(final_energy - initial_energy)
        if energy_drift > absolute_tolerance:
            result.add_warning(f"Energy drift detected: {energy_drift:.2e}")
        
        # Check for excessive fluctuations
        if std_energy > absolute_tolerance:
            result.add_warning(f"Large energy fluctuations: std = {std_energy:.2e}")
        
        return result
    
    def validate_probability_conservation(self, state_trajectory: List[DensityMatrix]) -> ValidationResult:
        """
        Validate probability conservation (trace preservation).
        
        Checks that Tr(ρ) = 1 for all density matrices in the trajectory
        and that all eigenvalues remain non-negative.
        
        Args:
            state_trajectory: List of density matrices over time
            
        Returns:
            ValidationResult with probability conservation status
        """
        result = ValidationResult(is_valid=True)
        
        if len(state_trajectory) == 0:
            result.add_error("Empty state trajectory")
            return result
        
        trace_tolerance = 1e-10
        eigenvalue_tolerance = -1e-10  # Allow small negative eigenvalues due to numerics
        
        for i, state in enumerate(state_trajectory):
            rho = state.matrix
            
            # Check trace normalization
            trace = np.trace(rho)
            if abs(trace - 1.0) > trace_tolerance:
                result.add_error(f"State {i} not normalized: Tr(ρ) = {trace:.2e}")
            
            # Check Hermiticity
            if not np.allclose(rho, rho.conj().T, atol=1e-12):
                result.add_error(f"State {i} not Hermitian")
            
            # Check positive semidefinite property
            eigenvals = np.linalg.eigvals(rho)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < eigenvalue_tolerance:
                result.add_error(f"State {i} has negative eigenvalue: {min_eigenval:.2e}")
            
            # Check for numerical issues
            if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
                result.add_error(f"State {i} contains NaN or Inf values")
        
        return result
    
    def generate_statistical_summary(self, results: SimulationResults) -> StatisticalSummary:
        """
        Generate comprehensive statistical analysis of results.
        
        Args:
            results: Complete simulation results
            
        Returns:
            StatisticalSummary with means, standard deviations, and confidence intervals
        """
        mean_values = {}
        std_deviations = {}
        confidence_intervals = {}
        
        # Analyze energy trajectory
        if len(results.energy_trajectory) > 0:
            energies = np.array(results.energy_trajectory)
            mean_values['energy'] = np.mean(energies)
            std_deviations['energy'] = np.std(energies, ddof=1) if len(energies) > 1 else 0.0
            confidence_intervals['energy'] = self._calculate_confidence_interval(energies)
        
        # Analyze coherence measures
        for measure_name, values in results.coherence_measures.items():
            if len(values) > 0:
                values_array = np.array(values)
                mean_values[measure_name] = np.mean(values_array)
                std_deviations[measure_name] = np.std(values_array, ddof=1) if len(values) > 1 else 0.0
                confidence_intervals[measure_name] = self._calculate_confidence_interval(values_array)
        
        # Analyze decoherence rates
        for rate_name, rate_value in results.decoherence_rates.items():
            mean_values[f'{rate_name}_rate'] = rate_value
            std_deviations[f'{rate_name}_rate'] = 0.0  # Single value
            confidence_intervals[f'{rate_name}_rate'] = (rate_value, rate_value)
        
        # Analyze state trajectory properties
        if len(results.state_trajectory) > 0:
            purities = [self.calculate_purity(state) for state in results.state_trajectory]
            entropies = [self.calculate_von_neumann_entropy(state) for state in results.state_trajectory]
            
            mean_values['purity'] = np.mean(purities)
            std_deviations['purity'] = np.std(purities, ddof=1) if len(purities) > 1 else 0.0
            confidence_intervals['purity'] = self._calculate_confidence_interval(np.array(purities))
            
            mean_values['entropy'] = np.mean(entropies)
            std_deviations['entropy'] = np.std(entropies, ddof=1) if len(entropies) > 1 else 0.0
            confidence_intervals['entropy'] = self._calculate_confidence_interval(np.array(entropies))
        
        sample_size = max(len(results.state_trajectory), len(results.energy_trajectory), 1)
        
        return StatisticalSummary(
            mean_values=mean_values,
            std_deviations=std_deviations,
            confidence_intervals=confidence_intervals,
            sample_size=sample_size
        )
    
    def detect_outliers(self, data: np.ndarray, method: str = "iqr") -> List[int]:
        """
        Detect outliers in simulation data using various statistical methods.
        
        Args:
            data: 1D array of data points
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            List of indices of detected outliers
        """
        if len(data) < 3:
            return []  # Need at least 3 points for meaningful outlier detection
        
        outlier_indices = []
        
        if method == "iqr":
            # Interquartile Range method
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
            
        elif method == "zscore":
            # Z-score method
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            
            if std > 0:
                z_scores = np.abs((data - mean) / std)
                outlier_indices = np.where(z_scores > 3.0)[0].tolist()
            
        elif method == "modified_zscore":
            # Modified Z-score using median absolute deviation
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            
            if mad > 0:
                modified_z_scores = 0.6745 * (data - median) / mad
                outlier_indices = np.where(np.abs(modified_z_scores) > 3.5)[0].tolist()
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outlier_indices
    
    def calculate_uncertainty_estimates(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate uncertainty estimates for measured quantities.
        
        Args:
            data: Array of measurement values
            
        Returns:
            Dictionary with various uncertainty estimates
        """
        if len(data) == 0:
            return {}
        
        if len(data) == 1:
            return {
                'mean': data[0],
                'std_error': 0.0,
                'confidence_interval_95': (data[0], data[0]),
                'relative_uncertainty': 0.0
            }
        
        # Basic statistics
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)
        std_error = std_dev / np.sqrt(len(data))
        
        # Confidence intervals (95% by default)
        confidence_level = 0.95
        alpha = 1 - confidence_level
        
        # Use t-distribution for small samples
        if len(data) < 30:
            from scipy.stats import t
            t_value = t.ppf(1 - alpha/2, len(data) - 1)
        else:
            # Use normal distribution for large samples
            t_value = 1.96  # 95% confidence
        
        margin_of_error = t_value * std_error
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        # Relative uncertainty
        relative_uncertainty = std_error / abs(mean) if abs(mean) > 1e-12 else np.inf
        
        # Bootstrap confidence interval (alternative method)
        bootstrap_ci = self._bootstrap_confidence_interval(data, confidence_level)
        
        return {
            'mean': mean,
            'std_dev': std_dev,
            'std_error': std_error,
            'confidence_interval_95': (ci_lower, ci_upper),
            'bootstrap_ci_95': bootstrap_ci,
            'relative_uncertainty': relative_uncertainty,
            'sample_size': len(data)
        }
    
    def calculate_entanglement_measure(self, bipartite_state: DensityMatrix) -> float:
        """
        Calculate entanglement measure (concurrence) for bipartite state.
        
        Args:
            bipartite_state: Density matrix of bipartite quantum system
            
        Returns:
            Entanglement measure (0 = separable, 1 = maximally entangled)
        """
        rho = bipartite_state.matrix
        n = rho.shape[0]
        
        # For 2x2 systems (two qubits), calculate concurrence
        if n == 4:
            return self._calculate_concurrence(rho)
        else:
            # For larger systems, use negativity as entanglement measure
            return self._calculate_negativity(rho)
    
    def calculate_purity(self, state: DensityMatrix) -> float:
        """
        Calculate purity of quantum state: Tr(ρ²).
        
        Args:
            state: Density matrix
            
        Returns:
            Purity (1 = pure state, 1/d = maximally mixed)
        """
        rho = state.matrix
        return np.real(np.trace(rho @ rho))
    
    def calculate_von_neumann_entropy(self, state: DensityMatrix) -> float:
        """
        Calculate von Neumann entropy: S = -Tr(ρ log ρ).
        
        Args:
            state: Density matrix
            
        Returns:
            von Neumann entropy
        """
        return self._von_neumann_entropy(state.matrix)
    
    def analyze_decoherence_statistics(self, state_trajectory: List[DensityMatrix]) -> Dict[str, float]:
        """
        Perform statistical analysis of decoherence processes.
        
        Args:
            state_trajectory: List of density matrices over time
            
        Returns:
            Dictionary with decoherence statistics
        """
        if len(state_trajectory) < 3:
            raise ValueError("Need at least 3 time points for statistical analysis")
        
        # Calculate time-dependent quantities
        times = np.array([state.time for state in state_trajectory])
        purities = [self.calculate_purity(state) for state in state_trajectory]
        entropies = [self.calculate_von_neumann_entropy(state) for state in state_trajectory]
        
        # Calculate coherence decay rates
        coherences = []
        for state in state_trajectory:
            rho = state.matrix
            n = rho.shape[0]
            coherence = sum(abs(rho[i, j])**2 for i in range(n) for j in range(i+1, n))
            coherences.append(coherence)
        
        # Fit exponential decay rates
        purity_rate = self._fit_decay_rate(times, purities)
        entropy_rate = self._fit_growth_rate(times, entropies)
        coherence_rate = self._fit_decay_rate(times, coherences)
        
        return {
            'purity_decay_rate': purity_rate,
            'entropy_growth_rate': entropy_rate,
            'coherence_decay_rate': coherence_rate,
            'final_purity': purities[-1],
            'final_entropy': entropies[-1],
            'initial_coherence': coherences[0],
            'final_coherence': coherences[-1]
        }
    
    def generate_coherence_metrics(self, state_trajectory: List[DensityMatrix]) -> CoherenceMetrics:
        """
        Generate comprehensive coherence metrics for a state trajectory.
        
        Args:
            state_trajectory: List of density matrices over time
            
        Returns:
            CoherenceMetrics object with all calculated measures
        """
        if len(state_trajectory) == 0:
            raise ValueError("Empty state trajectory")
        
        # Calculate coherence lifetime
        coherence_lifetime = self.calculate_coherence_lifetime(state_trajectory)
        
        # Use final state for instantaneous measures
        final_state = state_trajectory[-1]
        
        # Calculate quantum discord (assume bipartite if possible)
        try:
            quantum_discord = self.measure_quantum_discord(final_state)
        except ValueError:
            quantum_discord = 0.0  # Not bipartite or other issue
        
        # Calculate entanglement measure
        try:
            entanglement_measure = self.calculate_entanglement_measure(final_state)
        except ValueError:
            entanglement_measure = 0.0
        
        # Calculate purity and entropy
        purity = self.calculate_purity(final_state)
        von_neumann_entropy = self.calculate_von_neumann_entropy(final_state)
        
        return CoherenceMetrics(
            coherence_lifetime=coherence_lifetime,
            quantum_discord=quantum_discord,
            entanglement_measure=entanglement_measure,
            purity=purity,
            von_neumann_entropy=von_neumann_entropy
        )
    
    # Helper methods for quantum information calculations
    
    def _partial_trace(self, rho: np.ndarray, dim_A: int, dim_B: int, trace_over: str) -> np.ndarray:
        """Calculate partial trace of bipartite density matrix."""
        if trace_over == 'B':
            # Trace over subsystem B
            rho_A = np.zeros((dim_A, dim_A), dtype=complex)
            for i in range(dim_A):
                for j in range(dim_A):
                    for k in range(dim_B):
                        rho_A[i, j] += rho[i * dim_B + k, j * dim_B + k]
            return rho_A
        elif trace_over == 'A':
            # Trace over subsystem A
            rho_B = np.zeros((dim_B, dim_B), dtype=complex)
            for i in range(dim_B):
                for j in range(dim_B):
                    for k in range(dim_A):
                        rho_B[i, j] += rho[k * dim_B + i, k * dim_B + j]
            return rho_B
        else:
            raise ValueError("trace_over must be 'A' or 'B'")
    
    def _von_neumann_entropy(self, rho: np.ndarray) -> float:
        """Calculate von Neumann entropy of density matrix."""
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        return -np.sum(eigenvals * np.log2(eigenvals))
    
    def _calculate_classical_correlation(self, rho: np.ndarray, dim_A: int, dim_B: int) -> float:
        """Calculate classical correlation for quantum discord."""
        # Simplified calculation using computational basis measurements on B
        max_classical = 0.0
        
        # Try different measurement bases (just computational basis for now)
        for k in range(dim_B):
            # Measurement projector |k><k|
            proj = np.zeros((dim_B, dim_B))
            proj[k, k] = 1.0
            
            # Post-measurement state
            prob_k = 0.0
            conditional_entropy = 0.0
            
            for i in range(dim_A):
                for j in range(dim_A):
                    for l in range(dim_B):
                        if l == k:
                            prob_k += np.real(rho[i * dim_B + l, j * dim_B + l])
            
            if prob_k > 1e-12:
                # Calculate conditional entropy (simplified)
                rho_A_given_k = self._partial_trace(rho, dim_A, dim_B, 'B')
                conditional_entropy += prob_k * self._von_neumann_entropy(rho_A_given_k / prob_k)
        
        # Classical correlation approximation
        rho_A = self._partial_trace(rho, dim_A, dim_B, 'B')
        S_A = self._von_neumann_entropy(rho_A)
        classical_corr = S_A - conditional_entropy
        
        return max(0.0, classical_corr)
    
    def _calculate_concurrence(self, rho: np.ndarray) -> float:
        """Calculate concurrence for two-qubit state."""
        # Pauli-Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # Spin-flipped density matrix
        rho_tilde = np.kron(sigma_y, sigma_y) @ rho.conj() @ np.kron(sigma_y, sigma_y)
        
        # Calculate eigenvalues of rho * rho_tilde
        eigenvals = np.linalg.eigvals(rho @ rho_tilde)
        eigenvals = np.sqrt(np.maximum(0, np.real(eigenvals)))
        eigenvals = np.sort(eigenvals)[::-1]
        
        concurrence = max(0, eigenvals[0] - eigenvals[1] - eigenvals[2] - eigenvals[3])
        return concurrence
    
    def _calculate_negativity(self, rho: np.ndarray) -> float:
        """Calculate negativity as entanglement measure for larger systems."""
        n = rho.shape[0]
        dim_A = dim_B = int(np.sqrt(n))
        
        # Partial transpose with respect to subsystem B
        rho_pt = np.zeros_like(rho)
        for i in range(dim_A):
            for j in range(dim_A):
                for k in range(dim_B):
                    for l in range(dim_B):
                        rho_pt[i * dim_B + k, j * dim_B + l] = rho[i * dim_B + l, j * dim_B + k]
        
        # Calculate negativity
        eigenvals = np.linalg.eigvals(rho_pt)
        negativity = (np.sum(np.abs(eigenvals)) - 1) / 2
        return max(0.0, negativity)
    
    def validate_against_theoretical_predictions(self, 
                                               measured_values: Dict[str, float],
                                               theoretical_values: Dict[str, float],
                                               tolerances: Optional[Dict[str, float]] = None) -> ValidationResult:
        """
        Compare simulation results against theoretical predictions.
        
        Args:
            measured_values: Dictionary of measured quantities
            theoretical_values: Dictionary of expected theoretical values
            tolerances: Optional dictionary of acceptable tolerances for each quantity
            
        Returns:
            ValidationResult with comparison status
        """
        result = ValidationResult(is_valid=True)
        
        if tolerances is None:
            tolerances = {}
        
        # Default tolerance is 5% relative error
        default_tolerance = 0.05
        
        for quantity in measured_values:
            if quantity not in theoretical_values:
                result.add_warning(f"No theoretical value available for {quantity}")
                continue
            
            measured = measured_values[quantity]
            theoretical = theoretical_values[quantity]
            tolerance = tolerances.get(quantity, default_tolerance)
            
            # Calculate relative error
            if abs(theoretical) > 1e-12:
                relative_error = abs(measured - theoretical) / abs(theoretical)
                if relative_error > tolerance:
                    result.add_error(f"{quantity}: measured {measured:.4f}, "
                                   f"theoretical {theoretical:.4f}, "
                                   f"relative error {relative_error:.2%} > {tolerance:.2%}")
            else:
                # Use absolute error for near-zero theoretical values
                absolute_error = abs(measured - theoretical)
                if absolute_error > tolerance:
                    result.add_error(f"{quantity}: measured {measured:.4f}, "
                                   f"theoretical {theoretical:.4f}, "
                                   f"absolute error {absolute_error:.2e} > {tolerance:.2e}")
        
        return result
    
    def validate_physical_bounds(self, results: SimulationResults) -> ValidationResult:
        """
        Validate that all physical quantities are within expected bounds.
        
        Args:
            results: Complete simulation results
            
        Returns:
            ValidationResult with bounds validation status
        """
        result = ValidationResult(is_valid=True)
        
        # Validate coherence measures
        if 'coherence_lifetime' in results.coherence_measures:
            lifetimes = results.coherence_measures['coherence_lifetime']
            for i, lifetime in enumerate(lifetimes):
                if lifetime < 0:
                    result.add_error(f"Negative coherence lifetime at step {i}: {lifetime}")
        
        # Validate energy trajectory
        if len(results.energy_trajectory) > 0:
            energies = np.array(results.energy_trajectory)
            if np.any(np.isnan(energies)) or np.any(np.isinf(energies)):
                result.add_error("Energy trajectory contains NaN or Inf values")
        
        # Validate decoherence rates
        for rate_name, rate_value in results.decoherence_rates.items():
            if rate_value < 0:
                result.add_error(f"Negative decoherence rate {rate_name}: {rate_value}")
        
        # Validate state trajectory physical properties
        prob_result = self.validate_probability_conservation(results.state_trajectory)
        if not prob_result.is_valid:
            result.errors.extend(prob_result.errors)
            result.warnings.extend(prob_result.warnings)
            result.is_valid = False
        
        return result
    
    def perform_comprehensive_validation(self, results: SimulationResults,
                                       theoretical_benchmarks: Optional[Dict[str, float]] = None) -> ValidationResult:
        """
        Perform comprehensive validation of simulation results.
        
        Args:
            results: Complete simulation results
            theoretical_benchmarks: Optional theoretical values for comparison
            
        Returns:
            ValidationResult with comprehensive validation status
        """
        result = ValidationResult(is_valid=True)
        
        # 1. Validate energy conservation
        energy_result = self.validate_energy_conservation(results.energy_trajectory)
        if not energy_result.is_valid:
            result.errors.extend([f"Energy: {err}" for err in energy_result.errors])
            result.warnings.extend([f"Energy: {warn}" for warn in energy_result.warnings])
            result.is_valid = False
        
        # 2. Validate probability conservation
        prob_result = self.validate_probability_conservation(results.state_trajectory)
        if not prob_result.is_valid:
            result.errors.extend([f"Probability: {err}" for err in prob_result.errors])
            result.warnings.extend([f"Probability: {warn}" for warn in prob_result.warnings])
            result.is_valid = False
        
        # 3. Validate physical bounds
        bounds_result = self.validate_physical_bounds(results)
        if not bounds_result.is_valid:
            result.errors.extend([f"Bounds: {err}" for err in bounds_result.errors])
            result.warnings.extend([f"Bounds: {warn}" for warn in bounds_result.warnings])
            result.is_valid = False
        
        # 4. Compare against theoretical benchmarks if provided
        if theoretical_benchmarks:
            # Extract key quantities from results for comparison
            measured_quantities = {}
            
            if len(results.state_trajectory) > 0:
                final_state = results.state_trajectory[-1]
                measured_quantities['final_purity'] = self.calculate_purity(final_state)
                measured_quantities['final_entropy'] = self.calculate_von_neumann_entropy(final_state)
            
            if 'coherence_lifetime' in results.coherence_measures:
                lifetimes = results.coherence_measures['coherence_lifetime']
                if len(lifetimes) > 0:
                    measured_quantities['coherence_lifetime'] = np.mean(lifetimes)
            
            theory_result = self.validate_against_theoretical_predictions(
                measured_quantities, theoretical_benchmarks)
            if not theory_result.is_valid:
                result.errors.extend([f"Theory: {err}" for err in theory_result.errors])
                result.warnings.extend([f"Theory: {warn}" for warn in theory_result.warnings])
                result.is_valid = False
        
        # 5. Check for numerical stability issues
        if len(results.state_trajectory) > 1:
            # Check for sudden jumps in purity
            purities = [self.calculate_purity(state) for state in results.state_trajectory]
            purity_diffs = np.diff(purities)
            max_purity_jump = np.max(np.abs(purity_diffs))
            
            if max_purity_jump > 0.1:  # Arbitrary threshold for sudden changes
                result.add_warning(f"Large purity jump detected: {max_purity_jump:.3f}")
        
        return result
    
    def assess_data_quality(self, results: SimulationResults) -> Dict[str, any]:
        """
        Assess the quality of simulation data.
        
        Args:
            results: Complete simulation results
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_metrics = {}
        
        # Check for missing or invalid data
        quality_metrics['has_state_trajectory'] = len(results.state_trajectory) > 0
        quality_metrics['has_energy_trajectory'] = len(results.energy_trajectory) > 0
        quality_metrics['trajectory_length'] = len(results.state_trajectory)
        
        # Check for NaN or infinite values
        if len(results.energy_trajectory) > 0:
            energies = np.array(results.energy_trajectory)
            quality_metrics['energy_has_nan'] = np.any(np.isnan(energies))
            quality_metrics['energy_has_inf'] = np.any(np.isinf(energies))
        
        # Detect outliers in key quantities
        if len(results.state_trajectory) > 2:
            purities = np.array([self.calculate_purity(state) for state in results.state_trajectory])
            outlier_indices = self.detect_outliers(purities, method='iqr')
            quality_metrics['purity_outliers'] = len(outlier_indices)
            quality_metrics['purity_outlier_fraction'] = len(outlier_indices) / len(purities)
        
        if len(results.energy_trajectory) > 2:
            energies = np.array(results.energy_trajectory)
            outlier_indices = self.detect_outliers(energies, method='iqr')
            quality_metrics['energy_outliers'] = len(outlier_indices)
            quality_metrics['energy_outlier_fraction'] = len(outlier_indices) / len(energies)
        
        # Check temporal consistency
        if len(results.state_trajectory) > 1:
            times = [state.time for state in results.state_trajectory]
            time_diffs = np.diff(times)
            quality_metrics['uniform_time_steps'] = np.allclose(time_diffs, time_diffs[0], rtol=1e-6)
            quality_metrics['time_step_variation'] = np.std(time_diffs) / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
        
        # Overall quality score (0-1, higher is better)
        quality_score = 1.0
        
        if quality_metrics.get('energy_has_nan', False) or quality_metrics.get('energy_has_inf', False):
            quality_score -= 0.5
        
        if quality_metrics.get('purity_outlier_fraction', 0) > 0.1:
            quality_score -= 0.2
        
        if quality_metrics.get('energy_outlier_fraction', 0) > 0.1:
            quality_score -= 0.2
        
        if not quality_metrics.get('uniform_time_steps', True):
            quality_score -= 0.1
        
        quality_metrics['overall_quality_score'] = max(0.0, quality_score)
        
        return quality_metrics
    
    def perform_trend_analysis(self, time_series: np.ndarray, times: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Perform trend analysis on time series data.
        
        Args:
            time_series: Array of values over time
            times: Optional array of time points (default: 0, 1, 2, ...)
            
        Returns:
            Dictionary with trend analysis results
        """
        if len(time_series) < 3:
            return {'trend_slope': 0.0, 'trend_significance': 0.0}
        
        if times is None:
            times = np.arange(len(time_series))
        
        # Linear trend analysis
        coeffs = np.polyfit(times, time_series, 1)
        trend_slope = coeffs[0]
        
        # Calculate R-squared for trend significance
        y_pred = np.polyval(coeffs, times)
        ss_res = np.sum((time_series - y_pred) ** 2)
        ss_tot = np.sum((time_series - np.mean(time_series)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Detect change points using simple method
        change_points = []
        if len(time_series) > 10:
            # Look for significant changes in local slope
            window_size = max(3, len(time_series) // 5)
            for i in range(window_size, len(time_series) - window_size):
                left_slope = np.polyfit(times[i-window_size:i], time_series[i-window_size:i], 1)[0]
                right_slope = np.polyfit(times[i:i+window_size], time_series[i:i+window_size], 1)[0]
                
                if abs(left_slope - right_slope) > 2 * abs(trend_slope):
                    change_points.append(i)
        
        return {
            'trend_slope': trend_slope,
            'trend_r_squared': r_squared,
            'trend_significance': r_squared,  # Simple measure
            'change_points': change_points,
            'is_stationary': abs(trend_slope) < 0.1 * np.std(time_series) / np.sqrt(len(time_series))
        }
    
    # Helper methods for statistical calculations
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for data."""
        if len(data) <= 1:
            mean_val = np.mean(data) if len(data) == 1 else 0.0
            return (mean_val, mean_val)
        
        mean = np.mean(data)
        std_error = np.std(data, ddof=1) / np.sqrt(len(data))
        
        alpha = 1 - confidence_level
        
        # Use t-distribution for small samples
        if len(data) < 30:
            from scipy.stats import t
            t_value = t.ppf(1 - alpha/2, len(data) - 1)
        else:
            t_value = 1.96  # 95% confidence for normal distribution
        
        margin_of_error = t_value * std_error
        return (mean - margin_of_error, mean + margin_of_error)
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if len(data) <= 1:
            mean_val = np.mean(data) if len(data) == 1 else 0.0
            return (mean_val, mean_val)
        
        # Generate bootstrap samples
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _fit_decay_rate(self, times: np.ndarray, values: np.ndarray) -> float:
        """Fit exponential decay rate to time series data."""
        if len(times) < 3 or values[0] <= 0:
            return 0.0
        
        try:
            # Fit y = y0 * exp(-rate * t)
            log_values = np.log(np.maximum(values, 1e-12))
            coeffs = np.polyfit(times, log_values, 1)
            return -coeffs[0]  # Return positive decay rate
        except (ValueError, RuntimeWarning):
            return 0.0
    
    def _fit_growth_rate(self, times: np.ndarray, values: np.ndarray) -> float:
        """Fit exponential growth rate to time series data."""
        if len(times) < 3:
            return 0.0
        
        try:
            # Linear fit to find growth rate
            coeffs = np.polyfit(times, values, 1)
            return coeffs[0]  # Return growth rate
        except ValueError:
            return 0.0