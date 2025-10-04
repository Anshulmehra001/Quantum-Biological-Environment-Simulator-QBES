"""
Error handling and recovery utilities.
"""

import logging
import traceback
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.interfaces import ErrorHandlerInterface
from ..core.data_models import SimulationResults, ValidationResult, DensityMatrix


class ErrorHandler(ErrorHandlerInterface):
    """
    Handles errors and provides recovery strategies for QBES simulations.
    
    This class implements comprehensive error detection, logging, and recovery
    mechanisms for quantum biological simulations, including numerical stability
    monitoring, resource management, and physical validation.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_log = []
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.logger = logging.getLogger(__name__)
        
        # Numerical stability thresholds
        self.stability_thresholds = {
            'trace_tolerance': 1e-10,
            'hermiticity_tolerance': 1e-10,
            'positivity_tolerance': 1e-12,
            'energy_drift_threshold': 0.1,  # 10% energy drift
            'norm_tolerance': 1e-10,
            'max_eigenvalue_ratio': 1e12  # Condition number threshold
        }
        
        # Resource monitoring thresholds
        self.resource_thresholds = {
            'memory_warning': 80.0,  # % of available memory
            'memory_critical': 95.0,
            'cpu_warning': 90.0,     # % CPU usage
            'max_system_size': 10000  # Maximum number of atoms
        }
    
    def handle_convergence_failure(self, simulation_state: Dict[str, Any]) -> str:
        """
        Handle simulation convergence failures.
        
        Args:
            simulation_state: Current state of the simulation including parameters
            
        Returns:
            str: Recovery action recommendation
        """
        self.logger.warning("Handling convergence failure")
        
        # Log the convergence failure
        error_entry = {
            'timestamp': datetime.now(),
            'error_type': 'convergence_failure',
            'context': simulation_state,
            'traceback': traceback.format_stack()
        }
        self.error_log.append(error_entry)
        
        # Analyze the cause of convergence failure
        time_step = simulation_state.get('time_step', 0.1)
        system_size = simulation_state.get('system_size', 0)
        current_energy = simulation_state.get('current_energy', 0.0)
        
        recovery_action = "unknown"
        
        # Check if time step is too large
        if time_step > 0.5:
            recovery_action = "reduce_time_step"
            self.logger.info("Convergence failure likely due to large time step")
        
        # Check if system is too large
        elif system_size > self.resource_thresholds['max_system_size']:
            recovery_action = "reduce_system_size"
            self.logger.info("Convergence failure likely due to large system size")
        
        # Check for energy instability
        elif abs(current_energy) > 1e6:
            recovery_action = "reset_initial_conditions"
            self.logger.info("Convergence failure likely due to unstable energy")
        
        else:
            recovery_action = "increase_numerical_precision"
            self.logger.info("Convergence failure cause unclear, increasing precision")
        
        return recovery_action
    
    def handle_resource_exhaustion(self, system_size: int) -> Dict[str, Any]:
        """
        Handle computational resource limitations.
        
        Args:
            system_size: Size of the system causing resource issues
            
        Returns:
            Dict containing resource management recommendations
        """
        self.logger.warning(f"Handling resource exhaustion for system size {system_size}")
        
        # Get current resource usage
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        recommendations = {
            'action': 'unknown',
            'new_system_size': system_size,
            'memory_usage': memory_percent,
            'cpu_usage': cpu_percent,
            'available_memory_gb': available_memory_gb
        }
        
        # Determine appropriate action based on resource constraints
        if memory_percent > self.resource_thresholds['memory_critical']:
            # Critical memory situation - aggressive reduction needed
            recommendations['action'] = 'aggressive_reduction'
            recommendations['new_system_size'] = max(100, system_size // 4)
            self.logger.critical(f"Critical memory usage: {memory_percent}%")
        
        elif memory_percent > self.resource_thresholds['memory_warning']:
            # Warning level - moderate reduction
            recommendations['action'] = 'moderate_reduction'
            recommendations['new_system_size'] = max(500, system_size // 2)
            self.logger.warning(f"High memory usage: {memory_percent}%")
        
        elif system_size > self.resource_thresholds['max_system_size']:
            # System too large for efficient computation
            recommendations['action'] = 'size_optimization'
            recommendations['new_system_size'] = self.resource_thresholds['max_system_size']
            self.logger.info(f"System size {system_size} exceeds recommended maximum")
        
        else:
            # Use approximation methods
            recommendations['action'] = 'use_approximations'
            recommendations['approximation_methods'] = [
                'qm_mm_hybrid',
                'reduced_basis',
                'mean_field_approximation'
            ]
        
        # Log resource exhaustion event
        error_entry = {
            'timestamp': datetime.now(),
            'error_type': 'resource_exhaustion',
            'context': {
                'system_size': system_size,
                'memory_percent': memory_percent,
                'cpu_percent': cpu_percent,
                'recommendations': recommendations
            }
        }
        self.error_log.append(error_entry)
        
        return recommendations
    
    def handle_unphysical_results(self, results: SimulationResults) -> ValidationResult:
        """
        Handle detection of unphysical simulation results.
        
        Args:
            results: Simulation results to validate
            
        Returns:
            ValidationResult: Validation outcome with specific issues identified
        """
        self.logger.info("Validating simulation results for physical consistency")
        
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Validate energy conservation
            energy_validation = self._validate_energy_conservation(results.energy_trajectory)
            if not energy_validation.is_valid:
                validation_result.errors.extend(energy_validation.errors)
                validation_result.warnings.extend(energy_validation.warnings)
                validation_result.is_valid = False
            
            # Validate probability conservation (trace preservation)
            prob_validation = self._validate_probability_conservation(results.state_trajectory)
            if not prob_validation.is_valid:
                validation_result.errors.extend(prob_validation.errors)
                validation_result.warnings.extend(prob_validation.warnings)
                validation_result.is_valid = False
            
            # Validate quantum state properties
            state_validation = self._validate_quantum_states(results.state_trajectory)
            if not state_validation.is_valid:
                validation_result.errors.extend(state_validation.errors)
                validation_result.warnings.extend(state_validation.warnings)
                validation_result.is_valid = False
            
            # Validate coherence measures
            coherence_validation = self._validate_coherence_measures(results.coherence_measures)
            if not coherence_validation.is_valid:
                validation_result.errors.extend(coherence_validation.errors)
                validation_result.warnings.extend(coherence_validation.warnings)
                validation_result.is_valid = False
            
            # Log validation results
            if not validation_result.is_valid:
                error_entry = {
                    'timestamp': datetime.now(),
                    'error_type': 'unphysical_results',
                    'context': {
                        'num_errors': len(validation_result.errors),
                        'num_warnings': len(validation_result.warnings),
                        'errors': validation_result.errors,
                        'warnings': validation_result.warnings
                    }
                }
                self.error_log.append(error_entry)
                self.logger.error(f"Unphysical results detected: {len(validation_result.errors)} errors")
            else:
                self.logger.info("All physical validation checks passed")
        
        except Exception as e:
            validation_result.add_error(f"Validation process failed: {str(e)}")
            self.logger.error(f"Validation process failed: {str(e)}")
        
        return validation_result
    
    def generate_diagnostic_report(self, error_type: str, context: Dict[str, Any]) -> str:
        """
        Generate detailed diagnostic report for troubleshooting.
        
        Args:
            error_type: Type of error encountered
            context: Additional context information
            
        Returns:
            str: Formatted diagnostic report
        """
        report_lines = [
            "QBES Diagnostic Report",
            "=" * 50,
            f"Timestamp: {datetime.now()}",
            f"Error Type: {error_type}",
            "",
            "System Information:",
            f"  Memory Usage: {psutil.virtual_memory().percent:.1f}%",
            f"  CPU Usage: {psutil.cpu_percent(interval=1):.1f}%",
            f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB",
            "",
            "Error Context:"
        ]
        
        # Add context information
        for key, value in context.items():
            if isinstance(value, (int, float, str, bool)):
                report_lines.append(f"  {key}: {value}")
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                report_lines.append(f"  {key}: {value}")
            else:
                report_lines.append(f"  {key}: <complex object>")
        
        # Add recent error log entries
        report_lines.extend([
            "",
            "Recent Error Log:",
            "-" * 20
        ])
        
        recent_errors = self.error_log[-5:]  # Last 5 errors
        for i, error in enumerate(recent_errors):
            report_lines.append(f"{i+1}. {error['timestamp']} - {error['error_type']}")
        
        # Add recovery suggestions
        suggestions = self.suggest_recovery_actions(error_type)
        if suggestions:
            report_lines.extend([
                "",
                "Suggested Recovery Actions:",
                "-" * 25
            ])
            for i, suggestion in enumerate(suggestions):
                report_lines.append(f"{i+1}. {suggestion}")
        
        return "\n".join(report_lines)
    
    def suggest_recovery_actions(self, error_type: str) -> List[str]:
        """
        Suggest specific actions to recover from errors.
        
        Args:
            error_type: Type of error encountered
            
        Returns:
            List of suggested recovery actions
        """
        return self.recovery_strategies.get(error_type, [
            "Check system configuration",
            "Verify input parameters",
            "Consult documentation",
            "Contact support"
        ])
    
    def monitor_numerical_stability(self, state: DensityMatrix) -> ValidationResult:
        """
        Monitor numerical stability of quantum states during simulation.
        
        Args:
            state: Current quantum state to monitor
            
        Returns:
            ValidationResult: Stability assessment
        """
        validation_result = ValidationResult(is_valid=True)
        
        try:
            # Check trace preservation
            trace = np.trace(state.matrix)
            if abs(trace - 1.0) > self.stability_thresholds['trace_tolerance']:
                validation_result.add_error(f"Trace not preserved: {trace}")
            
            # Check Hermiticity
            hermiticity_error = np.max(np.abs(state.matrix - state.matrix.conj().T))
            if hermiticity_error > self.stability_thresholds['hermiticity_tolerance']:
                validation_result.add_error(f"Matrix not Hermitian: max error = {hermiticity_error}")
            
            # Check positive semidefinite property
            eigenvals = np.linalg.eigvals(state.matrix)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < -self.stability_thresholds['positivity_tolerance']:
                validation_result.add_error(f"Negative eigenvalue detected: {min_eigenval}")
            
            # Check condition number
            max_eigenval = np.max(eigenvals)
            if max_eigenval > 0 and min_eigenval > 0:
                condition_number = max_eigenval / min_eigenval
                if condition_number > self.stability_thresholds['max_eigenvalue_ratio']:
                    validation_result.add_warning(f"Poor conditioning: {condition_number}")
        
        except Exception as e:
            validation_result.add_error(f"Stability monitoring failed: {str(e)}")
        
        return validation_result
    
    def _initialize_recovery_strategies(self) -> Dict[str, List[str]]:
        """Initialize recovery strategies for different error types."""
        return {
            'convergence_failure': [
                "Reduce time step by factor of 2-10",
                "Increase numerical precision",
                "Check initial conditions",
                "Verify Hamiltonian construction",
                "Use adaptive time stepping"
            ],
            'resource_exhaustion': [
                "Reduce system size",
                "Use QM/MM hybrid approach",
                "Apply basis set reduction",
                "Increase available memory",
                "Use distributed computing"
            ],
            'unphysical_results': [
                "Check input parameters",
                "Verify noise model parameters",
                "Reduce time step",
                "Check for numerical instabilities",
                "Validate initial conditions"
            ],
            'numerical_instability': [
                "Increase numerical precision",
                "Use more stable integration method",
                "Reduce time step",
                "Check matrix conditioning",
                "Apply regularization techniques"
            ],
            'memory_error': [
                "Reduce system size",
                "Use sparse matrix representations",
                "Implement checkpointing",
                "Increase virtual memory",
                "Use out-of-core algorithms"
            ]
        }
    
    def _validate_energy_conservation(self, energy_trajectory: List[float]) -> ValidationResult:
        """Validate energy conservation throughout simulation."""
        validation_result = ValidationResult(is_valid=True)
        
        if len(energy_trajectory) < 2:
            validation_result.add_warning("Insufficient data for energy conservation check")
            return validation_result
        
        initial_energy = energy_trajectory[0]
        final_energy = energy_trajectory[-1]
        
        if abs(initial_energy) > 1e-10:  # Avoid division by zero
            energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
            if energy_drift > self.stability_thresholds['energy_drift_threshold']:
                validation_result.add_error(
                    f"Energy not conserved: {energy_drift*100:.2f}% drift"
                )
        
        # Check for sudden energy jumps
        energy_array = np.array(energy_trajectory)
        energy_diffs = np.abs(np.diff(energy_array))
        max_jump = np.max(energy_diffs) if len(energy_diffs) > 0 else 0.0
        
        if max_jump > abs(initial_energy) * 0.1:  # 10% jump threshold
            validation_result.add_warning(f"Large energy jump detected: {max_jump}")
        
        return validation_result
    
    def _validate_probability_conservation(self, state_trajectory: List[DensityMatrix]) -> ValidationResult:
        """Validate probability conservation (trace preservation)."""
        validation_result = ValidationResult(is_valid=True)
        
        for i, state in enumerate(state_trajectory):
            trace = np.trace(state.matrix)
            if abs(trace - 1.0) > self.stability_thresholds['trace_tolerance']:
                validation_result.add_error(
                    f"Probability not conserved at step {i}: trace = {trace}"
                )
        
        return validation_result
    
    def _validate_quantum_states(self, state_trajectory: List[DensityMatrix]) -> ValidationResult:
        """Validate quantum state properties."""
        validation_result = ValidationResult(is_valid=True)
        
        for i, state in enumerate(state_trajectory):
            # Check Hermiticity
            hermiticity_error = np.max(np.abs(state.matrix - state.matrix.conj().T))
            if hermiticity_error > self.stability_thresholds['hermiticity_tolerance']:
                validation_result.add_error(
                    f"Non-Hermitian state at step {i}: error = {hermiticity_error}"
                )
            
            # Check positive semidefinite
            eigenvals = np.linalg.eigvals(state.matrix)
            min_eigenval = np.min(eigenvals)
            if min_eigenval < -self.stability_thresholds['positivity_tolerance']:
                validation_result.add_error(
                    f"Negative eigenvalue at step {i}: {min_eigenval}"
                )
        
        return validation_result
    
    def _validate_coherence_measures(self, coherence_measures: Dict[str, List[float]]) -> ValidationResult:
        """Validate coherence measures for physical consistency."""
        validation_result = ValidationResult(is_valid=True)
        
        # Check purity values (should be between 0 and 1)
        if 'purity' in coherence_measures:
            purity_values = coherence_measures['purity']
            for i, purity in enumerate(purity_values):
                if not (0.0 <= purity <= 1.0):
                    validation_result.add_error(
                        f"Invalid purity at step {i}: {purity}"
                    )
        
        # Check von Neumann entropy (should be non-negative)
        if 'von_neumann_entropy' in coherence_measures:
            entropy_values = coherence_measures['von_neumann_entropy']
            for i, entropy in enumerate(entropy_values):
                if entropy < 0.0:
                    validation_result.add_error(
                        f"Negative entropy at step {i}: {entropy}"
                    )
        
        # Check coherence lifetime (should be positive)
        if 'coherence_lifetime' in coherence_measures:
            lifetime_values = coherence_measures['coherence_lifetime']
            for i, lifetime in enumerate(lifetime_values):
                if lifetime < 0.0:
                    validation_result.add_error(
                        f"Negative coherence lifetime at step {i}: {lifetime}"
                    )
        
        return validation_result