"""
Error handling and recovery utilities.
"""

import logging
import traceback
import psutil
import numpy as np
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..core.interfaces import ErrorHandlerInterface
from ..core.data_models import SimulationResults, ValidationResult, DensityMatrix


class ImprovedErrorHandler:
    """
    Enhanced error handler with specific suggestions for common issues.
    
    This class provides user-friendly error messages with actionable solutions
    for common configuration mistakes and runtime errors.
    """
    
    def __init__(self):
        """Initialize the improved error handler."""
        self.common_errors = self._initialize_common_error_patterns()
        self.suggestion_database = self._initialize_suggestion_database()
    
    def handle_file_not_found(self, filepath: str, context: str = "") -> str:
        """Provide specific guidance for missing files."""
        suggestions = [
            f"• Check that the file path is correct: '{filepath}'",
            "• Ensure the file exists in the specified location",
            "• Use absolute paths if relative paths aren't working",
            "• Check file permissions and access rights"
        ]
        
        # Add context-specific suggestions
        if filepath.endswith('.pdb'):
            suggestions.extend([
                "• Verify the PDB file is properly formatted",
                "• Try downloading the PDB file again if it's from a database",
                "• Check if the file extension should be .pdb or .PDB"
            ])
        elif filepath.endswith(('.yaml', '.yml')):
            suggestions.extend([
                "• Generate a new configuration file with: qbes generate-config config.yaml",
                "• Use the interactive wizard: qbes generate-config config.yaml --interactive"
            ])
        
        # Check if similar files exist
        directory = os.path.dirname(filepath) or "."
        filename = os.path.basename(filepath)
        
        if os.path.exists(directory):
            similar_files = self._find_similar_files(directory, filename)
            if similar_files:
                suggestions.append(f"• Did you mean one of these files? {', '.join(similar_files[:3])}")
        
        error_msg = f"File not found: {filepath}"
        if context:
            error_msg += f" (Context: {context})"
        
        return self._format_error_with_suggestions(error_msg, suggestions)
    
    def handle_invalid_parameter(self, param: str, value: Any, expected: str, context: str = "") -> str:
        """Provide specific guidance for invalid parameters."""
        suggestions = []
        
        # Parameter-specific suggestions
        if param.lower() in ['temperature', 'temp']:
            suggestions.extend([
                "• Temperature must be positive (in Kelvin)",
                "• Typical values: 300K (room temp), 310K (body temp), 77K (liquid nitrogen)",
                "• Check if you meant to use Celsius and need to convert to Kelvin"
            ])
        elif param.lower() in ['time_step', 'timestep', 'dt']:
            suggestions.extend([
                "• Time step must be positive and small (typically 1e-15 to 1e-12 seconds)",
                "• For quantum systems, use femtosecond time steps (1e-15 s)",
                "• Larger time steps may cause numerical instability"
            ])
        elif param.lower() in ['simulation_time', 'total_time']:
            suggestions.extend([
                "• Simulation time must be positive",
                "• Typical values: 1e-12 s (1 ps) to 1e-9 s (1 ns)",
                "• Start with shorter times for testing"
            ])
        elif 'path' in param.lower() or 'file' in param.lower():
            suggestions.extend([
                "• Use forward slashes (/) or double backslashes (\\\\) in paths",
                "• Avoid spaces in file paths or use quotes",
                "• Check that the directory exists"
            ])
        else:
            suggestions.append(f"• Expected format: {expected}")
        
        # Value-specific suggestions
        if isinstance(value, str):
            if not value.strip():
                suggestions.append("• Parameter cannot be empty")
            elif value.lower() in ['none', 'null', 'n/a']:
                suggestions.append("• Use a valid value instead of None/null")
        elif isinstance(value, (int, float)):
            if value <= 0 and 'positive' in expected.lower():
                suggestions.append("• Value must be greater than zero")
            elif value < 0 and 'non-negative' in expected.lower():
                suggestions.append("• Value must be zero or positive")
        
        error_msg = f"Invalid parameter '{param}': got '{value}', expected {expected}"
        if context:
            error_msg += f" (Context: {context})"
        
        return self._format_error_with_suggestions(error_msg, suggestions)
    
    def handle_configuration_error(self, config_error: str, config_file: str = "") -> str:
        """Provide specific guidance for configuration errors."""
        suggestions = []
        
        error_lower = config_error.lower()
        
        if 'yaml' in error_lower or 'syntax' in error_lower:
            suggestions.extend([
                "• Check YAML syntax - ensure proper indentation (use spaces, not tabs)",
                "• Verify all quotes are properly closed",
                "• Use a YAML validator online to check syntax",
                "• Generate a new config file: qbes generate-config new_config.yaml"
            ])
        elif 'missing' in error_lower or 'required' in error_lower:
            suggestions.extend([
                "• Check that all required fields are present in the configuration",
                "• Use the interactive wizard: qbes generate-config config.yaml --interactive",
                "• Compare with a working example configuration"
            ])
        elif 'pdb' in error_lower:
            suggestions.extend([
                "• Ensure the PDB file path is correct and the file exists",
                "• Check that the PDB file is properly formatted",
                "• Try using an absolute path to the PDB file"
            ])
        else:
            suggestions.extend([
                "• Validate your configuration: qbes validate config.yaml",
                "• Generate a new configuration file as a reference",
                "• Check the documentation for configuration examples"
            ])
        
        error_msg = f"Configuration error: {config_error}"
        if config_file:
            error_msg += f" in file '{config_file}'"
        
        return self._format_error_with_suggestions(error_msg, suggestions)
    
    def handle_dependency_error(self, missing_dependency: str, import_error: str = "") -> str:
        """Provide specific installation instructions for missing dependencies."""
        suggestions = []
        
        # Common dependency installation instructions
        dependency_instructions = {
            'numpy': "pip install numpy",
            'scipy': "pip install scipy",
            'matplotlib': "pip install matplotlib",
            'yaml': "pip install PyYAML",
            'click': "pip install click",
            'psutil': "pip install psutil",
            'openmm': "conda install -c conda-forge openmm",
            'mdtraj': "conda install -c conda-forge mdtraj",
            'rdkit': "conda install -c conda-forge rdkit"
        }
        
        # Find installation command
        install_cmd = None
        for dep, cmd in dependency_instructions.items():
            if dep.lower() in missing_dependency.lower():
                install_cmd = cmd
                break
        
        if install_cmd:
            suggestions.extend([
                f"• Install the missing dependency: {install_cmd}",
                "• If using conda, try: conda install -c conda-forge " + missing_dependency.lower(),
                "• Ensure you're in the correct Python environment"
            ])
        else:
            suggestions.extend([
                f"• Install the missing dependency: pip install {missing_dependency}",
                f"• Try conda if pip fails: conda install {missing_dependency}",
                "• Check if the package name is spelled correctly"
            ])
        
        # General suggestions
        suggestions.extend([
            "• Update pip: python -m pip install --upgrade pip",
            "• Check your Python environment is activated",
            "• Try installing QBES again: pip install -e ."
        ])
        
        error_msg = f"Missing dependency: {missing_dependency}"
        if import_error:
            error_msg += f" ({import_error})"
        
        return self._format_error_with_suggestions(error_msg, suggestions)
    
    def handle_memory_error(self, system_size: int = 0, operation: str = "") -> str:
        """Provide specific guidance for memory-related errors."""
        suggestions = [
            "• Reduce the system size or simulation time",
            "• Close other applications to free up memory",
            "• Use a machine with more RAM",
            "• Enable checkpointing to reduce memory usage"
        ]
        
        if system_size > 0:
            if system_size > 10000:
                suggestions.insert(0, f"• System size ({system_size} atoms) is very large - consider reducing to <5000 atoms")
            elif system_size > 5000:
                suggestions.insert(0, f"• System size ({system_size} atoms) is large - try reducing to <2000 atoms")
        
        if operation:
            if 'matrix' in operation.lower():
                suggestions.append("• Use sparse matrix representations if available")
            elif 'trajectory' in operation.lower():
                suggestions.append("• Reduce trajectory saving frequency")
        
        # Add system-specific suggestions
        try:
            available_gb = psutil.virtual_memory().available / (1024**3)
            suggestions.append(f"• Available memory: {available_gb:.1f} GB")
            
            if available_gb < 2:
                suggestions.append("• Very low memory available - close other applications")
            elif available_gb < 8:
                suggestions.append("• Limited memory - consider using a smaller system")
        except:
            pass
        
        error_msg = "Memory error: Insufficient memory for operation"
        if operation:
            error_msg += f" ({operation})"
        
        return self._format_error_with_suggestions(error_msg, suggestions)
    
    def handle_numerical_error(self, error_type: str, context: str = "") -> str:
        """Provide specific guidance for numerical errors."""
        suggestions = []
        
        error_lower = error_type.lower()
        
        if 'convergence' in error_lower:
            suggestions.extend([
                "• Reduce the time step by a factor of 2-10",
                "• Check initial conditions are reasonable",
                "• Verify system parameters are physical",
                "• Try a different integration method"
            ])
        elif 'singular' in error_lower or 'invert' in error_lower:
            suggestions.extend([
                "• Check for linear dependencies in the basis set",
                "• Add regularization to avoid singular matrices",
                "• Verify the Hamiltonian is properly constructed",
                "• Check for zero eigenvalues"
            ])
        elif 'overflow' in error_lower or 'underflow' in error_lower:
            suggestions.extend([
                "• Reduce time step to avoid numerical overflow",
                "• Check parameter values are reasonable",
                "• Use double precision arithmetic",
                "• Scale the problem to avoid extreme values"
            ])
        elif 'nan' in error_lower or 'inf' in error_lower:
            suggestions.extend([
                "• Check for division by zero in calculations",
                "• Verify all input parameters are finite",
                "• Reduce time step to improve stability",
                "• Check initial conditions"
            ])
        else:
            suggestions.extend([
                "• Reduce time step for better numerical stability",
                "• Check all parameters are within reasonable ranges",
                "• Verify input data is properly formatted"
            ])
        
        error_msg = f"Numerical error: {error_type}"
        if context:
            error_msg += f" (Context: {context})"
        
        return self._format_error_with_suggestions(error_msg, suggestions)
    
    def suggest_configuration_fixes(self, validation_errors: List[str]) -> List[str]:
        """Suggest fixes for common configuration validation errors."""
        suggestions = []
        
        for error in validation_errors:
            error_lower = error.lower()
            
            if 'pdb' in error_lower and 'not found' in error_lower:
                suggestions.append("Fix PDB file path in configuration")
            elif 'temperature' in error_lower:
                suggestions.append("Set temperature to a positive value (e.g., 300.0 K)")
            elif 'time' in error_lower and 'step' in error_lower:
                suggestions.append("Set time_step to a small positive value (e.g., 1e-15 s)")
            elif 'simulation_time' in error_lower:
                suggestions.append("Set simulation_time to a positive value (e.g., 1e-12 s)")
            elif 'output' in error_lower and 'directory' in error_lower:
                suggestions.append("Create output directory or set writable path")
            elif 'selection' in error_lower:
                suggestions.append("Check quantum subsystem selection criteria")
            else:
                suggestions.append(f"Review and fix: {error}")
        
        return suggestions
    
    def _format_error_with_suggestions(self, error_msg: str, suggestions: List[str]) -> str:
        """Format error message with suggestions."""
        formatted_msg = f"\n{'='*60}\n"
        formatted_msg += f"❌ ERROR\n"
        formatted_msg += f"{'='*60}\n"
        formatted_msg += f"{error_msg}\n\n"
        
        if suggestions:
            formatted_msg += f"💡 SUGGESTIONS:\n"
            for suggestion in suggestions:
                formatted_msg += f"{suggestion}\n"
        
        formatted_msg += f"{'='*60}\n"
        
        return formatted_msg
    
    def _find_similar_files(self, directory: str, target_filename: str) -> List[str]:
        """Find files with similar names in the directory."""
        try:
            files = os.listdir(directory)
            target_lower = target_filename.lower()
            target_base = os.path.splitext(target_lower)[0]
            
            similar = []
            for file in files:
                file_lower = file.lower()
                file_base = os.path.splitext(file_lower)[0]
                
                # Check for similar names
                if (target_base in file_base or file_base in target_base or 
                    abs(len(target_base) - len(file_base)) <= 2):
                    similar.append(file)
            
            return similar[:5]  # Return up to 5 similar files
        except:
            return []
    
    def _initialize_common_error_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for common errors."""
        return {
            'file_not_found': [
                'No such file or directory',
                'FileNotFoundError',
                'cannot find file',
                'file does not exist'
            ],
            'invalid_parameter': [
                'ValueError',
                'invalid value',
                'out of range',
                'must be positive'
            ],
            'configuration_error': [
                'YAMLError',
                'configuration invalid',
                'missing required field',
                'syntax error'
            ],
            'dependency_error': [
                'ModuleNotFoundError',
                'ImportError',
                'No module named',
                'cannot import'
            ],
            'memory_error': [
                'MemoryError',
                'out of memory',
                'insufficient memory',
                'allocation failed'
            ],
            'numerical_error': [
                'convergence failed',
                'singular matrix',
                'overflow',
                'underflow',
                'NaN',
                'infinity'
            ]
        }
    
    def _initialize_suggestion_database(self) -> Dict[str, List[str]]:
        """Initialize database of suggestions for different error types."""
        return {
            'general': [
                "Check the documentation for examples",
                "Validate your configuration file",
                "Try with a smaller test system first",
                "Ensure all dependencies are installed"
            ],
            'installation': [
                "Reinstall QBES: pip install -e .",
                "Update your Python environment",
                "Check system requirements",
                "Try using conda instead of pip"
            ],
            'performance': [
                "Reduce system size",
                "Use shorter simulation times",
                "Enable checkpointing",
                "Close other applications"
            ]
        }


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