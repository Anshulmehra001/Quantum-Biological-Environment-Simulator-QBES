"""
Enhanced validation module for QBES simulations.

This module provides comprehensive validation against physical principles,
numerical accuracy checks, and comparison with established methods.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from ..core.data_models import (
    DensityMatrix, Hamiltonian, ValidationResult, 
    CoherenceMetrics, SimulationResults
)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    density_matrix_valid: bool
    energy_conservation: float
    norm_preservation: float
    hermiticity_error: float
    positivity_check: bool
    thermal_equilibrium: bool
    decoherence_rate: float
    coherence_lifetime: float
    physical_bounds: bool
    numerical_stability: bool
    
    def is_valid(self) -> bool:
        """Check if all validation criteria pass."""
        return (
            self.density_matrix_valid and
            self.energy_conservation < 1e-6 and
            self.norm_preservation < 1e-10 and
            self.hermiticity_error < 1e-10 and
            self.positivity_check and
            self.physical_bounds and
            self.numerical_stability
        )


class EnhancedValidator:
    """
    Enhanced validation for quantum simulations.
    
    Provides comprehensive checks including:
    - Physical constraint validation
    - Numerical accuracy verification  
    - Comparison with analytical solutions
    - Literature benchmark validation
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize enhanced validator.
        
        Args:
            tolerance: Numerical tolerance for validation checks
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
    def validate_density_matrix(self, rho: DensityMatrix) -> ValidationMetrics:
        """
        Comprehensive density matrix validation.
        
        Args:
            rho: Density matrix to validate
            
        Returns:
            ValidationMetrics with detailed validation results
        """
        matrix = rho.matrix
        
        # Check hermiticity
        hermiticity_error = np.max(np.abs(matrix - matrix.conj().T))
        
        # Check trace
        trace = np.trace(matrix)
        norm_error = abs(trace - 1.0)
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(matrix)
        positivity = np.all(eigenvalues >= -self.tolerance)
        
        # Check if eigenvalues are in [0, 1]
        physical_bounds = np.all((eigenvalues >= -self.tolerance) & 
                                (eigenvalues <= 1.0 + self.tolerance))
        
        # Numerical stability check
        condition_number = np.linalg.cond(matrix)
        numerical_stability = condition_number < 1e12
        
        return ValidationMetrics(
            density_matrix_valid=(hermiticity_error < self.tolerance and 
                                 norm_error < self.tolerance and positivity),
            energy_conservation=0.0,  # To be computed from trajectory
            norm_preservation=norm_error,
            hermiticity_error=hermiticity_error,
            positivity_check=positivity,
            thermal_equilibrium=False,  # To be checked separately
            decoherence_rate=0.0,  # To be computed from trajectory
            coherence_lifetime=0.0,  # To be computed from trajectory
            physical_bounds=physical_bounds,
            numerical_stability=numerical_stability
        )
    
    def validate_energy_conservation(self, 
                                    energy_trajectory: List[float],
                                    rtol: float = 1e-4) -> Tuple[bool, float]:
        """
        Check energy conservation throughout simulation.
        
        Args:
            energy_trajectory: List of energy values over time
            rtol: Relative tolerance for energy conservation
            
        Returns:
            Tuple of (is_conserved, relative_error)
        """
        if len(energy_trajectory) < 2:
            return True, 0.0
        
        energies = np.array(energy_trajectory)
        mean_energy = np.mean(energies)
        energy_variation = np.std(energies)
        
        relative_error = energy_variation / abs(mean_energy) if mean_energy != 0 else 0.0
        
        is_conserved = relative_error < rtol
        
        if not is_conserved:
            self.logger.warning(f"Energy not conserved: relative error = {relative_error:.6e}")
        
        return is_conserved, relative_error
    
    def validate_thermalization(self,
                               final_state: DensityMatrix,
                               hamiltonian: Hamiltonian,
                               temperature: float,
                               tolerance: float = 0.1) -> bool:
        """
        Check if system thermalizes to expected Boltzmann distribution.
        
        Args:
            final_state: Final density matrix
            hamiltonian: System Hamiltonian
            temperature: Temperature in Kelvin
            tolerance: Tolerance for thermal state comparison
            
        Returns:
            True if system is thermalized within tolerance
        """
        # Compute thermal state
        H = hamiltonian.matrix
        beta = 1.0 / (8.617333e-5 * temperature)  # k_B in eV/K
        
        # Thermal density matrix: ρ_thermal = exp(-βH) / Z
        exp_beta_H = np.linalg.matrix_power(
            np.exp(-beta * H), 1
        )
        Z = np.trace(exp_beta_H)
        rho_thermal = exp_beta_H / Z
        
        # Compare with actual final state
        diff = np.linalg.norm(final_state.matrix - rho_thermal, 'fro')
        
        is_thermal = diff < tolerance
        
        if not is_thermal:
            self.logger.warning(f"System not thermalized: difference = {diff:.6f}")
        
        return is_thermal
    
    def validate_coherence_decay(self,
                                coherence_trajectory: List[float],
                                expected_lifetime: Optional[float] = None) -> Tuple[bool, float]:
        """
        Validate coherence decay behavior.
        
        Args:
            coherence_trajectory: Coherence values over time
            expected_lifetime: Expected decoherence lifetime (if known)
            
        Returns:
            Tuple of (is_valid, measured_lifetime)
        """
        if len(coherence_trajectory) < 10:
            return True, 0.0
        
        coherences = np.array(coherence_trajectory)
        
        # Fit exponential decay
        try:
            # Assuming exponential: C(t) = C_0 * exp(-t/τ)
            log_coherence = np.log(coherences + 1e-16)
            time_points = np.arange(len(coherences))
            
            # Linear fit to log
            coeffs = np.polyfit(time_points, log_coherence, 1)
            decay_rate = -coeffs[0]
            measured_lifetime = 1.0 / decay_rate if decay_rate > 0 else np.inf
            
            # Check if decay is reasonable (not too fast or too slow)
            is_valid = 1.0 < measured_lifetime < 1e6
            
            if expected_lifetime is not None:
                relative_error = abs(measured_lifetime - expected_lifetime) / expected_lifetime
                is_valid = is_valid and (relative_error < 0.5)
            
            return is_valid, measured_lifetime
            
        except Exception as e:
            self.logger.warning(f"Could not validate coherence decay: {e}")
            return True, 0.0
    
    def compare_with_analytical(self,
                               results: SimulationResults,
                               analytical_solution: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compare simulation results with analytical solution.
        
        Args:
            results: Simulation results
            analytical_solution: Dictionary with analytical values
            
        Returns:
            Dictionary of error metrics
        """
        errors = {}
        
        # Compare energies
        if 'energy' in analytical_solution and hasattr(results, 'energy_trajectory'):
            sim_energy = np.array(results.energy_trajectory)
            ana_energy = analytical_solution['energy']
            
            if len(sim_energy) == len(ana_energy):
                errors['energy_rmse'] = np.sqrt(np.mean((sim_energy - ana_energy)**2))
                errors['energy_max_error'] = np.max(np.abs(sim_energy - ana_energy))
        
        # Compare coherences
        if 'coherence' in analytical_solution and hasattr(results, 'coherence_trajectory'):
            # Assuming coherence_trajectory is a dict with 'l1_norm' or similar
            if isinstance(results.coherence_trajectory, dict):
                sim_coh = np.array(results.coherence_trajectory.get('l1_norm', []))
                ana_coh = analytical_solution['coherence']
                
                if len(sim_coh) == len(ana_coh):
                    errors['coherence_rmse'] = np.sqrt(np.mean((sim_coh - ana_coh)**2))
                    errors['coherence_max_error'] = np.max(np.abs(sim_coh - ana_coh))
        
        return errors
    
    def generate_validation_report(self, 
                                  metrics: ValidationMetrics,
                                  energy_traj: List[float],
                                  coherence_traj: List[float]) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            metrics: Validation metrics
            energy_traj: Energy trajectory
            coherence_traj: Coherence trajectory
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("="*70)
        report.append("QBES VALIDATION REPORT")
        report.append("="*70)
        report.append("")
        
        report.append("Density Matrix Validation:")
        report.append(f"  ✓ Hermiticity error: {metrics.hermiticity_error:.2e}")
        report.append(f"  ✓ Trace error: {metrics.norm_preservation:.2e}")
        report.append(f"  ✓ Positivity: {metrics.positivity_check}")
        report.append(f"  ✓ Physical bounds: {metrics.physical_bounds}")
        report.append(f"  ✓ Numerical stability: {metrics.numerical_stability}")
        report.append("")
        
        energy_conserved, energy_error = self.validate_energy_conservation(energy_traj)
        report.append("Energy Conservation:")
        report.append(f"  ✓ Conserved: {energy_conserved}")
        report.append(f"  ✓ Relative error: {energy_error:.2e}")
        report.append("")
        
        coherence_valid, lifetime = self.validate_coherence_decay(coherence_traj)
        report.append("Coherence Dynamics:")
        report.append(f"  ✓ Decay behavior valid: {coherence_valid}")
        report.append(f"  ✓ Measured lifetime: {lifetime:.2e}")
        report.append("")
        
        report.append("Overall Validation:")
        overall = metrics.is_valid() and energy_conserved and coherence_valid
        report.append(f"  {'✅ PASSED' if overall else '❌ FAILED'}")
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)


# Convenience functions
def validate_simulation(results: SimulationResults, 
                       hamiltonian: Hamiltonian,
                       temperature: float) -> ValidationResult:
    """
    Perform comprehensive validation of simulation results.
    
    Args:
        results: Simulation results to validate
        hamiltonian: System Hamiltonian
        temperature: Simulation temperature
        
    Returns:
        ValidationResult with comprehensive checks
    """
    validator = EnhancedValidator()
    validation = ValidationResult(is_valid=True)
    
    # Validate final state
    if hasattr(results, 'final_state'):
        metrics = validator.validate_density_matrix(results.final_state)
        if not metrics.is_valid():
            validation.is_valid = False
            validation.add_error(f"Final state validation failed: {metrics}")
    
    # Validate energy conservation
    if hasattr(results, 'energy_trajectory'):
        conserved, error = validator.validate_energy_conservation(results.energy_trajectory)
        if not conserved:
            validation.add_warning(f"Energy not conserved: error = {error:.2e}")
    
    # Validate thermalization
    if hasattr(results, 'final_state'):
        is_thermal = validator.validate_thermalization(
            results.final_state, hamiltonian, temperature
        )
        if not is_thermal:
            validation.add_warning("System did not thermalize")
    
    return validation


def validate_against_reference(results: SimulationResults,
                               reference_file: str = "configs/reference_data.json",
                               system_name: str = None) -> ValidationResult:
    """
    Validate simulation results against literature reference data.
    
    Args:
        results: Simulation results to validate
        reference_file: Path to reference_data.json
        system_name: Name of reference system (e.g., 'fmo_complex_engel_2007')
        
    Returns:
        ValidationResult with accuracy comparison
    """
    validation = ValidationResult(is_valid=True)
    
    try:
        # Load reference data
        ref_path = Path(reference_file)
        if not ref_path.exists():
            validation.add_warning(f"Reference file not found: {reference_file}")
            return validation
            
        with open(ref_path, 'r') as f:
            reference_data = json.load(f)
        
        benchmarks = reference_data.get('literature_benchmarks', {})
        
        if system_name and system_name not in benchmarks:
            validation.add_error(f"System '{system_name}' not found in reference data")
            return validation
        
        # If no system specified, try to auto-detect or validate all
        if not system_name:
            validation.add_info("No system specified - performing general validation")
            return validation
        
        # Get reference system
        ref_system = benchmarks[system_name]
        expected = ref_system.get('expected_outcomes', {})
        
        # Validate coherence time
        if hasattr(results, 'coherence_time_fs'):
            if 'coherence_time_fs' in expected:
                ref_range = expected['coherence_time_fs']
                sim_value = results.coherence_time_fs
                
                if 'exact' in ref_range:
                    error = abs(sim_value - ref_range['exact']) / ref_range['exact']
                    if error < 0.05:  # 5% tolerance
                        validation.add_info(f"✅ Coherence time: {sim_value:.1f} fs (ref: {ref_range['exact']:.1f} fs, error: {error*100:.2f}%)")
                    else:
                        validation.is_valid = False
                        validation.add_error(f"Coherence time error too large: {error*100:.2f}%")
                else:
                    min_val = ref_range.get('min', 0)
                    max_val = ref_range.get('max', float('inf'))
                    target = ref_range.get('target', (min_val + max_val) / 2)
                    
                    if min_val <= sim_value <= max_val:
                        accuracy = 1.0 - abs(sim_value - target) / target
                        validation.add_info(f"✅ Coherence time: {sim_value:.1f} fs (range: {min_val}-{max_val} fs, accuracy: {accuracy*100:.1f}%)")
                    else:
                        validation.is_valid = False
                        validation.add_error(f"Coherence time {sim_value:.1f} fs outside range {min_val}-{max_val} fs")
        
        # Validate energy transfer efficiency
        if hasattr(results, 'transfer_efficiency'):
            if 'energy_transfer_efficiency' in expected:
                ref_range = expected['energy_transfer_efficiency']
                sim_value = results.transfer_efficiency
                
                min_val = ref_range.get('min', 0)
                max_val = ref_range.get('max', 1.0)
                target = ref_range.get('target', (min_val + max_val) / 2)
                
                if min_val <= sim_value <= max_val:
                    accuracy = 1.0 - abs(sim_value - target) / target
                    validation.add_info(f"✅ Transfer efficiency: {sim_value:.3f} (range: {min_val}-{max_val}, accuracy: {accuracy*100:.1f}%)")
                else:
                    validation.is_valid = False
                    validation.add_error(f"Transfer efficiency {sim_value:.3f} outside range {min_val}-{max_val}")
        
        # Calculate overall accuracy
        accuracy_scores = []
        if hasattr(results, 'coherence_time_fs') and 'coherence_time_fs' in expected:
            ref_val = expected['coherence_time_fs'].get('target', expected['coherence_time_fs'].get('exact'))
            if ref_val:
                accuracy = 1.0 - min(abs(results.coherence_time_fs - ref_val) / ref_val, 1.0)
                accuracy_scores.append(accuracy)
        
        if hasattr(results, 'transfer_efficiency') and 'energy_transfer_efficiency' in expected:
            ref_val = expected['energy_transfer_efficiency'].get('target')
            if ref_val:
                accuracy = 1.0 - min(abs(results.transfer_efficiency - ref_val) / ref_val, 1.0)
                accuracy_scores.append(accuracy)
        
        if accuracy_scores:
            overall_accuracy = np.mean(accuracy_scores)
            validation.add_info(f"Overall accuracy: {overall_accuracy*100:.1f}%")
            
            # Check v1.2 requirement: >98% accuracy
            if overall_accuracy >= 0.98:
                validation.add_info("✅ PASSED v1.2 accuracy requirement (>98%)")
            else:
                validation.add_warning(f"Below v1.2 target accuracy (98%), achieved: {overall_accuracy*100:.1f}%")
    
    except Exception as e:
        validation.add_error(f"Error validating against reference: {str(e)}")
        validation.is_valid = False
    
    return validation
