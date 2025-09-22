"""
Validation utilities for QBES components.
"""

import numpy as np
from typing import Any, Dict

from ..core.data_models import ValidationResult


class ValidationUtils:
    """
    Utility functions for validating simulation parameters and results.
    """
    
    @staticmethod
    def validate_physical_parameters(parameters: Dict[str, Any]) -> ValidationResult:
        """Validate that parameters are within physically reasonable ranges."""
        result = ValidationResult(is_valid=True)
        
        # Temperature validation
        if 'temperature' in parameters:
            temp = parameters['temperature']
            if temp <= 0:
                result.add_error("Temperature must be positive")
            elif temp > 1000:  # Kelvin
                result.add_warning("Temperature is very high (>1000K)")
        
        # Time step validation
        if 'time_step' in parameters:
            dt = parameters['time_step']
            if dt <= 0:
                result.add_error("Time step must be positive")
            elif dt > 1e-12:  # seconds
                result.add_warning("Time step may be too large for quantum dynamics")
        
        return result
    
    @staticmethod
    def validate_matrix_properties(matrix: np.ndarray, matrix_type: str) -> ValidationResult:
        """Validate mathematical properties of matrices."""
        result = ValidationResult(is_valid=True)
        
        if matrix_type == "hermitian":
            if not np.allclose(matrix, matrix.conj().T):
                result.add_error("Matrix is not Hermitian")
        
        elif matrix_type == "density_matrix":
            # Check Hermiticity
            if not np.allclose(matrix, matrix.conj().T):
                result.add_error("Density matrix is not Hermitian")
            
            # Check trace = 1
            trace = np.trace(matrix)
            if not np.isclose(trace, 1.0, rtol=1e-10):
                result.add_error(f"Density matrix trace is not 1: {trace}")
            
            # Check positive semidefinite
            eigenvals = np.linalg.eigvals(matrix)
            if np.any(eigenvals < -1e-10):
                result.add_error("Density matrix has negative eigenvalues")
        
        return result
    
    @staticmethod
    def validate_file_exists(filepath: str) -> ValidationResult:
        """Validate that a file exists and is readable."""
        import os
        result = ValidationResult(is_valid=True)
        
        if not os.path.exists(filepath):
            result.add_error(f"File does not exist: {filepath}")
        elif not os.path.isfile(filepath):
            result.add_error(f"Path is not a file: {filepath}")
        elif not os.access(filepath, os.R_OK):
            result.add_error(f"File is not readable: {filepath}")
        
        return result