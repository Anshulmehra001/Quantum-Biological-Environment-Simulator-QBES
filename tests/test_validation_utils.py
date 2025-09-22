"""
Unit tests for validation utilities.
"""

import pytest
import numpy as np
import tempfile
import os
from qbes.utils.validation import ValidationUtils


class TestValidationUtils:
    """Test cases for ValidationUtils."""
    
    def test_valid_physical_parameters(self):
        """Test validation of valid physical parameters."""
        params = {
            'temperature': 300.0,
            'time_step': 1e-15
        }
        result = ValidationUtils.validate_physical_parameters(params)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_negative_temperature_error(self):
        """Test that negative temperature produces error."""
        params = {'temperature': -100.0}
        result = ValidationUtils.validate_physical_parameters(params)
        assert not result.is_valid
        assert any("Temperature must be positive" in error for error in result.errors)
    
    def test_high_temperature_warning(self):
        """Test that very high temperature produces warning."""
        params = {'temperature': 1500.0}
        result = ValidationUtils.validate_physical_parameters(params)
        assert result.is_valid  # Still valid, just warning
        assert any("Temperature is very high" in warning for warning in result.warnings)
    
    def test_negative_time_step_error(self):
        """Test that negative time step produces error."""
        params = {'time_step': -1e-15}
        result = ValidationUtils.validate_physical_parameters(params)
        assert not result.is_valid
        assert any("Time step must be positive" in error for error in result.errors)
    
    def test_large_time_step_warning(self):
        """Test that large time step produces warning."""
        params = {'time_step': 1e-10}
        result = ValidationUtils.validate_physical_parameters(params)
        assert result.is_valid  # Still valid, just warning
        assert any("Time step may be too large" in warning for warning in result.warnings)


class TestMatrixValidation:
    """Test cases for matrix validation."""
    
    def test_valid_hermitian_matrix(self):
        """Test validation of Hermitian matrix."""
        matrix = np.array([[1.0, 1.0j], [-1.0j, 2.0]])
        result = ValidationUtils.validate_matrix_properties(matrix, "hermitian")
        assert result.is_valid
    
    def test_non_hermitian_matrix_error(self):
        """Test that non-Hermitian matrix produces error."""
        matrix = np.array([[1.0, 1.0], [0.0, 2.0]])  # Not Hermitian
        result = ValidationUtils.validate_matrix_properties(matrix, "hermitian")
        assert not result.is_valid
        assert any("not Hermitian" in error for error in result.errors)
    
    def test_valid_density_matrix(self):
        """Test validation of valid density matrix."""
        matrix = np.array([[0.5, 0.0], [0.0, 0.5]])  # Mixed state
        result = ValidationUtils.validate_matrix_properties(matrix, "density_matrix")
        assert result.is_valid
    
    def test_density_matrix_trace_error(self):
        """Test that density matrix with wrong trace produces error."""
        matrix = np.array([[0.3, 0.0], [0.0, 0.3]])  # Trace = 0.6
        result = ValidationUtils.validate_matrix_properties(matrix, "density_matrix")
        assert not result.is_valid
        assert any("trace is not 1" in error for error in result.errors)
    
    def test_density_matrix_negative_eigenvalue_error(self):
        """Test that density matrix with negative eigenvalues produces error."""
        matrix = np.array([[1.5, 0.0], [0.0, -0.5]])  # Negative eigenvalue
        result = ValidationUtils.validate_matrix_properties(matrix, "density_matrix")
        assert not result.is_valid
        assert any("negative eigenvalues" in error for error in result.errors)


class TestFileValidation:
    """Test cases for file validation."""
    
    def test_existing_file_validation(self):
        """Test validation of existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            result = ValidationUtils.validate_file_exists(tmp_path)
            assert result.is_valid
        finally:
            os.unlink(tmp_path)
    
    def test_nonexistent_file_error(self):
        """Test that nonexistent file produces error."""
        result = ValidationUtils.validate_file_exists("/nonexistent/file.txt")
        assert not result.is_valid
        assert any("File does not exist" in error for error in result.errors)
    
    def test_directory_instead_of_file_error(self):
        """Test that directory path produces error when file expected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = ValidationUtils.validate_file_exists(tmp_dir)
            assert not result.is_valid
            assert any("Path is not a file" in error for error in result.errors)