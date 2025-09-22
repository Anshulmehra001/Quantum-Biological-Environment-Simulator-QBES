"""
Unit tests for QBES interfaces.
"""

import pytest
from abc import ABC
from qbes.core.interfaces import (
    QuantumEngineInterface, MDEngineInterface, NoiseModelInterface,
    AnalysisInterface, VisualizationInterface, ConfigurationManagerInterface
)


class TestInterfaceDefinitions:
    """Test that interfaces are properly defined as abstract base classes."""
    
    def test_quantum_engine_interface_is_abstract(self):
        """Test that QuantumEngineInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            QuantumEngineInterface()
    
    def test_md_engine_interface_is_abstract(self):
        """Test that MDEngineInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            MDEngineInterface()
    
    def test_noise_model_interface_is_abstract(self):
        """Test that NoiseModelInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            NoiseModelInterface()
    
    def test_analysis_interface_is_abstract(self):
        """Test that AnalysisInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            AnalysisInterface()
    
    def test_visualization_interface_is_abstract(self):
        """Test that VisualizationInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            VisualizationInterface()
    
    def test_configuration_manager_interface_is_abstract(self):
        """Test that ConfigurationManagerInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            ConfigurationManagerInterface()


class TestInterfaceInheritance:
    """Test that interfaces properly inherit from ABC."""
    
    def test_interfaces_inherit_from_abc(self):
        """Test that all interfaces inherit from ABC."""
        interfaces = [
            QuantumEngineInterface,
            MDEngineInterface, 
            NoiseModelInterface,
            AnalysisInterface,
            VisualizationInterface,
            ConfigurationManagerInterface
        ]
        
        for interface in interfaces:
            assert issubclass(interface, ABC)
    
    def test_interface_methods_are_abstract(self):
        """Test that interface methods are marked as abstract."""
        # Check that QuantumEngineInterface has abstract methods
        abstract_methods = QuantumEngineInterface.__abstractmethods__
        expected_methods = {
            'initialize_hamiltonian',
            'evolve_state', 
            'calculate_coherence_measures',
            'apply_lindblad_operators',
            'validate_quantum_state',
            'calculate_expectation_value'
        }
        assert expected_methods.issubset(abstract_methods)