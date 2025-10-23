"""
Unit tests for NoiseModelFactory create_noise_model method.
"""

import pytest
import numpy as np
from qbes.noise_models import NoiseModelFactory, OhmicNoiseModel, ProteinNoiseModel, MembraneNoiseModel, SolventNoiseModel


class TestNoiseModelFactory:
    """Test cases for NoiseModelFactory create_noise_model method."""
    
    def test_create_ohmic_noise_model(self):
        """Test creating ohmic noise model."""
        model = NoiseModelFactory.create_noise_model("ohmic", temperature=300.0)
        
        assert isinstance(model, OhmicNoiseModel)
        assert model.model_type == "ohmic"
        assert model.parameters["coupling_strength"] == 0.1  # default value
        assert model.parameters["cutoff_frequency"] == 1.0   # default value
    
    def test_create_protein_ohmic_noise_model(self):
        """Test creating protein_ohmic noise model."""
        model = NoiseModelFactory.create_noise_model("protein_ohmic", temperature=300.0)
        
        assert isinstance(model, ProteinNoiseModel)
        assert model.model_type == "protein_ohmic"
        assert "temperature" in model.parameters
        assert model.parameters["temperature"] == 300.0
    
    def test_create_membrane_noise_model(self):
        """Test creating membrane noise model."""
        model = NoiseModelFactory.create_noise_model("membrane", temperature=300.0)
        
        assert isinstance(model, MembraneNoiseModel)
        assert model.model_type == "membrane"
        assert "lipid_composition" in model.parameters
        assert "membrane_thickness" in model.parameters
    
    def test_create_solvent_ionic_noise_model(self):
        """Test creating solvent_ionic noise model."""
        model = NoiseModelFactory.create_noise_model("solvent_ionic", temperature=300.0)
        
        assert isinstance(model, SolventNoiseModel)
        assert model.model_type == "solvent_ionic"
        assert "ionic_strength" in model.parameters
        assert "solvent_type" in model.parameters
    
    def test_create_noise_model_with_custom_parameters(self):
        """Test creating noise model with custom parameters."""
        model = NoiseModelFactory.create_noise_model(
            "ohmic", 
            temperature=310.0,
            coupling_strength=0.2,
            cutoff_frequency=1.5
        )
        
        assert isinstance(model, OhmicNoiseModel)
        assert model.parameters["coupling_strength"] == 0.2
        assert model.parameters["cutoff_frequency"] == 1.5
    
    def test_create_noise_model_invalid_type(self):
        """Test creating noise model with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown noise model type"):
            NoiseModelFactory.create_noise_model("invalid_type", temperature=300.0)
    
    def test_create_noise_model_default_temperature(self):
        """Test creating noise model with default temperature."""
        model = NoiseModelFactory.create_noise_model("ohmic")
        
        assert isinstance(model, OhmicNoiseModel)
        # Should work without specifying temperature
    
    def test_all_supported_model_types(self):
        """Test that all supported model types can be created."""
        supported_types = ["ohmic", "protein_ohmic", "membrane", "solvent_ionic"]
        
        for model_type in supported_types:
            model = NoiseModelFactory.create_noise_model(model_type, temperature=300.0)
            assert model is not None
            assert hasattr(model, 'get_spectral_density')
            assert hasattr(model, 'calculate_decoherence_rates')
    
    def test_spectral_density_calculation(self):
        """Test that created models can calculate spectral density."""
        model = NoiseModelFactory.create_noise_model("ohmic", temperature=300.0)
        
        # Test spectral density calculation
        frequency = 1.0  # rad/s
        temperature = 300.0  # K
        
        spectral_density = model.get_spectral_density(frequency, temperature)
        assert isinstance(spectral_density, float)
        assert spectral_density >= 0.0
    
    def test_decoherence_rates_calculation(self):
        """Test that created models can calculate decoherence rates."""
        from qbes.core.data_models import QuantumSubsystem, QuantumState, Atom
        
        model = NoiseModelFactory.create_noise_model("ohmic", temperature=300.0)
        
        # Create test atoms
        atoms = [
            Atom(element="C", position=np.array([0.0, 0.0, 0.0]), charge=0.0, mass=12.0, atom_id=1),
            Atom(element="N", position=np.array([1.0, 0.0, 0.0]), charge=0.0, mass=14.0, atom_id=2)
        ]
        
        # Create test quantum states
        ground_state = QuantumState(
            coefficients=np.array([1.0, 0.0]),
            basis_labels=["ground", "excited"]
        )
        excited_state = QuantumState(
            coefficients=np.array([0.0, 1.0]),
            basis_labels=["ground", "excited"]
        )
        basis_states = [ground_state, excited_state]
        
        # Create coupling matrix
        coupling_matrix = np.array([[0.0, 0.1], [0.1, 1.0]])
        
        # Create hamiltonian parameters
        hamiltonian_parameters = {"coupling_strength": 0.1, "energy_gap": 1.0}
        
        system = QuantumSubsystem(
            atoms=atoms,
            hamiltonian_parameters=hamiltonian_parameters,
            coupling_matrix=coupling_matrix,
            basis_states=basis_states,
            subsystem_id="test"
        )
        
        rates = model.calculate_decoherence_rates(system, temperature=300.0)
        
        assert isinstance(rates, dict)
        assert "dephasing" in rates
        assert "relaxation" in rates
        assert all(rate >= 0.0 for rate in rates.values())


if __name__ == "__main__":
    pytest.main([__file__])