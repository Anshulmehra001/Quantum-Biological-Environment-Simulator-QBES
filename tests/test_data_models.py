"""
Unit tests for QBES data models.
"""

import pytest
import numpy as np
from qbes.core.data_models import (
    SimulationConfig, Atom, QuantumState, DensityMatrix,
    QuantumSubsystem, ValidationResult, CoherenceMetrics,
    MolecularSystem, StatisticalSummary, SimulationResults,
    LindbladOperator, Hamiltonian, SpectralDensity
)


class TestSimulationConfig:
    """Test cases for SimulationConfig."""
    
    def test_valid_config(self):
        """Test creation of valid configuration."""
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory="./output"
        )
        assert config.temperature == 300.0
        assert config.force_field == "amber14"  # default value
    
    def test_invalid_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SimulationConfig(
                system_pdb="test.pdb",
                temperature=-100.0,
                simulation_time=1e-12,
                time_step=1e-15,
                quantum_subsystem_selection="chromophores",
                noise_model_type="protein",
                output_directory="./output"
            )


class TestAtom:
    """Test cases for Atom data model."""
    
    def test_valid_atom(self):
        """Test creation of valid atom."""
        position = np.array([1.0, 2.0, 3.0])
        atom = Atom(
            element="C",
            position=position,
            charge=0.0,
            mass=12.01,
            atom_id=1
        )
        assert atom.element == "C"
        assert np.array_equal(atom.position, position)
    
    def test_invalid_position(self):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError, match="Position must be a 3D vector"):
            Atom(
                element="C",
                position=np.array([1.0, 2.0]),  # Only 2D
                charge=0.0,
                mass=12.01,
                atom_id=1
            )


class TestQuantumState:
    """Test cases for QuantumState."""
    
    def test_normalized_state(self):
        """Test creation of normalized quantum state."""
        coeffs = np.array([1.0, 0.0]) / np.sqrt(1.0)
        state = QuantumState(
            coefficients=coeffs,
            basis_labels=["ground", "excited"]
        )
        assert len(state.coefficients) == 2
        assert np.isclose(np.linalg.norm(state.coefficients), 1.0)
    
    def test_unnormalized_state_error(self):
        """Test that unnormalized state raises error."""
        with pytest.raises(ValueError, match="State is not normalized"):
            QuantumState(
                coefficients=np.array([1.0, 1.0]),  # Not normalized
                basis_labels=["ground", "excited"]
            )


class TestDensityMatrix:
    """Test cases for DensityMatrix."""
    
    def test_valid_density_matrix(self):
        """Test creation of valid density matrix."""
        # Pure state density matrix
        matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        dm = DensityMatrix(
            matrix=matrix,
            basis_labels=["ground", "excited"]
        )
        assert dm.matrix.shape == (2, 2)
        assert np.isclose(np.trace(dm.matrix), 1.0)
    
    def test_non_hermitian_error(self):
        """Test that non-Hermitian matrix raises error."""
        with pytest.raises(ValueError, match="Density matrix is not Hermitian"):
            DensityMatrix(
                matrix=np.array([[1.0, 1.0], [0.0, 0.0]]),  # Not Hermitian
                basis_labels=["ground", "excited"]
            )
    
    def test_trace_not_one_error(self):
        """Test that matrix with trace != 1 raises error."""
        with pytest.raises(ValueError, match="Density matrix trace is not 1"):
            DensityMatrix(
                matrix=np.array([[0.5, 0.0], [0.0, 0.0]]),  # Trace = 0.5
                basis_labels=["ground", "excited"]
            )


class TestValidationResult:
    """Test cases for ValidationResult."""
    
    def test_valid_result(self):
        """Test creation of valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """Test adding error invalidates result."""
        result = ValidationResult(is_valid=True)
        result.add_error("Test error")
        assert not result.is_valid
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding warning doesn't invalidate result."""
        result = ValidationResult(is_valid=True)
        result.add_warning("Test warning")
        assert result.is_valid
        assert "Test warning" in result.warnings


class TestQuantumSubsystem:
    """Test cases for QuantumSubsystem."""
    
    def test_valid_quantum_subsystem(self):
        """Test creation of valid quantum subsystem."""
        atoms = [
            Atom("C", np.array([0.0, 0.0, 0.0]), 0.0, 12.01, 1),
            Atom("N", np.array([1.0, 0.0, 0.0]), 0.0, 14.01, 2)
        ]
        basis_states = [
            QuantumState(np.array([1.0, 0.0]), ["ground", "excited"]),
            QuantumState(np.array([0.0, 1.0]), ["ground", "excited"])
        ]
        coupling_matrix = np.array([[0.0, 0.1], [0.1, 0.0]])
        
        subsystem = QuantumSubsystem(
            atoms=atoms,
            hamiltonian_parameters={"coupling": 0.1},
            coupling_matrix=coupling_matrix,
            basis_states=basis_states
        )
        
        assert len(subsystem.atoms) == 2
        assert subsystem.coupling_matrix.shape == (2, 2)
    
    def test_invalid_coupling_matrix_size(self):
        """Test that mismatched coupling matrix size raises error."""
        atoms = [Atom("C", np.array([0.0, 0.0, 0.0]), 0.0, 12.01, 1)]
        basis_states = [QuantumState(np.array([1.0, 0.0]), ["ground", "excited"])]
        coupling_matrix = np.array([[0.0, 0.1], [0.1, 0.0]])  # 2x2 but only 1 basis state
        
        with pytest.raises(ValueError, match="Coupling matrix size doesn't match"):
            QuantumSubsystem(
                atoms=atoms,
                hamiltonian_parameters={},
                coupling_matrix=coupling_matrix,
                basis_states=basis_states
            )


class TestCoherenceMetrics:
    """Test cases for CoherenceMetrics."""
    
    def test_valid_coherence_metrics(self):
        """Test creation of valid coherence metrics."""
        metrics = CoherenceMetrics(
            coherence_lifetime=1e-12,
            quantum_discord=0.5,
            entanglement_measure=0.3,
            purity=0.8,
            von_neumann_entropy=0.2
        )
        assert metrics.coherence_lifetime == 1e-12
        assert metrics.purity == 0.8
    
    def test_negative_coherence_lifetime_error(self):
        """Test that negative coherence lifetime raises error."""
        with pytest.raises(ValueError, match="Coherence lifetime cannot be negative"):
            CoherenceMetrics(
                coherence_lifetime=-1.0,
                quantum_discord=0.5,
                entanglement_measure=0.3,
                purity=0.8,
                von_neumann_entropy=0.2
            )
    
    def test_invalid_purity_error(self):
        """Test that invalid purity raises error."""
        with pytest.raises(ValueError, match="Purity must be between 0 and 1"):
            CoherenceMetrics(
                coherence_lifetime=1e-12,
                quantum_discord=0.5,
                entanglement_measure=0.3,
                purity=1.5,  # Invalid purity > 1
                von_neumann_entropy=0.2
            )


class TestMolecularSystem:
    """Test cases for MolecularSystem."""
    
    def test_valid_molecular_system(self):
        """Test creation of valid molecular system."""
        atoms = [
            Atom("C", np.array([0.0, 0.0, 0.0]), 0.0, 12.01, 1),
            Atom("N", np.array([1.0, 0.0, 0.0]), 0.0, 14.01, 2)
        ]
        bonds = [(0, 1)]
        residues = {1: "ALA"}
        
        system = MolecularSystem(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            system_name="test_system"
        )
        
        assert len(system.atoms) == 2
        assert system.system_name == "test_system"
        assert len(system.bonds) == 1
    
    def test_get_quantum_subsystem_atoms(self):
        """Test quantum subsystem atom extraction."""
        atoms = [Atom("C", np.array([i, 0.0, 0.0]), 0.0, 12.01, i) for i in range(15)]
        system = MolecularSystem(atoms=atoms, bonds=[], residues={})
        
        quantum_atoms = system.get_quantum_subsystem_atoms("test")
        assert len(quantum_atoms) == 10  # Returns first 10 atoms as per implementation


class TestStatisticalSummary:
    """Test cases for StatisticalSummary."""
    
    def test_valid_statistical_summary(self):
        """Test creation of valid statistical summary."""
        summary = StatisticalSummary(
            mean_values={"coherence": 0.5},
            std_deviations={"coherence": 0.1},
            confidence_intervals={"coherence": (0.4, 0.6)},
            sample_size=100
        )
        assert summary.sample_size == 100
        assert summary.mean_values["coherence"] == 0.5
    
    def test_invalid_sample_size_error(self):
        """Test that invalid sample size raises error."""
        with pytest.raises(ValueError, match="Sample size must be positive"):
            StatisticalSummary(
                mean_values={},
                std_deviations={},
                confidence_intervals={},
                sample_size=0
            )


class TestSimulationResults:
    """Test cases for SimulationResults."""
    
    def test_valid_simulation_results(self):
        """Test creation of valid simulation results."""
        # Create test density matrix
        dm = DensityMatrix(
            matrix=np.array([[1.0, 0.0], [0.0, 0.0]]),
            basis_labels=["ground", "excited"]
        )
        
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory="./output"
        )
        
        summary = StatisticalSummary(
            mean_values={},
            std_deviations={},
            confidence_intervals={},
            sample_size=1
        )
        
        results = SimulationResults(
            state_trajectory=[dm],
            coherence_measures={"purity": [1.0]},
            energy_trajectory=[0.0],
            decoherence_rates={"dephasing": 1e12},
            statistical_summary=summary,
            simulation_config=config
        )
        
        assert len(results.state_trajectory) == 1
        assert len(results.energy_trajectory) == 1
    
    def test_empty_trajectory_error(self):
        """Test that empty state trajectory raises error."""
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein",
            output_directory="./output"
        )
        
        summary = StatisticalSummary(
            mean_values={},
            std_deviations={},
            confidence_intervals={},
            sample_size=1
        )
        
        with pytest.raises(ValueError, match="State trajectory cannot be empty"):
            SimulationResults(
                state_trajectory=[],
                coherence_measures={},
                energy_trajectory=[],
                decoherence_rates={},
                statistical_summary=summary,
                simulation_config=config
            )


class TestLindbladOperator:
    """Test cases for LindbladOperator."""
    
    def test_valid_lindblad_operator(self):
        """Test creation of valid Lindblad operator."""
        operator = LindbladOperator(
            operator=np.array([[0.0, 1.0], [0.0, 0.0]]),
            coupling_strength=0.1,
            operator_type="dephasing"
        )
        assert operator.coupling_strength == 0.1
        assert operator.operator_type == "dephasing"
    
    def test_negative_coupling_strength_error(self):
        """Test that negative coupling strength raises error."""
        with pytest.raises(ValueError, match="Coupling strength cannot be negative"):
            LindbladOperator(
                operator=np.array([[0.0, 1.0], [0.0, 0.0]]),
                coupling_strength=-0.1
            )


class TestHamiltonian:
    """Test cases for Hamiltonian."""
    
    def test_valid_hamiltonian(self):
        """Test creation of valid Hamiltonian."""
        matrix = np.array([[1.0, 0.1], [0.1, 2.0]])  # Hermitian matrix
        hamiltonian = Hamiltonian(
            matrix=matrix,
            basis_labels=["ground", "excited"]
        )
        assert hamiltonian.matrix.shape == (2, 2)
        assert not hamiltonian.time_dependent
    
    def test_non_hermitian_hamiltonian_error(self):
        """Test that non-Hermitian Hamiltonian raises error."""
        with pytest.raises(ValueError, match="Hamiltonian is not Hermitian"):
            Hamiltonian(
                matrix=np.array([[1.0, 1.0], [0.0, 2.0]]),  # Not Hermitian
                basis_labels=["ground", "excited"]
            )
    
    def test_size_mismatch_error(self):
        """Test that size mismatch raises error."""
        with pytest.raises(ValueError, match="Hamiltonian matrix size doesn't match basis"):
            Hamiltonian(
                matrix=np.array([[1.0, 0.1], [0.1, 2.0]]),
                basis_labels=["ground"]  # Only 1 label for 2x2 matrix
            )


class TestSpectralDensity:
    """Test cases for SpectralDensity."""
    
    def test_valid_spectral_density(self):
        """Test creation of valid spectral density."""
        frequencies = np.linspace(0, 1000, 100)
        spectral_values = np.ones_like(frequencies)
        
        spectral_density = SpectralDensity(
            frequencies=frequencies,
            spectral_values=spectral_values,
            temperature=300.0,
            spectral_type="ohmic"
        )
        
        assert len(spectral_density.frequencies) == 100
        assert spectral_density.temperature == 300.0
    
    def test_mismatched_array_lengths_error(self):
        """Test that mismatched array lengths raise error."""
        with pytest.raises(ValueError, match="Frequency and spectral value arrays must have same length"):
            SpectralDensity(
                frequencies=np.array([1, 2, 3]),
                spectral_values=np.array([1, 2]),  # Different length
                temperature=300.0
            )
    
    def test_negative_temperature_error(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SpectralDensity(
                frequencies=np.array([1, 2, 3]),
                spectral_values=np.array([1, 2, 3]),
                temperature=-100.0
            )