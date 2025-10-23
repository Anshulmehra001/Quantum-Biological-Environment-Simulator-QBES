"""
Pytest configuration and shared fixtures for QBES tests.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add the project root to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from qbes.core.data_models import (
    DensityMatrix, Hamiltonian, QuantumSubsystem, 
    LindbladOperator, MolecularSystem, Atom
)
from qbes.quantum_engine import QuantumEngine
from qbes.simulation_engine import SimulationEngine
from qbes.config_manager import ConfigurationManager


@pytest.fixture
def quantum_engine():
    """Create a QuantumEngine instance for testing."""
    return QuantumEngine()


@pytest.fixture
def simulation_engine():
    """Create a SimulationEngine instance for testing."""
    return SimulationEngine()


@pytest.fixture
def config_manager():
    """Create a ConfigurationManager instance for testing."""
    return ConfigurationManager()


@pytest.fixture
def simple_density_matrix():
    """Create a simple 2x2 density matrix for testing."""
    matrix = np.array([[0.6, 0.2j], [-0.2j, 0.4]], dtype=complex)
    return DensityMatrix(
        matrix=matrix,
        basis_labels=["ground", "excited"],
        time=0.0
    )


@pytest.fixture
def two_level_hamiltonian():
    """Create a simple two-level Hamiltonian for testing."""
    matrix = np.array([[0.0, 0.1], [0.1, 2.0]], dtype=complex)
    return Hamiltonian(
        matrix=matrix,
        basis_labels=["ground", "excited"],
        time_dependent=False
    )


@pytest.fixture
def simple_quantum_subsystem():
    """Create a simple quantum subsystem for testing."""
    # Create mock atoms
    atoms = [
        Atom(
            element="C",
            position=np.array([0.0, 0.0, 0.0]),
            charge=0.0,
            mass=12.0,
            atom_id=0,
            residue_id=1,
            residue_name="TEST"
        ),
        Atom(
            element="N",
            position=np.array([1.0, 0.0, 0.0]),
            charge=0.0,
            mass=14.0,
            atom_id=1,
            residue_id=1,
            residue_name="TEST"
        )
    ]
    
    basis_states = ["ground", "excited"]
    coupling_matrix = np.array([[0.0, 0.1], [0.1, 2.0]])
    
    return QuantumSubsystem(
        atoms=atoms,
        basis_states=basis_states,
        coupling_matrix=coupling_matrix
    )


@pytest.fixture
def simple_lindblad_operator():
    """Create a simple Lindblad operator for testing."""
    operator = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return LindbladOperator(
        operator=operator,
        coupling_strength=0.1,
        description="Decay operator"
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="qbes_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pdb_content():
    """Sample PDB content for testing."""
    return """HEADER    TEST SYSTEM                             01-JAN-25   TEST            
TITLE     TEST SYSTEM FOR QBES TESTING                                          
ATOM      1  N   ALA A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  27.462  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.693  16.849  27.462  1.00 20.00           C  
ATOM      4  O   ALA A   1      16.632  16.329  27.462  1.00 20.00           O  
CONECT    1    2                                                                
CONECT    2    1    3                                                           
CONECT    3    2    4                                                           
CONECT    4    3                                                                
END                                                                             
"""


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "system": {
            "pdb_file": "test_system.pdb",
            "force_field": "amber14",
            "temperature": 300.0
        },
        "simulation": {
            "simulation_time": 1e-12,
            "time_step": 1e-15
        },
        "quantum_subsystem": {
            "selection_method": "chromophores",
            "max_quantum_atoms": 100
        },
        "noise_model": {
            "type": "protein_ohmic",
            "coupling_strength": 1.0
        },
        "output": {
            "directory": "./test_output",
            "save_trajectory": True
        }
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to all tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add 'integration' marker to all tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add 'benchmark' marker to all tests in benchmarks/ directory
        elif "benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Add 'slow' marker to tests that might take longer
        if "benchmark" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)