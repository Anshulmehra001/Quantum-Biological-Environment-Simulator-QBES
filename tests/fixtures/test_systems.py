"""
Test systems and fixtures for QBES testing.
"""

import numpy as np
from qbes.core.data_models import (
    DensityMatrix, Hamiltonian, QuantumSubsystem, 
    LindbladOperator, MolecularSystem, Atom
)


def create_two_level_system():
    """Create a simple two-level quantum system for testing."""
    # Create Hamiltonian
    hamiltonian_matrix = np.array([[0.0, 0.1], [0.1, 2.0]], dtype=complex)
    hamiltonian = Hamiltonian(
        matrix=hamiltonian_matrix,
        basis_labels=["ground", "excited"],
        time_dependent=False
    )
    
    # Create initial state (ground state)
    initial_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    initial_state = DensityMatrix(
        matrix=initial_matrix,
        basis_labels=["ground", "excited"],
        time=0.0
    )
    
    # Create Lindblad operator (decay)
    lindblad_matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    lindblad_operator = LindbladOperator(
        operator=lindblad_matrix,
        coupling_strength=0.1,
        description="Spontaneous decay"
    )
    
    return hamiltonian, initial_state, [lindblad_operator]


def create_three_level_system():
    """Create a three-level quantum system for testing."""
    # Create Hamiltonian (ladder system)
    hamiltonian_matrix = np.array([
        [0.0, 0.1, 0.0],
        [0.1, 1.0, 0.1],
        [0.0, 0.1, 2.0]
    ], dtype=complex)
    
    hamiltonian = Hamiltonian(
        matrix=hamiltonian_matrix,
        basis_labels=["ground", "intermediate", "excited"],
        time_dependent=False
    )
    
    # Create initial state (ground state)
    initial_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=complex)
    
    initial_state = DensityMatrix(
        matrix=initial_matrix,
        basis_labels=["ground", "intermediate", "excited"],
        time=0.0
    )
    
    # Create Lindblad operators
    # Decay from excited to intermediate
    decay_2_1 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ], dtype=complex)
    
    # Decay from intermediate to ground
    decay_1_0 = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=complex)
    
    lindblad_operators = [
        LindbladOperator(
            operator=decay_2_1,
            coupling_strength=0.05,
            description="Decay 2->1"
        ),
        LindbladOperator(
            operator=decay_1_0,
            coupling_strength=0.1,
            description="Decay 1->0"
        )
    ]
    
    return hamiltonian, initial_state, lindblad_operators


def create_photosynthetic_complex():
    """Create a simplified photosynthetic complex model."""
    # 4-site model (simplified FMO complex)
    site_energies = [0.0, 0.2, 0.1, 0.3]  # eV
    
    # Coupling matrix (symmetric)
    coupling_matrix = np.array([
        [0.0,  0.1,  0.05, 0.02],
        [0.1,  0.2,  0.08, 0.03],
        [0.05, 0.08, 0.1,  0.06],
        [0.02, 0.03, 0.06, 0.3]
    ], dtype=complex)
    
    # Set diagonal elements to site energies
    for i, energy in enumerate(site_energies):
        coupling_matrix[i, i] = energy
    
    hamiltonian = Hamiltonian(
        matrix=coupling_matrix,
        basis_labels=[f"site_{i}" for i in range(4)],
        time_dependent=False
    )
    
    # Initial state (excitation on site 0)
    initial_matrix = np.zeros((4, 4), dtype=complex)
    initial_matrix[0, 0] = 1.0
    
    initial_state = DensityMatrix(
        matrix=initial_matrix,
        basis_labels=[f"site_{i}" for i in range(4)],
        time=0.0
    )
    
    # Dephasing operators for each site
    lindblad_operators = []
    for i in range(4):
        dephasing_op = np.zeros((4, 4), dtype=complex)
        dephasing_op[i, i] = 1.0
        
        lindblad_operators.append(LindbladOperator(
            operator=dephasing_op,
            coupling_strength=0.01,  # Weak dephasing
            description=f"Dephasing site {i}"
        ))
    
    return hamiltonian, initial_state, lindblad_operators


def create_enzyme_active_site():
    """Create a simplified enzyme active site model."""
    # Two-level system representing substrate states
    hamiltonian_matrix = np.array([
        [0.0, 0.05],  # Ground state and coupling
        [0.05, 1.5]   # Excited state (transition state)
    ], dtype=complex)
    
    hamiltonian = Hamiltonian(
        matrix=hamiltonian_matrix,
        basis_labels=["reactant", "transition_state"],
        time_dependent=False
    )
    
    # Initial state (reactant)
    initial_matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    initial_state = DensityMatrix(
        matrix=initial_matrix,
        basis_labels=["reactant", "transition_state"],
        time=0.0
    )
    
    # Environmental coupling (protein fluctuations)
    protein_coupling = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    
    lindblad_operators = [
        LindbladOperator(
            operator=protein_coupling,
            coupling_strength=0.2,  # Strong protein coupling
            description="Protein environment coupling"
        )
    ]
    
    return hamiltonian, initial_state, lindblad_operators


def create_test_molecular_system():
    """Create a test molecular system with atoms and bonds."""
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
            position=np.array([1.4, 0.0, 0.0]),
            charge=0.0,
            mass=14.0,
            atom_id=1,
            residue_id=1,
            residue_name="TEST"
        ),
        Atom(
            element="O",
            position=np.array([0.0, 1.2, 0.0]),
            charge=0.0,
            mass=16.0,
            atom_id=2,
            residue_id=1,
            residue_name="TEST"
        ),
        Atom(
            element="H",
            position=np.array([2.0, 0.8, 0.0]),
            charge=0.0,
            mass=1.0,
            atom_id=3,
            residue_id=1,
            residue_name="TEST"
        )
    ]
    
    bonds = [(0, 1), (0, 2), (1, 3)]  # C-N, C-O, N-H bonds
    
    residues = {1: "TEST"}
    
    return MolecularSystem(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        system_name="Test System",
        total_charge=0.0
    )


def create_analytical_test_cases():
    """Create test cases with known analytical solutions."""
    test_cases = []
    
    # Case 1: Pure dephasing (analytical solution known)
    hamiltonian = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    initial_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    dephasing_op = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    
    test_cases.append({
        "name": "pure_dephasing",
        "hamiltonian": hamiltonian,
        "initial_state": initial_state,
        "lindblad_operators": [(dephasing_op, 0.1)],
        "analytical_solution": lambda t: np.array([
            [0.5, 0.5 * np.exp(-0.2 * t)],
            [0.5 * np.exp(-0.2 * t), 0.5]
        ], dtype=complex)
    })
    
    # Case 2: Pure decay (analytical solution known)
    hamiltonian = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=complex)
    initial_state = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    decay_op = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    
    test_cases.append({
        "name": "pure_decay",
        "hamiltonian": hamiltonian,
        "initial_state": initial_state,
        "lindblad_operators": [(decay_op, 0.1)],
        "analytical_solution": lambda t: np.array([
            [1.0 - np.exp(-0.1 * t), 0.0],
            [0.0, np.exp(-0.1 * t)]
        ], dtype=complex)
    })
    
    return test_cases


def create_benchmark_systems():
    """Create systems for benchmarking against literature."""
    systems = {}
    
    # Fenna-Matthews-Olson (FMO) complex parameters
    # Based on Adolphs & Renger, Biophys. J. 91, 2778 (2006)
    fmo_site_energies = np.array([
        12410, 12530, 12210, 12320, 12480, 12630, 12440
    ])  # cm^-1
    
    # Convert to eV (1 eV = 8065.5 cm^-1)
    fmo_site_energies = fmo_site_energies / 8065.5
    
    # Coupling matrix (simplified)
    fmo_coupling = np.array([
        [0.0,  -87.7, 5.5,   -5.9,  6.7,   -13.7, -9.9],
        [-87.7, 0.0,  30.8,  8.2,   0.7,   11.8,  4.3],
        [5.5,   30.8, 0.0,   -53.5, -2.2,  -9.6,  6.0],
        [-5.9,  8.2,  -53.5, 0.0,   -70.7, -17.0, -63.3],
        [6.7,   0.7,  -2.2,  -70.7, 0.0,   81.1,  -1.3],
        [-13.7, 11.8, -9.6,  -17.0, 81.1,  0.0,   39.7],
        [-9.9,  4.3,  6.0,   -63.3, -1.3,  39.7,  0.0]
    ]) / 8065.5  # Convert to eV
    
    # Set diagonal elements
    for i, energy in enumerate(fmo_site_energies):
        fmo_coupling[i, i] = energy
    
    systems["fmo_complex"] = {
        "hamiltonian": fmo_coupling,
        "description": "Fenna-Matthews-Olson complex",
        "reference": "Adolphs & Renger, Biophys. J. 91, 2778 (2006)",
        "expected_transfer_time": 1.5e-12,  # ~1.5 ps
        "expected_efficiency": 0.95
    }
    
    return systems


# Sample PDB content for testing
SAMPLE_PDB_CONTENT = """HEADER    TEST SYSTEM                             01-JAN-25   TEST            
TITLE     TEST SYSTEM FOR QBES TESTING                                          
ATOM      1  N   ALA A   1      20.154  16.967  27.462  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  27.462  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.693  16.849  27.462  1.00 20.00           C  
ATOM      4  O   ALA A   1      16.632  16.329  27.462  1.00 20.00           O  
ATOM      5  CB  ALA A   1      19.030  15.235  26.196  1.00 20.00           C  
ATOM      6  N   GLY A   2      17.693  18.115  27.462  1.00 20.00           N  
ATOM      7  CA  GLY A   2      16.456  18.863  27.462  1.00 20.00           C  
ATOM      8  C   GLY A   2      15.219  18.115  27.462  1.00 20.00           C  
ATOM      9  O   GLY A   2      14.158  18.635  27.462  1.00 20.00           O  
CONECT    1    2                                                                
CONECT    2    1    3    5                                                      
CONECT    3    2    4    6                                                      
CONECT    4    3                                                                
CONECT    5    2                                                                
CONECT    6    3    7                                                           
CONECT    7    6    8                                                           
CONECT    8    7    9                                                           
CONECT    9    8                                                                
END                                                                             
"""