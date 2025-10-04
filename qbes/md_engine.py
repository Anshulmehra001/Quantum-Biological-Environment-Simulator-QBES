"""
Molecular dynamics simulation interface with OpenMM integration.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import os
import tempfile
import logging

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from openmm import Platform
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    openmm = None
    app = None
    unit = None
    Platform = None
    logging.warning("OpenMM not available. MD functionality will be limited.")

try:
    import mdtraj as md
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
    logging.warning("MDTraj not available. Trajectory analysis will be limited.")

from .core.interfaces import MDEngineInterface
from .core.data_models import MolecularSystem, SpectralDensity, Atom, ValidationResult


class MDEngine(MDEngineInterface):
    """
    Handles molecular dynamics simulations for environmental noise generation using OpenMM.
    
    This class provides an interface to OpenMM for molecular dynamics simulations and extracts
    quantum parameters from classical trajectories for quantum decoherence modeling.
    """
    
    def __init__(self, platform_name: str = "CPU"):
        """
        Initialize the MD engine with OpenMM.
        
        Args:
            platform_name: OpenMM platform to use ("CPU", "CUDA", "OpenCL", "Reference")
        """
        if not HAS_OPENMM:
            raise ImportError("OpenMM is required for MD functionality. Install with: pip install openmm")
        
        self.system = None
        self.topology = None
        self.positions = None
        self.simulation = None
        self.trajectory_data = None
        self.force_field = None
        
        # Set up OpenMM platform
        self.platform = self._setup_platform(platform_name)
        
        # Available force fields
        self.available_force_fields = {
            'amber14': 'amber14-all.xml',
            'amber99sb': 'amber99sb.xml', 
            'charmm36': 'charmm36.xml',
            'amoeba2013': 'amoeba2013.xml'
        }
        
        # Available water models
        self.available_water_models = {
            'tip3p': 'tip3p.xml',
            'tip4pew': 'tip4pew.xml',
            'tip5p': 'tip5p.xml',
            'spce': 'spce.xml'
        }
        
        logging.info(f"MDEngine initialized with platform: {self.platform.getName()}")
    
    def _setup_platform(self, platform_name: str):
        """Set up OpenMM platform with error handling."""
        try:
            platform = Platform.getPlatformByName(platform_name)
            logging.info(f"Using OpenMM platform: {platform_name}")
            return platform
        except Exception as e:
            logging.warning(f"Failed to initialize {platform_name} platform: {e}")
            logging.info("Falling back to CPU platform")
            return Platform.getPlatformByName("CPU")
    
    def initialize_system(self, pdb_file: str, force_field: str) -> MolecularSystem:
        """
        Initialize MD system from PDB file with specified force field.
        
        Args:
            pdb_file: Path to PDB file
            force_field: Force field name (e.g., 'amber14', 'charmm36')
            
        Returns:
            MolecularSystem object containing parsed molecular structure
            
        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If force field is not supported
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")
        
        if force_field not in self.available_force_fields:
            raise ValueError(f"Unsupported force field: {force_field}. "
                           f"Available: {list(self.available_force_fields.keys())}")
        
        try:
            # Load PDB file
            pdb = app.PDBFile(pdb_file)
            self.topology = pdb.topology
            self.positions = pdb.positions
            
            # Set up force field
            forcefield_files = [self.available_force_fields[force_field]]
            self.force_field = app.ForceField(*forcefield_files)
            
            # Create OpenMM system
            self.system = self.force_field.createSystem(
                self.topology,
                nonbondedMethod=app.NoCutoff,  # Will be updated in setup_environment
                constraints=app.HBonds,
                rigidWater=True
            )
            
            # Convert to our MolecularSystem format
            molecular_system = self._convert_to_molecular_system(pdb)
            
            logging.info(f"Successfully initialized system from {pdb_file} with {force_field}")
            logging.info(f"System contains {molecular_system.atoms.__len__()} atoms")
            
            return molecular_system
            
        except Exception as e:
            logging.error(f"Failed to initialize system: {e}")
            raise RuntimeError(f"System initialization failed: {e}")
    
    def _convert_to_molecular_system(self, pdb) -> MolecularSystem:
        """Convert OpenMM PDB to our MolecularSystem format."""
        atoms = []
        bonds = []
        residues = {}
        
        # Extract atoms
        for atom_idx, atom in enumerate(pdb.topology.atoms()):
            position = self.positions[atom_idx].value_in_unit(unit.nanometer)
            
            atom_obj = Atom(
                element=atom.element.symbol,
                position=np.array(position),
                charge=0.0,  # Will be updated from force field if needed
                mass=atom.element.mass.value_in_unit(unit.dalton),
                atom_id=atom_idx,
                residue_id=atom.residue.id,
                residue_name=atom.residue.name
            )
            atoms.append(atom_obj)
            
            # Track residues
            if atom.residue.id not in residues:
                residues[atom.residue.id] = atom.residue.name
        
        # Extract bonds
        for bond in pdb.topology.bonds():
            atom1_idx = bond[0].index
            atom2_idx = bond[1].index
            bonds.append((atom1_idx, atom2_idx))
        
        # Calculate total charge (placeholder - would need force field info)
        total_charge = 0.0
        
        return MolecularSystem(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            system_name=os.path.basename(pdb.name) if hasattr(pdb, 'name') else "unknown",
            total_charge=total_charge
        )
    
    def run_trajectory(self, duration: float, time_step: float, 
                      temperature: float) -> Dict[str, np.ndarray]:
        """
        Run MD trajectory and return atomic coordinates over time.
        
        Args:
            duration: Simulation duration in picoseconds
            time_step: Integration time step in femtoseconds
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary containing trajectory data:
            - 'positions': Array of shape (n_frames, n_atoms, 3) in nm
            - 'times': Array of time points in ps
            - 'energies': Array of potential energies in kJ/mol
            
        Raises:
            RuntimeError: If system not initialized or simulation fails
        """
        if self.system is None or self.topology is None or self.positions is None:
            raise RuntimeError("System must be initialized before running trajectory")
        
        try:
            # Create integrator
            integrator = openmm.LangevinIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,  # friction coefficient
                time_step * unit.femtosecond
            )
            
            # Create simulation
            self.simulation = app.Simulation(
                self.topology,
                self.system,
                integrator,
                self.platform
            )
            
            # Set initial positions
            self.simulation.context.setPositions(self.positions)
            
            # Set initial velocities
            self.simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
            
            # Calculate number of steps and reporting frequency
            total_steps = int(duration * 1000 / time_step)  # Convert ps to fs
            report_frequency = max(1, total_steps // 1000)  # Report ~1000 frames
            
            # Storage for trajectory data
            positions_list = []
            times_list = []
            energies_list = []
            
            logging.info(f"Starting MD trajectory: {duration} ps, {total_steps} steps")
            logging.info(f"Temperature: {temperature} K, Time step: {time_step} fs")
            
            # Run simulation
            for step in range(0, total_steps, report_frequency):
                # Run simulation steps
                self.simulation.step(report_frequency)
                
                # Get current state
                state = self.simulation.context.getState(
                    getPositions=True,
                    getEnergy=True
                )
                
                # Store data
                positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                time_ps = step * time_step / 1000.0  # Convert fs to ps
                
                positions_list.append(positions)
                energies_list.append(energy)
                times_list.append(time_ps)
                
                # Progress logging
                if step % (total_steps // 10) == 0:
                    progress = (step / total_steps) * 100
                    logging.info(f"MD Progress: {progress:.1f}% - Energy: {energy:.2f} kJ/mol")
            
            # Convert to numpy arrays
            trajectory_data = {
                'positions': np.array(positions_list),
                'times': np.array(times_list),
                'energies': np.array(energies_list)
            }
            
            # Store for later use
            self.trajectory_data = trajectory_data
            
            logging.info(f"MD trajectory completed: {len(positions_list)} frames")
            logging.info(f"Average energy: {np.mean(energies_list):.2f} ± {np.std(energies_list):.2f} kJ/mol")
            
            return trajectory_data
            
        except Exception as e:
            logging.error(f"MD trajectory failed: {e}")
            raise RuntimeError(f"MD trajectory failed: {e}")
    
    def extract_quantum_parameters(self, trajectory: Dict[str, np.ndarray], 
                                  quantum_atoms: List[int]) -> Dict[str, np.ndarray]:
        """
        Extract time-dependent quantum parameters from MD trajectory.
        
        Args:
            trajectory: Trajectory data from run_trajectory()
            quantum_atoms: List of atom indices for quantum subsystem
            
        Returns:
            Dictionary containing time-dependent parameters:
            - 'distances': Inter-atomic distances for quantum atoms
            - 'angles': Bond angles involving quantum atoms
            - 'dihedrals': Dihedral angles for quantum atoms
            - 'coupling_fluctuations': Electronic coupling fluctuations
            
        Raises:
            ValueError: If quantum_atoms indices are invalid
        """
        if 'positions' not in trajectory:
            raise ValueError("Trajectory must contain position data")
        
        positions = trajectory['positions']  # Shape: (n_frames, n_atoms, 3)
        n_frames, n_atoms, _ = positions.shape
        
        # Validate quantum atom indices
        if max(quantum_atoms) >= n_atoms:
            raise ValueError(f"Quantum atom index {max(quantum_atoms)} exceeds system size {n_atoms}")
        
        try:
            # Extract positions for quantum atoms only
            quantum_positions = positions[:, quantum_atoms, :]  # Shape: (n_frames, n_quantum, 3)
            n_quantum = len(quantum_atoms)
            
            # Calculate pairwise distances
            distances = np.zeros((n_frames, n_quantum, n_quantum))
            for frame in range(n_frames):
                for i in range(n_quantum):
                    for j in range(i + 1, n_quantum):
                        dist = np.linalg.norm(quantum_positions[frame, i] - quantum_positions[frame, j])
                        distances[frame, i, j] = dist
                        distances[frame, j, i] = dist  # Symmetric
            
            # Calculate bond angles (for triplets of atoms)
            angles = []
            if n_quantum >= 3:
                for i in range(n_quantum - 2):
                    angle_series = []
                    for frame in range(n_frames):
                        # Vectors from middle atom to outer atoms
                        v1 = quantum_positions[frame, i] - quantum_positions[frame, i + 1]
                        v2 = quantum_positions[frame, i + 2] - quantum_positions[frame, i + 1]
                        
                        # Calculate angle
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                        angle = np.arccos(cos_angle)
                        angle_series.append(angle)
                    
                    angles.append(np.array(angle_series))
            
            # Estimate coupling fluctuations based on distance fluctuations
            # Simple exponential decay model: J(r) = J0 * exp(-beta * r)
            coupling_fluctuations = np.zeros((n_frames, n_quantum, n_quantum))
            j0 = 100.0  # Base coupling in cm^-1
            beta = 1.0  # Decay parameter in nm^-1
            
            for frame in range(n_frames):
                for i in range(n_quantum):
                    for j in range(i + 1, n_quantum):
                        r = distances[frame, i, j]
                        coupling = j0 * np.exp(-beta * r)
                        coupling_fluctuations[frame, i, j] = coupling
                        coupling_fluctuations[frame, j, i] = coupling
            
            # Calculate dihedral angles (for quartets of atoms)
            dihedrals = []
            if n_quantum >= 4:
                for i in range(n_quantum - 3):
                    dihedral_series = []
                    for frame in range(n_frames):
                        # Four consecutive atoms
                        p1 = quantum_positions[frame, i]
                        p2 = quantum_positions[frame, i + 1]
                        p3 = quantum_positions[frame, i + 2]
                        p4 = quantum_positions[frame, i + 3]
                        
                        # Calculate dihedral angle
                        b1 = p2 - p1
                        b2 = p3 - p2
                        b3 = p4 - p3
                        
                        n1 = np.cross(b1, b2)
                        n2 = np.cross(b2, b3)
                        
                        n1_norm = np.linalg.norm(n1)
                        n2_norm = np.linalg.norm(n2)
                        
                        if n1_norm > 1e-10 and n2_norm > 1e-10:
                            cos_dihedral = np.dot(n1, n2) / (n1_norm * n2_norm)
                            cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
                            dihedral = np.arccos(cos_dihedral)
                        else:
                            dihedral = 0.0
                        
                        dihedral_series.append(dihedral)
                    
                    dihedrals.append(np.array(dihedral_series))
            
            parameters = {
                'distances': distances,
                'angles': np.array(angles) if angles else np.array([]),
                'dihedrals': np.array(dihedrals) if dihedrals else np.array([]),
                'coupling_fluctuations': coupling_fluctuations
            }
            
            logging.info(f"Extracted quantum parameters for {n_quantum} atoms over {n_frames} frames")
            
            return parameters
            
        except Exception as e:
            logging.error(f"Quantum parameter extraction failed: {e}")
            raise RuntimeError(f"Quantum parameter extraction failed: {e}")
    
    def analyze_parameter_fluctuations(self, parameters: Dict[str, np.ndarray], 
                                     time_step: float) -> Dict[str, Dict[str, float]]:
        """
        Analyze statistical properties of parameter fluctuations.
        
        Args:
            parameters: Parameter time series from extract_quantum_parameters
            time_step: Time step between data points in ps
            
        Returns:
            Dictionary containing statistical analysis for each parameter type
        """
        try:
            analysis = {}
            
            for param_name, param_data in parameters.items():
                if param_data.size == 0:
                    continue
                
                # Flatten multi-dimensional arrays for analysis
                if param_data.ndim > 1:
                    flattened = param_data.flatten()
                else:
                    flattened = param_data
                
                # Remove any NaN or infinite values
                valid_data = flattened[np.isfinite(flattened)]
                
                if len(valid_data) == 0:
                    continue
                
                # Calculate statistics
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'variance': float(np.var(valid_data)),
                    'skewness': float(self._calculate_skewness(valid_data)),
                    'kurtosis': float(self._calculate_kurtosis(valid_data))
                }
                
                # Calculate autocorrelation time if we have time series data
                if param_data.ndim >= 1 and len(param_data) > 10:
                    # Take first time series for autocorrelation analysis
                    if param_data.ndim == 1:
                        time_series = param_data
                    else:
                        # For matrices, take the first off-diagonal element
                        if param_data.ndim == 3:  # (time, i, j)
                            time_series = param_data[:, 0, 1] if param_data.shape[1] > 1 else param_data[:, 0, 0]
                        else:
                            time_series = param_data[0, :]
                    
                    # Calculate autocorrelation
                    autocorr = self._calculate_autocorrelation(time_series)
                    
                    # Find correlation time (where autocorr drops to 1/e)
                    correlation_time = self._find_correlation_time(autocorr, time_step)
                    stats['correlation_time_ps'] = correlation_time
                    
                    # Calculate power spectral density characteristics
                    psd_stats = self._analyze_power_spectrum(time_series, time_step)
                    stats.update(psd_stats)
                
                analysis[param_name] = stats
            
            logging.info(f"Parameter fluctuation analysis completed for {len(analysis)} parameter types")
            
            return analysis
            
        except Exception as e:
            logging.error(f"Parameter fluctuation analysis failed: {e}")
            raise RuntimeError(f"Parameter fluctuation analysis failed: {e}")
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3.0  # Excess kurtosis
            return kurtosis
        except Exception:
            return 0.0
    
    def _analyze_power_spectrum(self, time_series: np.ndarray, time_step: float) -> Dict[str, float]:
        """Analyze power spectral density characteristics."""
        try:
            # Calculate power spectral density
            n_points = len(time_series)
            dt = time_step * 1e-12  # Convert ps to s
            
            # Remove mean and apply window
            centered = time_series - np.mean(time_series)
            window = np.hanning(n_points)
            windowed = centered * window
            
            # FFT and power spectrum
            fft_result = np.fft.fft(windowed)
            power_spectrum = np.abs(fft_result) ** 2
            frequencies = np.fft.fftfreq(n_points, dt)
            
            # Take positive frequencies only
            positive_mask = frequencies > 0
            pos_freqs = frequencies[positive_mask]
            pos_power = power_spectrum[positive_mask]
            
            # Convert to cm^-1
            c_light = 2.998e10  # cm/s
            freqs_cm = pos_freqs / (c_light * 100)
            
            # Calculate spectral characteristics
            total_power = np.sum(pos_power)
            if total_power == 0:
                return {'dominant_frequency_cm': 0.0, 'spectral_width_cm': 0.0, 'spectral_centroid_cm': 0.0}
            
            # Dominant frequency (peak of spectrum)
            peak_idx = np.argmax(pos_power)
            dominant_freq = freqs_cm[peak_idx]
            
            # Spectral centroid (weighted average frequency)
            spectral_centroid = np.sum(freqs_cm * pos_power) / total_power
            
            # Spectral width (standard deviation of frequency distribution)
            spectral_width = np.sqrt(np.sum(pos_power * (freqs_cm - spectral_centroid) ** 2) / total_power)
            
            return {
                'dominant_frequency_cm': float(dominant_freq),
                'spectral_width_cm': float(spectral_width),
                'spectral_centroid_cm': float(spectral_centroid)
            }
            
        except Exception as e:
            logging.warning(f"Power spectrum analysis failed: {e}")
            return {'dominant_frequency_cm': 0.0, 'spectral_width_cm': 0.0, 'spectral_centroid_cm': 0.0}
    
    def _calculate_autocorrelation(self, time_series: np.ndarray) -> np.ndarray:
        """Calculate normalized autocorrelation function."""
        try:
            # Remove mean
            centered = time_series - np.mean(time_series)
            
            # Calculate autocorrelation using FFT for efficiency
            n = len(centered)
            
            # Pad with zeros to avoid circular correlation
            padded = np.zeros(2 * n)
            padded[:n] = centered
            
            # FFT-based correlation
            fft_data = np.fft.fft(padded)
            autocorr_fft = np.fft.ifft(fft_data * np.conj(fft_data))
            
            # Take real part and normalize
            autocorr = np.real(autocorr_fft[:n])
            autocorr = autocorr / autocorr[0]  # Normalize to 1 at t=0
            
            return autocorr
            
        except Exception as e:
            logging.warning(f"Autocorrelation calculation failed: {e}")
            return np.array([1.0])  # Return trivial autocorrelation
    
    def _find_correlation_time(self, autocorr: np.ndarray, time_step: float) -> float:
        """Find correlation time where autocorrelation drops to 1/e."""
        try:
            target = 1.0 / np.e  # 1/e ≈ 0.368
            
            # Find first point where autocorr drops below target
            below_target = np.where(autocorr < target)[0]
            
            if len(below_target) > 0:
                correlation_steps = below_target[0]
                correlation_time = correlation_steps * time_step
            else:
                # If never drops below 1/e, use the point where it's closest
                closest_idx = np.argmin(np.abs(autocorr - target))
                correlation_time = closest_idx * time_step
            
            return float(correlation_time)
            
        except Exception as e:
            logging.warning(f"Correlation time calculation failed: {e}")
            return 0.0
    
    def extract_coupling_matrix_evolution(self, trajectory: Dict[str, np.ndarray], 
                                        quantum_atoms: List[int],
                                        coupling_model: str = "exponential") -> np.ndarray:
        """
        Extract time evolution of electronic coupling matrix.
        
        Args:
            trajectory: Trajectory data from run_trajectory()
            quantum_atoms: List of atom indices for quantum subsystem
            coupling_model: Model for distance-dependent coupling ("exponential", "power_law")
            
        Returns:
            Array of shape (n_frames, n_quantum, n_quantum) with coupling matrices
        """
        try:
            positions = trajectory['positions']
            n_frames, n_atoms, _ = positions.shape
            n_quantum = len(quantum_atoms)
            
            # Validate inputs
            if max(quantum_atoms) >= n_atoms:
                raise ValueError(f"Quantum atom index exceeds system size")
            
            coupling_matrices = np.zeros((n_frames, n_quantum, n_quantum))
            
            # Extract quantum atom positions
            quantum_positions = positions[:, quantum_atoms, :]
            
            # Coupling parameters (these would normally come from quantum chemistry calculations)
            if coupling_model == "exponential":
                j0 = 100.0  # Base coupling in cm^-1
                beta = 1.0  # Decay parameter in nm^-1
                r0 = 0.5   # Reference distance in nm
            elif coupling_model == "power_law":
                j0 = 100.0  # Base coupling in cm^-1
                alpha = 3.0  # Power law exponent
                r0 = 0.5   # Reference distance in nm
            else:
                raise ValueError(f"Unknown coupling model: {coupling_model}")
            
            for frame in range(n_frames):
                for i in range(n_quantum):
                    for j in range(i + 1, n_quantum):
                        # Calculate distance
                        r_ij = np.linalg.norm(quantum_positions[frame, i] - quantum_positions[frame, j])
                        
                        # Calculate coupling based on model
                        if coupling_model == "exponential":
                            coupling = j0 * np.exp(-beta * (r_ij - r0))
                        elif coupling_model == "power_law":
                            coupling = j0 * (r0 / r_ij) ** alpha
                        
                        # Ensure coupling is positive and reasonable
                        coupling = max(0.0, min(coupling, 1000.0))  # Cap at 1000 cm^-1
                        
                        # Fill symmetric matrix
                        coupling_matrices[frame, i, j] = coupling
                        coupling_matrices[frame, j, i] = coupling
            
            logging.info(f"Extracted coupling matrix evolution: {n_frames} frames, {n_quantum}x{n_quantum} matrix")
            
            return coupling_matrices
            
        except Exception as e:
            logging.error(f"Coupling matrix extraction failed: {e}")
            raise RuntimeError(f"Coupling matrix extraction failed: {e}")
    
    def calculate_spectral_density(self, fluctuations: np.ndarray, 
                                  time_step: float, temperature: float = 300.0) -> SpectralDensity:
        """
        Calculate spectral density from parameter fluctuations using FFT.
        
        Args:
            fluctuations: Time series of parameter fluctuations
            time_step: Time step between data points in ps
            temperature: Temperature in Kelvin for quantum corrections
            
        Returns:
            SpectralDensity object containing frequency-dependent spectral density
            
        Raises:
            ValueError: If fluctuations array is invalid
        """
        if fluctuations.size == 0:
            raise ValueError("Fluctuations array cannot be empty")
        
        try:
            # Ensure 1D array
            if fluctuations.ndim > 1:
                fluctuations = fluctuations.flatten()
            
            n_points = len(fluctuations)
            
            # Remove mean (center the data)
            fluctuations_centered = fluctuations - np.mean(fluctuations)
            
            # Apply window function to reduce spectral leakage
            window = np.hanning(n_points)
            fluctuations_windowed = fluctuations_centered * window
            
            # Calculate power spectral density using FFT
            fft_result = np.fft.fft(fluctuations_windowed)
            power_spectrum = np.abs(fft_result) ** 2
            
            # Calculate frequencies
            dt = time_step * 1e-12  # Convert ps to s
            frequencies_hz = np.fft.fftfreq(n_points, dt)
            
            # Take only positive frequencies
            positive_freq_mask = frequencies_hz > 0  # Exclude zero frequency
            frequencies_hz = frequencies_hz[positive_freq_mask]
            power_spectrum = power_spectrum[positive_freq_mask]
            
            # Convert to cm^-1 (common in spectroscopy)
            c_light = 2.998e10  # cm/s
            frequencies_cm = frequencies_hz / (c_light * 100)  # Convert Hz to cm^-1
            
            # Normalize spectral density
            # Factor of 2 to account for positive frequencies only
            spectral_density = 2 * power_spectrum / (n_points * np.sum(window**2))
            
            # Apply quantum correction factor for finite temperature
            # S(ω) -> S(ω) * [1 + n(ω)] where n(ω) is Bose-Einstein distribution
            kb = 1.381e-23  # Boltzmann constant in J/K
            hbar = 1.055e-34  # Reduced Planck constant in J⋅s
            
            # Convert cm^-1 to angular frequency (rad/s)
            omega = frequencies_cm * c_light * 100 * 2 * np.pi
            
            # Bose-Einstein distribution
            with np.errstate(over='ignore', invalid='ignore'):
                bose_factor = 1.0 / (np.exp(hbar * omega / (kb * temperature)) - 1.0)
                # Handle numerical issues at high frequencies
                bose_factor = np.where(np.isfinite(bose_factor), bose_factor, 0.0)
            
            # Apply quantum correction
            quantum_corrected_density = spectral_density * (1.0 + bose_factor)
            
            spectral_density_obj = SpectralDensity(
                frequencies=frequencies_cm,
                spectral_values=quantum_corrected_density,
                temperature=temperature,
                spectral_type="md_derived"
            )
            
            logging.info(f"Calculated spectral density with {len(frequencies_cm)} frequency points")
            logging.info(f"Frequency range: {frequencies_cm[0]:.2e} - {frequencies_cm[-1]:.2e} cm^-1")
            logging.info(f"Applied quantum correction at T = {temperature} K")
            
            return spectral_density_obj
            
        except Exception as e:
            logging.error(f"Spectral density calculation failed: {e}")
            raise RuntimeError(f"Spectral density calculation failed: {e}")
    
    def extract_time_dependent_hamiltonian_parameters(self, trajectory: Dict[str, np.ndarray],
                                                    quantum_atoms: List[int],
                                                    parameter_types: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract time-dependent Hamiltonian parameters from MD trajectory.
        
        Args:
            trajectory: Trajectory data from run_trajectory()
            quantum_atoms: List of atom indices for quantum subsystem
            parameter_types: List of parameter types to extract ['site_energies', 'couplings', 'reorganization']
            
        Returns:
            Dictionary containing time-dependent Hamiltonian parameters
        """
        if parameter_types is None:
            parameter_types = ['site_energies', 'couplings', 'reorganization']
        
        try:
            positions = trajectory['positions']
            n_frames, n_atoms, _ = positions.shape
            n_quantum = len(quantum_atoms)
            
            # Validate inputs
            if max(quantum_atoms) >= n_atoms:
                raise ValueError("Quantum atom index exceeds system size")
            
            parameters = {}
            quantum_positions = positions[:, quantum_atoms, :]
            
            # Extract site energies (based on local environment)
            if 'site_energies' in parameter_types:
                site_energies = self._calculate_site_energy_fluctuations(
                    quantum_positions, positions, quantum_atoms
                )
                parameters['site_energies'] = site_energies
            
            # Extract electronic couplings
            if 'couplings' in parameter_types:
                couplings = self._calculate_coupling_fluctuations(quantum_positions)
                parameters['couplings'] = couplings
            
            # Extract reorganization energies
            if 'reorganization' in parameter_types:
                reorganization = self._calculate_reorganization_energy_fluctuations(
                    quantum_positions, trajectory.get('energies', np.zeros(n_frames))
                )
                parameters['reorganization'] = reorganization
            
            logging.info(f"Extracted Hamiltonian parameters: {list(parameters.keys())}")
            
            return parameters
            
        except Exception as e:
            logging.error(f"Hamiltonian parameter extraction failed: {e}")
            raise RuntimeError(f"Hamiltonian parameter extraction failed: {e}")
    
    def _calculate_site_energy_fluctuations(self, quantum_positions: np.ndarray,
                                          all_positions: np.ndarray,
                                          quantum_atoms: List[int]) -> np.ndarray:
        """Calculate site energy fluctuations based on electrostatic environment."""
        try:
            n_frames, n_quantum, _ = quantum_positions.shape
            n_atoms = all_positions.shape[1]
            
            site_energies = np.zeros((n_frames, n_quantum))
            
            # Simple electrostatic model for site energy shifts
            # E_site = Σ q_j / r_ij (sum over environment charges)
            
            for frame in range(n_frames):
                for i, quantum_idx in enumerate(quantum_atoms):
                    energy_shift = 0.0
                    
                    # Sum contributions from all other atoms
                    for j in range(n_atoms):
                        if j in quantum_atoms:
                            continue  # Skip other quantum atoms
                        
                        # Calculate distance
                        r_vec = all_positions[frame, j] - quantum_positions[frame, i]
                        r = np.linalg.norm(r_vec)
                        
                        if r > 0.01:  # Avoid singularities (1 Å cutoff)
                            # Simple charge model (would be improved with actual charges)
                            q_env = 0.1 if j % 2 == 0 else -0.1  # Alternating charges
                            energy_shift += q_env / r  # In atomic units
                    
                    site_energies[frame, i] = energy_shift
            
            # Convert to more reasonable units (cm^-1)
            # 1 Hartree = 219474.6 cm^-1, but we scale down for realistic values
            site_energies *= 1000.0  # Scale to get reasonable fluctuations
            
            return site_energies
            
        except Exception as e:
            logging.warning(f"Site energy calculation failed: {e}")
            return np.zeros((quantum_positions.shape[0], quantum_positions.shape[1]))
    
    def _calculate_coupling_fluctuations(self, quantum_positions: np.ndarray) -> np.ndarray:
        """Calculate electronic coupling fluctuations based on distance and orientation."""
        try:
            n_frames, n_quantum, _ = quantum_positions.shape
            couplings = np.zeros((n_frames, n_quantum, n_quantum))
            
            # Parameters for coupling model
            j0 = 100.0  # Base coupling in cm^-1
            beta = 1.5  # Exponential decay parameter in nm^-1
            r0 = 0.5   # Reference distance in nm
            
            for frame in range(n_frames):
                for i in range(n_quantum):
                    for j in range(i + 1, n_quantum):
                        # Calculate distance
                        r_vec = quantum_positions[frame, i] - quantum_positions[frame, j]
                        r = np.linalg.norm(r_vec)
                        
                        # Distance-dependent coupling with exponential decay
                        coupling = j0 * np.exp(-beta * (r - r0))
                        
                        # Add orientation dependence (simplified)
                        # In reality, this would depend on transition dipole orientations
                        if n_quantum > 2:
                            # Use relative positions to estimate orientation effects
                            orientation_factor = 1.0 + 0.2 * np.cos(frame * 0.1)  # Simple oscillation
                            coupling *= orientation_factor
                        
                        # Ensure coupling is positive and reasonable
                        coupling = max(0.0, min(coupling, 500.0))  # Cap at 500 cm^-1
                        
                        couplings[frame, i, j] = coupling
                        couplings[frame, j, i] = coupling  # Symmetric
            
            return couplings
            
        except Exception as e:
            logging.warning(f"Coupling fluctuation calculation failed: {e}")
            return np.zeros((quantum_positions.shape[0], quantum_positions.shape[1], quantum_positions.shape[1]))
    
    def _calculate_reorganization_energy_fluctuations(self, quantum_positions: np.ndarray,
                                                    energies: np.ndarray) -> np.ndarray:
        """Calculate reorganization energy fluctuations."""
        try:
            n_frames, n_quantum, _ = quantum_positions.shape
            reorganization = np.zeros((n_frames, n_quantum))
            
            # Calculate reorganization energy based on position fluctuations
            # λ = (1/2) * k * <Δr²> where k is effective force constant
            
            k_eff = 1000.0  # Effective force constant in cm^-1/nm²
            
            # Calculate mean positions
            mean_positions = np.mean(quantum_positions, axis=0)
            
            for frame in range(n_frames):
                for i in range(n_quantum):
                    # Displacement from mean position
                    displacement = quantum_positions[frame, i] - mean_positions[i]
                    displacement_sq = np.sum(displacement ** 2)
                    
                    # Reorganization energy
                    lambda_reorg = 0.5 * k_eff * displacement_sq
                    
                    reorganization[frame, i] = lambda_reorg
            
            return reorganization
            
        except Exception as e:
            logging.warning(f"Reorganization energy calculation failed: {e}")
            return np.zeros((quantum_positions.shape[0], quantum_positions.shape[1]))
    
    def calculate_parameter_cross_correlations(self, parameters: Dict[str, np.ndarray],
                                             time_step: float) -> Dict[str, np.ndarray]:
        """
        Calculate cross-correlations between different parameter types.
        
        Args:
            parameters: Dictionary of parameter time series
            time_step: Time step in ps
            
        Returns:
            Dictionary containing cross-correlation functions
        """
        try:
            correlations = {}
            param_names = list(parameters.keys())
            
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names[i:], i):
                    # Get time series (flatten if multidimensional)
                    data1 = parameters[param1]
                    data2 = parameters[param2]
                    
                    if data1.ndim > 1:
                        data1 = data1.flatten()
                    if data2.ndim > 1:
                        data2 = data2.flatten()
                    
                    # Ensure same length
                    min_len = min(len(data1), len(data2))
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]
                    
                    # Calculate cross-correlation
                    cross_corr = self._calculate_cross_correlation(data1, data2)
                    
                    correlation_key = f"{param1}_{param2}" if i != j else f"{param1}_auto"
                    correlations[correlation_key] = cross_corr
            
            logging.info(f"Calculated cross-correlations for {len(correlations)} parameter pairs")
            
            return correlations
            
        except Exception as e:
            logging.error(f"Cross-correlation calculation failed: {e}")
            raise RuntimeError(f"Cross-correlation calculation failed: {e}")
    
    def _calculate_cross_correlation(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        """Calculate normalized cross-correlation between two time series."""
        try:
            # Center the data
            data1_centered = data1 - np.mean(data1)
            data2_centered = data2 - np.mean(data2)
            
            # Calculate cross-correlation using FFT
            n = len(data1_centered)
            
            # Pad with zeros
            padded1 = np.zeros(2 * n)
            padded2 = np.zeros(2 * n)
            padded1[:n] = data1_centered
            padded2[:n] = data2_centered
            
            # FFT-based cross-correlation
            fft1 = np.fft.fft(padded1)
            fft2 = np.fft.fft(padded2)
            cross_corr_fft = np.fft.ifft(fft1 * np.conj(fft2))
            
            # Take real part and normalize
            cross_corr = np.real(cross_corr_fft[:n])
            
            # Normalize by the product of standard deviations
            norm_factor = np.std(data1) * np.std(data2) * n
            if norm_factor > 0:
                cross_corr = cross_corr / norm_factor
            
            return cross_corr
            
        except Exception as e:
            logging.warning(f"Cross-correlation calculation failed: {e}")
            return np.array([0.0])
    
    def setup_environment(self, system: MolecularSystem, 
                         solvent_model: str, ionic_strength: float) -> bool:
        """
        Set up solvation environment for the molecular system.
        
        Args:
            system: MolecularSystem to solvate
            solvent_model: Water model to use (e.g., 'tip3p', 'tip4pew')
            ionic_strength: Ionic strength in M for salt addition
            
        Returns:
            True if environment setup successful
            
        Raises:
            ValueError: If solvent model not supported
            RuntimeError: If solvation fails
        """
        if solvent_model not in self.available_water_models:
            raise ValueError(f"Unsupported solvent model: {solvent_model}. "
                           f"Available: {list(self.available_water_models.keys())}")
        
        if self.system is None or self.topology is None:
            raise RuntimeError("System must be initialized before setting up environment")
        
        try:
            # Add water model to force field
            water_file = self.available_water_models[solvent_model]
            self.force_field.loadFile(water_file)
            
            # Create modeller for solvation
            modeller = app.Modeller(self.topology, self.positions)
            
            # Add solvent box (10 Å padding)
            padding = 1.0 * unit.nanometer
            modeller.addSolvent(
                self.force_field,
                model=solvent_model,
                padding=padding,
                ionicStrength=ionic_strength * unit.molar
            )
            
            # Update topology and positions
            self.topology = modeller.topology
            self.positions = modeller.positions
            
            # Recreate system with proper environment
            self.system = self.force_field.createSystem(
                self.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True,
                ewaldErrorTolerance=0.0005
            )
            
            logging.info(f"Environment setup complete with {solvent_model} water model")
            logging.info(f"Ionic strength: {ionic_strength} M")
            logging.info(f"Total atoms after solvation: {self.topology.getNumAtoms()}")
            
            return True
            
        except Exception as e:
            logging.error(f"Environment setup failed: {e}")
            raise RuntimeError(f"Environment setup failed: {e}")
    
    def minimize_energy(self, max_iterations: int = 1000) -> float:
        """
        Perform energy minimization and return final energy.
        
        Args:
            max_iterations: Maximum number of minimization steps
            
        Returns:
            Final potential energy in kJ/mol
            
        Raises:
            RuntimeError: If system not initialized or minimization fails
        """
        if self.system is None or self.topology is None or self.positions is None:
            raise RuntimeError("System must be initialized before energy minimization")
        
        try:
            # Create integrator (not used for minimization but required)
            integrator = openmm.LangevinIntegrator(
                300 * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtosecond
            )
            
            # Create simulation
            simulation = app.Simulation(
                self.topology,
                self.system,
                integrator,
                self.platform
            )
            
            # Set positions
            simulation.context.setPositions(self.positions)
            
            # Get initial energy
            initial_state = simulation.context.getState(getEnergy=True)
            initial_energy = initial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            
            logging.info(f"Initial potential energy: {initial_energy:.2f} kJ/mol")
            
            # Perform minimization
            simulation.minimizeEnergy(maxIterations=max_iterations)
            
            # Get final energy and positions
            final_state = simulation.context.getState(getEnergy=True, getPositions=True)
            final_energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            
            # Update positions
            self.positions = final_state.getPositions()
            
            logging.info(f"Final potential energy: {final_energy:.2f} kJ/mol")
            logging.info(f"Energy change: {final_energy - initial_energy:.2f} kJ/mol")
            
            return final_energy
            
        except Exception as e:
            logging.error(f"Energy minimization failed: {e}")
            raise RuntimeError(f"Energy minimization failed: {e}")    
 
   # Utility methods for creating test systems
    
    def create_water_box(self, box_size: float = 2.0, water_model: str = "tip3p") -> MolecularSystem:
        """
        Create a simple water box for testing.
        
        Args:
            box_size: Box size in nm
            water_model: Water model to use
            
        Returns:
            MolecularSystem containing water box
        """
        try:
            # Create empty topology and modeller
            topology = app.Topology()
            positions = []
            
            # Create modeller
            modeller = app.Modeller(topology, positions)
            
            # Set up force field with water model
            if water_model not in self.available_water_models:
                raise ValueError(f"Unsupported water model: {water_model}")
            
            water_file = self.available_water_models[water_model]
            self.force_field = app.ForceField(water_file)
            
            # Add solvent box
            box_vectors = np.eye(3) * box_size
            modeller.addSolvent(
                self.force_field,
                model=water_model,
                boxSize=box_vectors * unit.nanometer
            )
            
            # Update system
            self.topology = modeller.topology
            self.positions = modeller.positions
            
            self.system = self.force_field.createSystem(
                self.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True
            )
            
            # Convert to MolecularSystem
            molecular_system = self._convert_topology_to_molecular_system()
            
            logging.info(f"Created water box: {box_size} nm, {molecular_system.atoms.__len__()} atoms")
            
            return molecular_system
            
        except Exception as e:
            logging.error(f"Water box creation failed: {e}")
            raise RuntimeError(f"Water box creation failed: {e}")
    
    def create_simple_protein_system(self, sequence: str = "ALA-GLY-ALA") -> MolecularSystem:
        """
        Create a simple protein system for testing.
        
        Args:
            sequence: Amino acid sequence (3-letter codes separated by dashes)
            
        Returns:
            MolecularSystem containing simple protein
        """
        try:
            # This is a simplified implementation
            # In practice, would use more sophisticated protein building tools
            
            # For now, create a minimal system with just a few atoms
            # representing a simple peptide
            atoms = []
            bonds = []
            residues = {}
            
            # Parse sequence
            residue_names = sequence.split('-')
            
            atom_id = 0
            for res_id, res_name in enumerate(residue_names):
                # Add backbone atoms (simplified)
                # N
                atoms.append(Atom(
                    element='N',
                    position=np.array([res_id * 0.4, 0.0, 0.0]),
                    charge=-0.4,
                    mass=14.007,
                    atom_id=atom_id,
                    residue_id=res_id,
                    residue_name=res_name
                ))
                n_idx = atom_id
                atom_id += 1
                
                # CA
                atoms.append(Atom(
                    element='C',
                    position=np.array([res_id * 0.4 + 0.1, 0.1, 0.0]),
                    charge=0.0,
                    mass=12.011,
                    atom_id=atom_id,
                    residue_id=res_id,
                    residue_name=res_name
                ))
                ca_idx = atom_id
                atom_id += 1
                
                # C
                atoms.append(Atom(
                    element='C',
                    position=np.array([res_id * 0.4 + 0.2, 0.0, 0.0]),
                    charge=0.6,
                    mass=12.011,
                    atom_id=atom_id,
                    residue_id=res_id,
                    residue_name=res_name
                ))
                c_idx = atom_id
                atom_id += 1
                
                # O
                atoms.append(Atom(
                    element='O',
                    position=np.array([res_id * 0.4 + 0.25, -0.1, 0.0]),
                    charge=-0.6,
                    mass=15.999,
                    atom_id=atom_id,
                    residue_id=res_id,
                    residue_name=res_name
                ))
                o_idx = atom_id
                atom_id += 1
                
                # Add bonds within residue
                bonds.append((n_idx, ca_idx))
                bonds.append((ca_idx, c_idx))
                bonds.append((c_idx, o_idx))
                
                # Add peptide bond to previous residue
                if res_id > 0:
                    prev_c_idx = c_idx - 4  # C atom of previous residue
                    bonds.append((prev_c_idx, n_idx))
                
                # Track residues
                residues[res_id] = res_name
            
            return MolecularSystem(
                atoms=atoms,
                bonds=bonds,
                residues=residues,
                system_name=f"peptide_{sequence}",
                total_charge=0.0
            )
            
        except Exception as e:
            logging.error(f"Simple protein system creation failed: {e}")
            raise RuntimeError(f"Simple protein system creation failed: {e}")
    
    def create_test_chromophore_system(self, n_chromophores: int = 2) -> MolecularSystem:
        """
        Create a test system with multiple chromophores for quantum coupling studies.
        
        Args:
            n_chromophores: Number of chromophores to create
            
        Returns:
            MolecularSystem containing chromophore system
        """
        try:
            atoms = []
            bonds = []
            residues = {}
            
            atom_id = 0
            
            for chrom_id in range(n_chromophores):
                # Create a simple chromophore (e.g., chlorophyll-like)
                # Central Mg atom
                atoms.append(Atom(
                    element='Mg',
                    position=np.array([chrom_id * 2.0, 0.0, 0.0]),
                    charge=2.0,
                    mass=24.305,
                    atom_id=atom_id,
                    residue_id=chrom_id,
                    residue_name='CHL'
                ))
                mg_idx = atom_id
                atom_id += 1
                
                # Surrounding N atoms (porphyrin-like ring)
                n_positions = [
                    np.array([chrom_id * 2.0 + 0.2, 0.2, 0.0]),
                    np.array([chrom_id * 2.0 - 0.2, 0.2, 0.0]),
                    np.array([chrom_id * 2.0 + 0.2, -0.2, 0.0]),
                    np.array([chrom_id * 2.0 - 0.2, -0.2, 0.0])
                ]
                
                n_indices = []
                for pos in n_positions:
                    atoms.append(Atom(
                        element='N',
                        position=pos,
                        charge=-0.5,
                        mass=14.007,
                        atom_id=atom_id,
                        residue_id=chrom_id,
                        residue_name='CHL'
                    ))
                    n_indices.append(atom_id)
                    bonds.append((mg_idx, atom_id))  # Mg-N bonds
                    atom_id += 1
                
                # Add some C atoms to complete the ring structure
                for i in range(4):
                    atoms.append(Atom(
                        element='C',
                        position=np.array([chrom_id * 2.0 + 0.3 * np.cos(i * np.pi/2), 
                                         0.3 * np.sin(i * np.pi/2), 0.0]),
                        charge=0.0,
                        mass=12.011,
                        atom_id=atom_id,
                        residue_id=chrom_id,
                        residue_name='CHL'
                    ))
                    # Bond to adjacent N atoms
                    bonds.append((n_indices[i], atom_id))
                    if i < 3:
                        bonds.append((atom_id, n_indices[i+1]))
                    else:
                        bonds.append((atom_id, n_indices[0]))  # Close the ring
                    atom_id += 1
                
                residues[chrom_id] = 'CHL'
            
            return MolecularSystem(
                atoms=atoms,
                bonds=bonds,
                residues=residues,
                system_name=f"chromophore_system_{n_chromophores}",
                total_charge=0.0
            )
            
        except Exception as e:
            logging.error(f"Chromophore system creation failed: {e}")
            raise RuntimeError(f"Chromophore system creation failed: {e}")
            
            logging.info(f"Created simple protein system: {sequence}, {len(atoms)} atoms")
            
            return molecular_system
            
        except Exception as e:
            logging.error(f"Simple protein creation failed: {e}")
            raise RuntimeError(f"Simple protein creation failed: {e}")
    
    def _convert_topology_to_molecular_system(self) -> MolecularSystem:
        """Convert current OpenMM topology to MolecularSystem format."""
        atoms = []
        bonds = []
        residues = {}
        
        # Extract atoms
        for atom_idx, atom in enumerate(self.topology.atoms()):
            position = self.positions[atom_idx].value_in_unit(unit.nanometer)
            
            atom_obj = Atom(
                element=atom.element.symbol,
                position=np.array(position),
                charge=0.0,
                mass=atom.element.mass.value_in_unit(unit.dalton),
                atom_id=atom_idx,
                residue_id=atom.residue.id,
                residue_name=atom.residue.name
            )
            atoms.append(atom_obj)
            
            # Track residues
            if atom.residue.id not in residues:
                residues[atom.residue.id] = atom.residue.name
        
        # Extract bonds
        for bond in self.topology.bonds():
            atom1_idx = bond[0].index
            atom2_idx = bond[1].index
            bonds.append((atom1_idx, atom2_idx))
        
        return MolecularSystem(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            system_name="openmm_system",
            total_charge=0.0
        )
    
    def validate_parameter_extraction_accuracy(self, parameters: Dict[str, np.ndarray],
                                             trajectory: Dict[str, np.ndarray]) -> ValidationResult:
        """
        Validate the accuracy of extracted quantum parameters.
        
        Args:
            parameters: Extracted quantum parameters
            trajectory: Original trajectory data
            
        Returns:
            ValidationResult with accuracy assessment
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Check parameter consistency
            n_frames = trajectory['positions'].shape[0]
            
            for param_name, param_data in parameters.items():
                if param_data.size == 0:
                    result.add_warning(f"Parameter '{param_name}' is empty")
                    continue
                
                # Check temporal consistency
                if param_data.ndim >= 1 and param_data.shape[0] != n_frames:
                    result.add_error(f"Parameter '{param_name}' has inconsistent time dimension")
                
                # Check for unphysical values
                if param_name == 'distances':
                    if np.any(param_data < 0):
                        result.add_error(f"Negative distances found in '{param_name}'")
                    if np.any(param_data > 10.0):  # 10 nm is very large for quantum systems
                        result.add_warning(f"Very large distances (>10 nm) found in '{param_name}'")
                
                elif param_name == 'coupling_fluctuations':
                    if np.any(param_data < 0):
                        result.add_error(f"Negative couplings found in '{param_name}'")
                    if np.any(param_data > 1000):  # 1000 cm^-1 is very strong coupling
                        result.add_warning(f"Very strong couplings (>1000 cm^-1) found in '{param_name}'")
                
                elif param_name == 'angles':
                    if np.any(param_data < 0) or np.any(param_data > np.pi):
                        result.add_error(f"Unphysical angles found in '{param_name}'")
                
                # Check for NaN or infinite values
                if not np.all(np.isfinite(param_data)):
                    result.add_error(f"NaN or infinite values found in '{param_name}'")
                
                # Check statistical properties
                if param_data.size > 1:
                    std_dev = np.std(param_data)
                    mean_val = np.mean(param_data)
                    
                    # Check for unreasonable fluctuations
                    if std_dev > 10 * abs(mean_val) and abs(mean_val) > 1e-10:
                        result.add_warning(f"Very large fluctuations in '{param_name}' (std/mean > 10)")
                    
                    # Check for constant values (might indicate extraction failure)
                    if std_dev < 1e-10:
                        result.add_warning(f"Parameter '{param_name}' appears constant (no fluctuations)")
            
            # Cross-parameter consistency checks
            if 'distances' in parameters and 'coupling_fluctuations' in parameters:
                # Check that coupling decreases with distance (generally expected)
                distances = parameters['distances']
                couplings = parameters['coupling_fluctuations']
                
                if distances.shape == couplings.shape:
                    # Sample a few frames for correlation check
                    sample_frames = min(10, distances.shape[0])
                    for frame in range(0, distances.shape[0], distances.shape[0] // sample_frames):
                        dist_matrix = distances[frame]
                        coup_matrix = couplings[frame]
                        
                        # Check off-diagonal elements
                        for i in range(dist_matrix.shape[0]):
                            for j in range(i + 1, dist_matrix.shape[1]):
                                if dist_matrix[i, j] > 0 and coup_matrix[i, j] > 0:
                                    # Very rough check: coupling should generally decrease with distance
                                    if dist_matrix[i, j] > 2.0 and coup_matrix[i, j] > 100:
                                        result.add_warning(f"High coupling ({coup_matrix[i, j]:.1f}) at large distance ({dist_matrix[i, j]:.2f} nm)")
            
            logging.info(f"Parameter extraction validation: {'PASSED' if result.is_valid else 'FAILED'}")
            if result.errors:
                logging.error(f"Validation errors: {result.errors}")
            if result.warnings:
                logging.warning(f"Validation warnings: {result.warnings}")
            
        except Exception as e:
            result.add_error(f"Parameter validation failed with exception: {e}")
        
        return result
    
    def validate_system(self) -> ValidationResult:
        """
        Validate the current MD system setup.
        
        Returns:
            ValidationResult indicating system validity
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Check if system is initialized
            if self.system is None:
                result.add_error("OpenMM system not initialized")
            
            if self.topology is None:
                result.add_error("Topology not loaded")
            
            if self.positions is None:
                result.add_error("Positions not set")
            
            # Check system consistency
            if self.system is not None and self.topology is not None:
                system_particles = self.system.getNumParticles()
                topology_atoms = self.topology.getNumAtoms()
                
                if system_particles != topology_atoms:
                    result.add_error(f"Particle count mismatch: system={system_particles}, topology={topology_atoms}")
            
            # Check for reasonable system size
            if self.topology is not None:
                n_atoms = self.topology.getNumAtoms()
                if n_atoms == 0:
                    result.add_error("System contains no atoms")
                elif n_atoms > 100000:
                    result.add_warning(f"Large system ({n_atoms} atoms) may be computationally expensive")
            
            # Check platform availability
            if self.platform is None:
                result.add_error("OpenMM platform not available")
            
            logging.info(f"System validation: {'PASSED' if result.is_valid else 'FAILED'}")
            if result.errors:
                logging.error(f"Validation errors: {result.errors}")
            if result.warnings:
                logging.warning(f"Validation warnings: {result.warnings}")
            
        except Exception as e:
            result.add_error(f"Validation failed with exception: {e}")
        
        return result