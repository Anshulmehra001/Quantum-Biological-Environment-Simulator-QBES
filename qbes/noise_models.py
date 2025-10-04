"""
Environmental noise models for biological systems.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
import numpy as np

from .core.interfaces import NoiseModelInterface
from .core.data_models import QuantumSubsystem, LindbladOperator, ValidationResult


class NoiseModel(NoiseModelInterface, ABC):
    """Abstract base class for environmental noise models."""
    
    def __init__(self, model_type: str, **parameters):
        """Initialize noise model with specified type and parameters."""
        self.model_type = model_type
        self.parameters = parameters
        self._validate_base_parameters()
    
    def _validate_base_parameters(self):
        """Validate base parameters common to all noise models."""
        if 'coupling_strength' in self.parameters:
            if self.parameters['coupling_strength'] < 0:
                raise ValueError("Coupling strength must be non-negative")
        
        if 'cutoff_frequency' in self.parameters:
            if self.parameters['cutoff_frequency'] <= 0:
                raise ValueError("Cutoff frequency must be positive")
    
    @abstractmethod
    def get_spectral_density(self, frequency: float, temperature: float) -> float:
        """Get spectral density value at given frequency and temperature.
        
        Args:
            frequency: Frequency in rad/s
            temperature: Temperature in Kelvin
            
        Returns:
            Spectral density value
        """
        pass
    
    def calculate_decoherence_rates(self, system: QuantumSubsystem, 
                                   temperature: float) -> Dict[str, float]:
        """Calculate temperature-dependent decoherence rates.
        
        Uses the spectral density and quantum system properties to calculate
        decoherence rates for different processes (dephasing, relaxation).
        
        Args:
            system: Quantum subsystem
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary of decoherence rates for different processes
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Boltzmann constant in eV/K
        k_B = 8.617333e-5
        
        # Calculate characteristic frequencies from system
        n_states = len(system.basis_states)
        if n_states < 2:
            return {"dephasing": 0.0, "relaxation": 0.0}
        
        # Estimate transition frequencies from coupling matrix
        eigenvals = np.linalg.eigvals(system.coupling_matrix)
        transition_freqs = np.abs(np.diff(np.sort(eigenvals)))
        
        if len(transition_freqs) == 0:
            return {"dephasing": 0.0, "relaxation": 0.0}
        
        # Calculate rates for different processes
        rates = {}
        
        # Dephasing rate (pure decoherence)
        avg_freq = np.mean(transition_freqs)
        spectral_val = self.get_spectral_density(avg_freq, temperature)
        
        # Temperature-dependent factor using more realistic model
        # For biological systems, use a combination of quantum and classical contributions
        if avg_freq > 0:
            # Quantum correction factor
            x = avg_freq / (k_B * temperature)
            if x > 0.1:  # Quantum regime
                # Use proper Bose-Einstein distribution
                thermal_factor = 1.0 / (np.exp(x) - 1.0) + 1.0
            else:  # Classical regime
                thermal_factor = k_B * temperature / avg_freq
        else:
            thermal_factor = k_B * temperature
        
        # Ensure thermal factor is always positive
        thermal_factor = max(thermal_factor, 0.01)
        
        rates["dephasing"] = spectral_val * thermal_factor
        
        # Relaxation rate (energy dissipation)
        # Typically smaller than dephasing in biological systems
        rates["relaxation"] = 0.1 * rates["dephasing"]
        
        return rates
    
    def generate_lindblad_operators(self, system: QuantumSubsystem, 
                                   temperature: float) -> List[LindbladOperator]:
        """Generate Lindblad operators from decoherence rates.
        
        Args:
            system: Quantum subsystem
            temperature: Temperature in Kelvin
            
        Returns:
            List of Lindblad operators
        """
        rates = self.calculate_decoherence_rates(system, temperature)
        operators = []
        
        n_states = len(system.basis_states)
        if n_states < 2:
            return operators
        
        # Create dephasing operators (diagonal in energy basis)
        if rates.get("dephasing", 0) > 0:
            for i in range(n_states):
                op_matrix = np.zeros((n_states, n_states), dtype=complex)
                op_matrix[i, i] = 1.0
                
                operator = LindbladOperator(
                    operator=op_matrix,
                    coupling_strength=np.sqrt(rates["dephasing"]),
                    operator_type="dephasing"
                )
                operators.append(operator)
        
        # Create relaxation operators (off-diagonal)
        if rates.get("relaxation", 0) > 0:
            for i in range(n_states - 1):
                # Lowering operator
                op_matrix = np.zeros((n_states, n_states), dtype=complex)
                op_matrix[i, i + 1] = 1.0
                
                operator = LindbladOperator(
                    operator=op_matrix,
                    coupling_strength=np.sqrt(rates["relaxation"]),
                    operator_type="relaxation"
                )
                operators.append(operator)
        
        return operators
    
    def validate_noise_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate noise model parameters."""
        result = ValidationResult(is_valid=True)
        
        # Check required parameters
        required_params = self._get_required_parameters()
        for param in required_params:
            if param not in parameters:
                result.add_error(f"Missing required parameter: {param}")
        
        # Check parameter ranges
        for param, value in parameters.items():
            validation_error = self._validate_parameter(param, value)
            if validation_error:
                result.add_error(validation_error)
        
        return result
    
    @abstractmethod
    def _get_required_parameters(self) -> List[str]:
        """Get list of required parameters for this noise model."""
        pass
    
    def _validate_parameter(self, param: str, value: Any) -> str:
        """Validate individual parameter. Returns error message if invalid."""
        if param == "coupling_strength" and value < 0:
            return "Coupling strength must be non-negative"
        elif param == "cutoff_frequency" and value <= 0:
            return "Cutoff frequency must be positive"
        elif param == "temperature" and value <= 0:
            return "Temperature must be positive"
        return ""


class OhmicNoiseModel(NoiseModel):
    """Ohmic spectral density noise model for protein environments.
    
    Spectral density: J(ω) = η * ω * exp(-ω/ωc)
    where η is coupling strength and ωc is cutoff frequency.
    """
    
    def __init__(self, coupling_strength: float = 0.1, cutoff_frequency: float = 1.0):
        """Initialize Ohmic noise model.
        
        Args:
            coupling_strength: Dimensionless coupling strength η
            cutoff_frequency: Cutoff frequency ωc in rad/s
        """
        super().__init__("ohmic", 
                        coupling_strength=coupling_strength,
                        cutoff_frequency=cutoff_frequency)
    
    def get_spectral_density(self, frequency: float, temperature: float) -> float:
        """Calculate Ohmic spectral density.
        
        Args:
            frequency: Frequency in rad/s
            temperature: Temperature in Kelvin (used for validation)
            
        Returns:
            Spectral density value
        """
        if frequency < 0:
            raise ValueError("Frequency must be non-negative")
        
        eta = self.parameters["coupling_strength"]
        omega_c = self.parameters["cutoff_frequency"]
        
        return eta * frequency * np.exp(-frequency / omega_c)
    
    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for Ohmic model."""
        return ["coupling_strength", "cutoff_frequency"]


class SuperOhmicNoiseModel(NoiseModel):
    """Super-Ohmic spectral density for structured environments.
    
    Spectral density: J(ω) = η * ω^3 * exp(-ω/ωc)
    """
    
    def __init__(self, coupling_strength: float = 0.05, cutoff_frequency: float = 1.0):
        """Initialize Super-Ohmic noise model."""
        super().__init__("super_ohmic",
                        coupling_strength=coupling_strength,
                        cutoff_frequency=cutoff_frequency)
    
    def get_spectral_density(self, frequency: float, temperature: float) -> float:
        """Calculate Super-Ohmic spectral density."""
        if frequency < 0:
            raise ValueError("Frequency must be non-negative")
        
        eta = self.parameters["coupling_strength"]
        omega_c = self.parameters["cutoff_frequency"]
        
        return eta * (frequency ** 3) * np.exp(-frequency / omega_c)
    
    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for Super-Ohmic model."""
        return ["coupling_strength", "cutoff_frequency"]


class SubOhmicNoiseModel(NoiseModel):
    """Sub-Ohmic spectral density for soft environments.
    
    Spectral density: J(ω) = η * ω^s * exp(-ω/ωc)
    where 0 < s < 1 is the sub-Ohmic exponent.
    """
    
    def __init__(self, coupling_strength: float = 0.2, cutoff_frequency: float = 1.0, 
                 exponent: float = 0.5):
        """Initialize Sub-Ohmic noise model.
        
        Args:
            coupling_strength: Dimensionless coupling strength η
            cutoff_frequency: Cutoff frequency ωc in rad/s
            exponent: Sub-Ohmic exponent s (0 < s < 1)
        """
        if not 0 < exponent < 1:
            raise ValueError("Sub-Ohmic exponent must be between 0 and 1")
        
        super().__init__("sub_ohmic",
                        coupling_strength=coupling_strength,
                        cutoff_frequency=cutoff_frequency,
                        exponent=exponent)
    
    def get_spectral_density(self, frequency: float, temperature: float) -> float:
        """Calculate Sub-Ohmic spectral density."""
        if frequency < 0:
            raise ValueError("Frequency must be non-negative")
        
        eta = self.parameters["coupling_strength"]
        omega_c = self.parameters["cutoff_frequency"]
        s = self.parameters["exponent"]
        
        return eta * (frequency ** s) * np.exp(-frequency / omega_c)
    
    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for Sub-Ohmic model."""
        return ["coupling_strength", "cutoff_frequency", "exponent"]
    
    def _validate_parameter(self, param: str, value: Any) -> str:
        """Validate Sub-Ohmic specific parameters."""
        base_error = super()._validate_parameter(param, value)
        if base_error:
            return base_error
        
        if param == "exponent" and not (0 < value < 1):
            return "Sub-Ohmic exponent must be between 0 and 1"
        
        return ""


class ProteinNoiseModel(OhmicNoiseModel):
    """Noise model for protein environments with Ohmic spectral density.
    
    Based on experimental studies of protein dynamics and quantum coherence
    in biological systems. Uses Ohmic spectral density with parameters
    typical for protein environments at physiological conditions.
    
    References:
    - Ishizaki & Fleming, PNAS 106, 17255 (2009)
    - Chin et al., Nature Physics 9, 113 (2013)
    """
    
    def __init__(self, coupling_strength: float = 0.1, cutoff_frequency: float = 1.0,
                 protein_type: str = "generic", temperature: float = 300.0):
        """Initialize protein noise model with typical protein parameters.
        
        Args:
            coupling_strength: System-bath coupling strength (dimensionless)
            cutoff_frequency: Cutoff frequency in rad/ps (typical: 0.5-2.0)
            protein_type: Type of protein environment ("generic", "photosynthetic", "enzyme")
            temperature: Temperature in Kelvin
        """
        # Adjust parameters based on protein type
        if protein_type == "photosynthetic":
            # Parameters for photosynthetic complexes (FMO, LHCII)
            coupling_strength = coupling_strength * 0.8  # Slightly weaker coupling
            cutoff_frequency = cutoff_frequency * 1.2    # Higher cutoff
        elif protein_type == "enzyme":
            # Parameters for enzyme active sites
            coupling_strength = coupling_strength * 1.5  # Stronger coupling
            cutoff_frequency = cutoff_frequency * 0.8    # Lower cutoff
        
        super().__init__(coupling_strength, cutoff_frequency)
        self.model_type = "protein_ohmic"
        self.parameters["protein_type"] = protein_type
        self.parameters["temperature"] = temperature
    
    def get_reorganization_energy(self) -> float:
        """Calculate reorganization energy for protein environment.
        
        Returns:
            Reorganization energy in eV
        """
        # λ = ∫₀^∞ J(ω)/ω dω for Ohmic spectral density
        # For J(ω) = η*ω*exp(-ω/ωc), λ = η*ωc
        eta = self.parameters["coupling_strength"]
        omega_c = self.parameters["cutoff_frequency"]
        
        # For biological systems, typical reorganization energies are 10-100 meV
        # The coupling strength is often given in units that already account for
        # the energy scale. Here we use a more realistic conversion.
        # Assume omega_c is in units of typical vibrational frequencies (~100 cm⁻¹)
        # and eta is dimensionless coupling strength
        
        # Convert: 1 cm⁻¹ = 1.24e-4 eV, typical ωc ~ 100-500 cm⁻¹
        # So ωc in rad/ps corresponds to ~100-500 cm⁻¹
        cm_inv_to_ev = 1.24e-4  # eV per cm⁻¹
        typical_freq_cm = 200.0  # cm⁻¹, typical protein vibrational frequency
        
        # Scale omega_c (rad/ps) to cm⁻¹
        omega_c_cm = omega_c * typical_freq_cm
        
        # Calculate reorganization energy
        reorganization_energy_ev = eta * omega_c_cm * cm_inv_to_ev
        
        return reorganization_energy_ev
    
    def get_characteristic_timescales(self) -> Dict[str, float]:
        """Get characteristic timescales for protein environment.
        
        Returns:
            Dictionary of timescales in ps
        """
        omega_c = self.parameters["cutoff_frequency"]
        
        return {
            "bath_correlation_time": 1.0 / omega_c,  # ps
            "vibrational_period": 2 * np.pi / omega_c,  # ps
            "dephasing_time": self._estimate_dephasing_time(),  # ps
        }
    
    def _estimate_dephasing_time(self) -> float:
        """Estimate pure dephasing time based on model parameters."""
        # Rough estimate: T₂* ≈ ℏ/(2*λ*kT) for high temperature
        lambda_reorg = self.get_reorganization_energy()  # eV
        temperature = self.parameters.get("temperature", 300.0)  # K
        k_B = 8.617333e-5  # eV/K
        
        if lambda_reorg > 0 and temperature > 0:
            # Convert to ps (ℏ ≈ 0.658 meV⋅ps = 6.58e-4 eV⋅ps)
            hbar_ev_ps = 6.58e-4  # eV⋅ps
            dephasing_time = hbar_ev_ps / (2 * lambda_reorg * k_B * temperature)
            return max(dephasing_time, 0.01)  # Minimum 0.01 ps (10 fs)
        else:
            return 1.0  # Default 1 ps


class MembraneNoiseModel(SuperOhmicNoiseModel):
    """Noise model for membrane environments with structured spectral density.
    
    Models quantum decoherence in lipid membrane environments, accounting for
    the structured nature of lipid bilayers and their collective modes.
    Uses super-Ohmic spectral density to capture membrane flexibility.
    
    References:
    - Olaya-Castro et al., Phys. Rev. B 78, 085115 (2008)
    - Mohseni et al., J. Chem. Phys. 129, 174106 (2008)
    """
    
    def __init__(self, coupling_strength: float = 0.05, cutoff_frequency: float = 0.5,
                 lipid_composition: Dict[str, float] = None, membrane_thickness: float = 4.0):
        """Initialize membrane noise model with typical membrane parameters.
        
        Args:
            coupling_strength: System-membrane coupling strength
            cutoff_frequency: Cutoff frequency in rad/ps (typical: 0.2-1.0)
            lipid_composition: Dictionary of lipid types and their fractions
            membrane_thickness: Membrane thickness in nm
        """
        if lipid_composition is None:
            lipid_composition = {"POPC": 0.6, "POPE": 0.3, "cholesterol": 0.1}
        
        # Adjust parameters based on membrane composition
        cholesterol_fraction = lipid_composition.get("cholesterol", 0.0)
        
        # Cholesterol increases membrane rigidity, reducing coupling
        coupling_adjustment = 1.0 - 0.3 * cholesterol_fraction
        adjusted_coupling = coupling_strength * coupling_adjustment
        
        # Membrane thickness affects cutoff frequency
        thickness_factor = membrane_thickness / 4.0  # Normalize to typical thickness
        adjusted_cutoff = cutoff_frequency / thickness_factor
        
        super().__init__(adjusted_coupling, adjusted_cutoff)
        self.model_type = "membrane"
        self.parameters["lipid_composition"] = lipid_composition
        self.parameters["membrane_thickness"] = membrane_thickness
    
    def get_membrane_properties(self) -> Dict[str, float]:
        """Calculate membrane-specific properties.
        
        Returns:
            Dictionary of membrane properties
        """
        lipid_comp = self.parameters["lipid_composition"]
        thickness = self.parameters["membrane_thickness"]
        
        # Estimate membrane fluidity based on composition
        cholesterol_fraction = lipid_comp.get("cholesterol", 0.0)
        unsaturated_fraction = lipid_comp.get("POPC", 0.0) + lipid_comp.get("POPE", 0.0)
        
        fluidity = unsaturated_fraction * (1.0 - 0.5 * cholesterol_fraction)
        
        return {
            "fluidity": fluidity,
            "thickness_nm": thickness,
            "cholesterol_content": cholesterol_fraction,
            "estimated_diffusion_coefficient": fluidity * 1e-8,  # cm²/s
        }
    
    def get_collective_mode_frequencies(self) -> np.ndarray:
        """Get characteristic frequencies of membrane collective modes.
        
        Returns:
            Array of mode frequencies in rad/ps
        """
        # Membrane collective modes: bending, undulation, thickness fluctuations
        thickness = self.parameters["membrane_thickness"]
        
        # Typical membrane mode frequencies (scaled by thickness)
        base_frequencies = np.array([0.1, 0.3, 0.8, 1.5])  # rad/ps
        thickness_scaling = 4.0 / thickness  # Thicker membranes have lower frequencies
        
        return base_frequencies * thickness_scaling


class SolventNoiseModel(SubOhmicNoiseModel):
    """Noise model for solvent environments with ionic effects.
    
    Models quantum decoherence in aqueous and organic solvent environments,
    including effects of ionic strength, pH, and solvent polarity.
    Uses sub-Ohmic spectral density to capture solvent cage dynamics.
    
    References:
    - Jang et al., J. Chem. Phys. 118, 9324 (2003)
    - Nalbach et al., Phys. Rev. Lett. 103, 220401 (2009)
    """
    
    def __init__(self, coupling_strength: float = 0.2, cutoff_frequency: float = 2.0,
                 ionic_strength: float = 0.15, solvent_type: str = "water",
                 ph: float = 7.0, dielectric_constant: float = 80.0):
        """Initialize solvent noise model.
        
        Args:
            coupling_strength: Base coupling strength
            cutoff_frequency: Cutoff frequency in rad/ps
            ionic_strength: Ionic strength in M (affects coupling)
            solvent_type: Type of solvent ("water", "organic", "mixed")
            ph: pH of the solution
            dielectric_constant: Relative dielectric constant of solvent
        """
        # Adjust coupling based on ionic strength and dielectric properties
        ionic_factor = 1.0 + 0.5 * ionic_strength  # Ionic screening effect
        dielectric_factor = 80.0 / dielectric_constant  # Relative to water
        
        adjusted_coupling = coupling_strength * ionic_factor * dielectric_factor
        
        # Adjust cutoff frequency based on solvent type
        if solvent_type == "organic":
            adjusted_cutoff = cutoff_frequency * 0.7  # Slower dynamics
            exponent = 0.5  # More sub-Ohmic
        elif solvent_type == "mixed":
            adjusted_cutoff = cutoff_frequency * 0.85
            exponent = 0.6
        else:  # water
            adjusted_cutoff = cutoff_frequency
            exponent = 0.7
        
        super().__init__(adjusted_coupling, adjusted_cutoff, exponent)
        self.model_type = "solvent_ionic"
        self.parameters.update({
            "ionic_strength": ionic_strength,
            "solvent_type": solvent_type,
            "ph": ph,
            "dielectric_constant": dielectric_constant
        })
    
    def get_debye_length(self) -> float:
        """Calculate Debye screening length in nm.
        
        Returns:
            Debye length in nm
        """
        if self.parameters["ionic_strength"] <= 0:
            return float('inf')
        
        # Debye length: λ_D = √(ε₀εᵣkT/2e²I)
        # Simplified formula: λ_D ≈ 0.304/√I nm (at 25°C in water)
        ionic_strength = self.parameters["ionic_strength"]
        dielectric = self.parameters["dielectric_constant"]
        
        # Adjust for dielectric constant
        debye_length = 0.304 * np.sqrt(dielectric / 80.0) / np.sqrt(ionic_strength)
        
        return debye_length
    
    def get_solvent_properties(self) -> Dict[str, float]:
        """Get solvent-specific properties.
        
        Returns:
            Dictionary of solvent properties
        """
        return {
            "ionic_strength_M": self.parameters["ionic_strength"],
            "ph": self.parameters["ph"],
            "dielectric_constant": self.parameters["dielectric_constant"],
            "debye_length_nm": self.get_debye_length(),
            "solvent_type": self.parameters["solvent_type"],
        }
    
    def calculate_ionic_screening_factor(self, distance_nm: float) -> float:
        """Calculate ionic screening factor at given distance.
        
        Args:
            distance_nm: Distance in nm
            
        Returns:
            Screening factor (0-1)
        """
        debye_length = self.get_debye_length()
        
        if debye_length == float('inf'):
            return 1.0  # No screening
        
        return np.exp(-distance_nm / debye_length)
    
    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for solvent model."""
        return super()._get_required_parameters() + [
            "ionic_strength", "solvent_type", "ph", "dielectric_constant"
        ]
    
    def _validate_parameter(self, param: str, value: Any) -> str:
        """Validate solvent-specific parameters."""
        base_error = super()._validate_parameter(param, value)
        if base_error:
            return base_error
        
        if param == "ionic_strength" and value < 0:
            return "Ionic strength must be non-negative"
        elif param == "ph" and not (0 <= value <= 14):
            return "pH must be between 0 and 14"
        elif param == "dielectric_constant" and value <= 0:
            return "Dielectric constant must be positive"
        elif param == "solvent_type" and value not in ["water", "organic", "mixed"]:
            return "Solvent type must be 'water', 'organic', or 'mixed'"
        
        return ""


class NoiseModelFactory:
    """Factory class for creating appropriate noise models."""
    
    @staticmethod
    def create_noise_model(model_type: str, temperature: float, **kwargs):
        """Create noise model of specified type."""
        if model_type == "protein_ohmic":
            return NoiseModelFactory.create_protein_noise_model(temperature, **kwargs)
        elif model_type == "membrane":
            return NoiseModelFactory.create_membrane_noise_model(**kwargs)
        elif model_type == "solvent_ionic":
            return NoiseModelFactory.create_solvent_noise_model(**kwargs)
        else:
            return NoiseModelFactory.create_custom_noise_model(model_type, **kwargs)
    
    @staticmethod
    def create_protein_noise_model(temperature: float, coupling_strength: float = 0.1,
                                  cutoff_frequency: float = 1.0) -> ProteinNoiseModel:
        """Create protein environment noise model.
        
        Args:
            temperature: Temperature in Kelvin (for validation)
            coupling_strength: Coupling strength parameter
            cutoff_frequency: Cutoff frequency in rad/s
            
        Returns:
            Configured protein noise model
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        return ProteinNoiseModel(coupling_strength, cutoff_frequency)
    
    @staticmethod
    def create_membrane_noise_model(lipid_composition: Dict = None,
                                   coupling_strength: float = 0.05,
                                   cutoff_frequency: float = 0.5) -> MembraneNoiseModel:
        """Create membrane environment noise model.
        
        Args:
            lipid_composition: Dictionary of lipid types and fractions (for future use)
            coupling_strength: Coupling strength parameter
            cutoff_frequency: Cutoff frequency in rad/s
            
        Returns:
            Configured membrane noise model
        """
        # Future enhancement: adjust parameters based on lipid composition
        if lipid_composition is not None:
            # Placeholder for lipid-specific adjustments
            pass
        
        return MembraneNoiseModel(coupling_strength, cutoff_frequency)
    
    @staticmethod
    def create_solvent_noise_model(solvent_type: str = "water", 
                                  ionic_strength: float = 0.15,
                                  coupling_strength: float = 0.2,
                                  cutoff_frequency: float = 2.0) -> SolventNoiseModel:
        """Create solvent environment noise model.
        
        Args:
            solvent_type: Type of solvent (for future parameter adjustment)
            ionic_strength: Ionic strength in M
            coupling_strength: Base coupling strength
            cutoff_frequency: Cutoff frequency in rad/s
            
        Returns:
            Configured solvent noise model
        """
        if ionic_strength < 0:
            raise ValueError("Ionic strength must be non-negative")
        
        # Future enhancement: adjust parameters based on solvent type
        if solvent_type not in ["water", "organic", "mixed"]:
            raise ValueError(f"Unsupported solvent type: {solvent_type}")
        
        return SolventNoiseModel(coupling_strength, cutoff_frequency, ionic_strength)
    
    @staticmethod
    def create_custom_noise_model(model_type: str, **parameters) -> NoiseModel:
        """Create custom noise model with specified parameters.
        
        Args:
            model_type: Type of noise model ("ohmic", "super_ohmic", "sub_ohmic")
            **parameters: Model-specific parameters
            
        Returns:
            Configured noise model
        """
        if model_type == "ohmic":
            return OhmicNoiseModel(**parameters)
        elif model_type == "super_ohmic":
            return SuperOhmicNoiseModel(**parameters)
        elif model_type == "sub_ohmic":
            return SubOhmicNoiseModel(**parameters)
        else:
            raise ValueError(f"Unknown noise model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available noise model types."""
        return ["protein_ohmic", "membrane", "solvent_ionic", "ohmic", 
                "super_ohmic", "sub_ohmic"]