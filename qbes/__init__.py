"""
Quantum Biological Environment Simulator (QBES)

A scientific software toolkit for simulating quantum mechanics within noisy biological environments.
"""

__version__ = "0.1.0"
__author__ = "QBES Development Team"
__email__ = "qbes@example.com"

from .config_manager import ConfigurationManager
from .quantum_engine import QuantumEngine
from .md_engine import MDEngine
from .noise_models import NoiseModelFactory
from .analysis import ResultsAnalyzer, CoherenceAnalyzer
from .visualization import VisualizationEngine

# Import simulation engine
try:
    from .simulation_engine import SimulationEngine
except ImportError:
    SimulationEngine = None

__all__ = [
    "ConfigurationManager",
    "QuantumEngine", 
    "MDEngine",
    "NoiseModelFactory",
    "ResultsAnalyzer",
    "CoherenceAnalyzer",
    "VisualizationEngine",
    "SimulationEngine"
]