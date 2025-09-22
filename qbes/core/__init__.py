"""
Core data structures and interfaces for QBES.
"""

from .data_models import (
    SimulationConfig,
    QuantumSubsystem,
    SimulationResults,
    Atom,
    QuantumState,
    DensityMatrix,
    MolecularSystem,
    ValidationResult,
    CoherenceMetrics,
    StatisticalSummary
)

from .interfaces import (
    SimulationEngineInterface,
    QuantumEngineInterface,
    MDEngineInterface,
    NoiseModelInterface,
    AnalysisInterface,
    VisualizationInterface
)

__all__ = [
    # Data models
    "SimulationConfig",
    "QuantumSubsystem", 
    "SimulationResults",
    "Atom",
    "QuantumState",
    "DensityMatrix",
    "MolecularSystem",
    "ValidationResult",
    "CoherenceMetrics",
    "StatisticalSummary",
    # Interfaces
    "SimulationEngineInterface",
    "QuantumEngineInterface",
    "MDEngineInterface", 
    "NoiseModelInterface",
    "AnalysisInterface",
    "VisualizationInterface"
]