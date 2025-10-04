"""
Minimal benchmark test for debugging.
"""

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

print("DEBUG: Starting minimal benchmark file")

@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    system_name: str
    test_passed: bool
    numerical_result: float
    analytical_result: float

print("DEBUG: BenchmarkResult defined")

class BenchmarkSystem(ABC):
    """Abstract base class for benchmark test systems."""
    
    def __init__(self, name: str):
        self.name = name

print("DEBUG: BenchmarkSystem defined")

class TwoLevelSystemBenchmark(BenchmarkSystem):
    """Simple two-level system benchmark."""
    
    def __init__(self):
        super().__init__("Two-Level System")

print("DEBUG: TwoLevelSystemBenchmark defined")
print("DEBUG: All classes defined successfully")