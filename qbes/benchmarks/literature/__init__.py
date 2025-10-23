"""
Literature benchmarks for QBES validation.

Compare QBES simulations with published results from quantum biology research.
"""

from .literature_benchmarks import (
    LiteratureBenchmark,
    LiteratureBenchmarks,
    validate_against_literature
)

__all__ = [
    'LiteratureBenchmark',
    'LiteratureBenchmarks',
    'validate_against_literature'
]
