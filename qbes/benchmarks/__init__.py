"""
Benchmarking and validation suite for QBES.

This module provides benchmark test systems with known analytical solutions,
literature validation, cross-validation against other packages, and 
comprehensive statistical analysis for scientific validation.
"""

from .benchmark_systems import (
    BenchmarkSystem,
    BenchmarkResult,
    TwoLevelSystemBenchmark,
    BenchmarkRunner,
    run_quick_benchmarks
)

# Import additional benchmarks if available
try:
    from .benchmark_systems import (
        HarmonicOscillatorBenchmark,
        DampedTwoLevelSystemBenchmark,
        PhotosyntheticComplexBenchmark
    )
    _additional_benchmarks_available = True
except ImportError:
    _additional_benchmarks_available = False

# Import literature validation
try:
    from .literature_validation import (
        LiteratureValidator,
        LiteratureValidationResult,
        FMOComplexDataset,
        PhotosystemIIDataset,
        run_literature_validation
    )
    _literature_validation_available = True
except ImportError:
    _literature_validation_available = False

# Import cross-validation
try:
    from .cross_validation import (
        CrossValidator,
        CrossValidationResult,
        QuTiPInterface,
        run_cross_validation
    )
    _cross_validation_available = True
except ImportError:
    _cross_validation_available = False

# Import statistical testing
try:
    from .statistical_testing import (
        StatisticalTester,
        ComprehensiveStatisticalReport,
        perform_statistical_validation
    )
    _statistical_testing_available = True
except ImportError:
    _statistical_testing_available = False

# Import comprehensive validation
try:
    from .validation_reports import (
        ComprehensiveValidationReporter,
        ValidationSummary,
        run_comprehensive_validation
    )
    _comprehensive_validation_available = True
except ImportError:
    _comprehensive_validation_available = False

# Import automated benchmarks and performance testing
try:
    from .automated_benchmarks import (
        AutomatedBenchmarkRunner,
        run_automated_benchmarks
    )
    from .performance_benchmarks import (
        PerformanceBenchmarker,
        run_performance_benchmarks
    )
    _automated_benchmarks_available = True
except ImportError:
    _automated_benchmarks_available = False

__all__ = [
    'BenchmarkSystem',
    'BenchmarkResult',
    'TwoLevelSystemBenchmark',
    'BenchmarkRunner',
    'run_quick_benchmarks'
]

if _additional_benchmarks_available:
    __all__.extend([
        'HarmonicOscillatorBenchmark',
        'DampedTwoLevelSystemBenchmark',
        'PhotosyntheticComplexBenchmark'
    ])

if _literature_validation_available:
    __all__.extend([
        'LiteratureValidator',
        'LiteratureValidationResult',
        'FMOComplexDataset',
        'PhotosystemIIDataset',
        'run_literature_validation'
    ])

if _cross_validation_available:
    __all__.extend([
        'CrossValidator',
        'CrossValidationResult',
        'QuTiPInterface',
        'run_cross_validation'
    ])

if _statistical_testing_available:
    __all__.extend([
        'StatisticalTester',
        'ComprehensiveStatisticalReport',
        'perform_statistical_validation'
    ])

if _comprehensive_validation_available:
    __all__.extend([
        'ComprehensiveValidationReporter',
        'ValidationSummary',
        'run_comprehensive_validation'
    ])

if _automated_benchmarks_available:
    __all__.extend([
        'AutomatedBenchmarkRunner',
        'run_automated_benchmarks',
        'PerformanceBenchmarker',
        'run_performance_benchmarks'
    ])