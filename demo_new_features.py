"""
Demonstration of QBES New Features

This script showcases the enhanced validation, performance profiling,
and literature benchmark capabilities added to QBES.
"""

import numpy as np
from qbes.validation import EnhancedValidator, validate_simulation
from qbes.performance import PerformanceProfiler, profile_simulation, quick_profile
from qbes.benchmarks.literature import LiteratureBenchmarks, validate_against_literature


def demo_validation():
    """Demonstrate enhanced validation capabilities."""
    print("=" * 80)
    print("DEMO 1: Enhanced Validation")
    print("=" * 80)
    
    validator = EnhancedValidator()
    
    # Create sample density matrix
    print("\n1. Creating sample density matrix...")
    n_sites = 3
    
    # Simple valid density matrix
    rho_matrix = np.array([
        [0.5, 0.2, 0.1],
        [0.2, 0.3, 0.05],
        [0.1, 0.05, 0.2]
    ], dtype=complex)
    
    # Create DensityMatrix object
    from dataclasses import dataclass
    
    @dataclass
    class DensityMatrix:
        matrix: np.ndarray
        time: float = 0.0
    
    rho = DensityMatrix(matrix=rho_matrix, time=0.0)
    
    # Validate
    print("2. Running validation...")
    metrics = validator.validate_density_matrix(rho)
    
    # Display results
    print("\n3. Validation Results:")
    print(f"   Hermiticity Error: {metrics.hermiticity_error:.2e}")
    print(f"   Trace Error: {metrics.norm_preservation:.2e}")
    print(f"   Positivity: {'✅ Yes' if metrics.positivity_check else '❌ No'}")
    print(f"   Physical Bounds: {'✅ Yes' if metrics.physical_bounds else '❌ No'}")
    print(f"   Numerical Stability: {'✅ Yes' if metrics.numerical_stability else '❌ No'}")
    print(f"   Overall: {'✅ PASS' if metrics.density_matrix_valid else '❌ FAIL'}")
    
    print("\n" + "=" * 80)


def demo_performance():
    """Demonstrate performance profiling."""
    print("=" * 80)
    print("DEMO 2: Performance Profiling")
    print("=" * 80)
    
    profiler = PerformanceProfiler("Demo Session")
    profiler.start_session("Performance Demo")
    
    # Simulate some operations
    print("\n1. Running profiled operations...")
    
    with profiler.profile_operation("Matrix Multiplication"):
        # Simulate heavy computation
        import time
        A = np.random.rand(1000, 1000)
        B = np.random.rand(1000, 1000)
        C = A @ B
        time.sleep(0.1)
    
    with profiler.profile_operation("Data Processing"):
        # Simulate data processing
        data = np.random.rand(10000, 100)
        result = np.linalg.svd(data, full_matrices=False)
        time.sleep(0.05)
    
    with profiler.profile_operation("I/O Operations"):
        # Simulate I/O
        import time
        time.sleep(0.2)
    
    profiler.end_session()
    
    # Get bottlenecks
    print("\n2. Identifying bottlenecks...")
    bottlenecks = profiler.get_bottlenecks()
    if bottlenecks:
        print(f"   Found {len(bottlenecks)} bottleneck(s):")
        for bn in bottlenecks:
            print(f"   - {bn}")
    
    print("\n" + "=" * 80)


def demo_literature_benchmarks():
    """Demonstrate literature benchmark comparison."""
    print("=" * 80)
    print("DEMO 3: Literature Benchmarks")
    print("=" * 80)
    
    benchmarks = LiteratureBenchmarks()
    
    # List available benchmarks
    print("\n1. Available Literature Benchmarks:")
    for name in benchmarks.list_benchmarks():
        bench = benchmarks.get_benchmark(name)
        print(f"   - {bench.name}")
        print(f"     Reference: {bench.reference}")
        print(f"     Year: {bench.year}")
        print()
    
    # Get FMO benchmark details
    print("\n2. FMO Complex Details (Engel et al. 2007):")
    fmo = benchmarks.get_benchmark('fmo_engel_2007')
    print(f"   System: {fmo.system}")
    print(f"   Temperature: {fmo.parameters['temperature']} K")
    print(f"   Number of sites: {fmo.parameters['n_sites']}")
    print(f"   Coherence lifetime: {fmo.results['coherence_lifetime'][0]*1e15:.0f} fs")
    print(f"   Transfer time: {fmo.results['transfer_time'][0]*1e12:.1f} ps")
    print(f"   Quantum efficiency: {fmo.results['quantum_efficiency'][0]:.2%}")
    
    print("\n3. Site Energies (cm⁻¹):")
    for i, energy in enumerate(fmo.parameters['site_energies'], 1):
        print(f"   Site {i}: {energy:.1f}")
    
    print("\n" + "=" * 80)


def demo_quick_profile():
    """Demonstrate quick profiling context manager."""
    print("=" * 80)
    print("DEMO 4: Quick Profiling")
    print("=" * 80)
    
    print("\n1. Using quick_profile for ad-hoc profiling...")
    
    with quick_profile("Quick Test Operation"):
        # Some operation
        import time
        result = sum(i**2 for i in range(1000000))
        time.sleep(0.1)
    
    print(f"   Result: {result}")
    print("\n" + "=" * 80)


def demo_complete_workflow():
    """Demonstrate complete workflow with all features."""
    print("=" * 80)
    print("DEMO 5: Complete Workflow Example")
    print("=" * 80)
    
    print("\nThis demonstrates how to use all features together:")
    print("""
# Import modules
from qbes import SimulationEngine, ConfigurationManager
from qbes.validation import validate_simulation
from qbes.performance import PerformanceProfiler
from qbes.benchmarks.literature import validate_against_literature

# Setup profiler
profiler = PerformanceProfiler()
profiler.start_session("Complete Analysis")

# Load and run simulation
with profiler.profile_operation("Simulation"):
    config = ConfigurationManager.load_config("fmo_config.yaml")
    engine = SimulationEngine(config)
    results = engine.run_simulation()

# Validate results
with profiler.profile_operation("Validation"):
    validation = validate_simulation(results)
    print(validation.generate_report())

# Compare with literature
with profiler.profile_operation("Literature Comparison"):
    lit_validation = validate_against_literature(results, 'fmo_engel_2007')
    print(f"Literature match: {lit_validation.is_valid}")

profiler.end_session()
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "QBES NEW FEATURES DEMONSTRATION" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    try:
        # Run demonstrations
        demo_validation()
        print()
        
        demo_performance()
        print()
        
        demo_literature_benchmarks()
        print()
        
        demo_quick_profile()
        print()
        
        demo_complete_workflow()
        print()
        
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 30 + "DEMO COMPLETE!" + " " * 33 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("✅ All new features demonstrated successfully!")
        print()
        print("For more information, see:")
        print("  - qbes/validation/README.md")
        print("  - qbes/performance/README.md")
        print("  - qbes/benchmarks/literature/README.md")
        print("  - PROJECT_STRUCTURE_UPDATE.md")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
