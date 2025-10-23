# Performance Optimization Module

## Overview

The performance module provides comprehensive profiling and optimization tools for QBES simulations. Track timing, memory usage, identify bottlenecks, and get optimization recommendations.

## Features

### Performance Profiling
- Wall-clock time measurement
- CPU time tracking
- Memory usage monitoring (peak and increase)
- Call counting
- Session-based organization

### Bottleneck Identification
- Automatic detection of slow operations
- CPU utilization analysis
- Memory hotspot identification
- Time distribution reports

### Optimization Analysis
- Scaling estimates for larger systems
- Optimization recommendations
- Session comparison
- Performance trends

## Usage

### Basic Profiling with Context Manager

```python
from qbes.performance import PerformanceProfiler

profiler = PerformanceProfiler("My Simulation")
profiler.start_session("Test Run")

with profiler.profile_operation("MD Integration"):
    run_md_simulation()

with profiler.profile_operation("Quantum Evolution"):
    solve_lindblad_equation()

profiler.end_session()
# Automatically prints performance report
```

### Decorator-Based Profiling

```python
from qbes.performance import profile_simulation

@profile_simulation
def run_my_simulation(config):
    # Your simulation code
    setup_system()
    run_dynamics()
    analyze_results()
    return results

# Automatically profiles entire function
results = run_my_simulation(my_config)
```

### Quick Profiling

```python
from qbes.performance import quick_profile

# Ad-hoc profiling for testing
with quick_profile("Data Processing"):
    process_large_dataset()
```

### Function Decorator

```python
profiler = PerformanceProfiler()

@profiler.profile_function
def expensive_operation(data):
    # Automatically profiled
    return complex_calculation(data)
```

## Performance Reports

The profiler automatically generates detailed reports:

```
================================================================================
PERFORMANCE PROFILING REPORT: QBES Simulation
================================================================================
Total Time: 45.234s

Time Breakdown:
--------------------------------------------------------------------------------
Operation                                Time (s)     CPU %      Memory (MB)    
--------------------------------------------------------------------------------
MD Integration                           25.123       98.5       1234.5         
Quantum Evolution                        15.678       95.2       567.8          
Hamiltonian Construction                 3.234        88.9       123.4          
Analysis                                 1.199        45.2       89.1           

⚠️  Performance Bottlenecks (>10% of total time):
  - MD Integration: 55.5% of total time
  - Quantum Evolution: 34.7% of total time

Memory Usage:
  Peak: 1234.5 MB
  Total Increase: 567.8 MB

================================================================================
```

## Optimization Recommendations

```python
from qbes.performance import OptimizationAnalyzer

analyzer = OptimizationAnalyzer()
recommendations = analyzer.analyze_simulation_performance(session)

for rec in recommendations:
    print(rec)
```

Example output:
```
⚡ MD Integration takes 25.1s. Consider: smaller timestep, fewer atoms, or GPU acceleration
⚡ Quantum simulation takes 15.7s. Consider: adaptive timestep, sparse matrices, or lower basis size
💾 High memory usage (1235MB). Consider: chunked processing, data streaming, or memory profiling
🔄 Low CPU utilization in Analysis (45%). Consider: parallelization or reducing I/O overhead
```

## Session Comparison

Compare performance between optimization attempts:

```python
profiler = PerformanceProfiler()

# First run
profiler.start_session("Before Optimization")
run_simulation(config)
profiler.end_session()

# Apply optimization
optimize_hamiltonian_caching()

# Second run
profiler.start_session("After Optimization")
run_simulation(config)
profiler.end_session()

# Compare
comparison = profiler.compare_sessions(0, 1)
for op, metrics in comparison.items():
    if metrics['improved']:
        speedup = metrics['speedup']
        print(f"{op}: {speedup:.2f}x faster! 🎉")
```

## Scaling Estimates

Estimate performance for larger systems:

```python
analyzer = OptimizationAnalyzer()

# Current: 100 atoms, 10 sites
current_metrics = session.metrics["MD Integration"]
estimates = analyzer.estimate_scaling(
    current_metrics,
    current_size=100,
    target_size=1000
)

print(f"Estimated time for 1000 atoms: {estimates['estimated_time']:.1f}s")
print(f"Estimated memory: {estimates['estimated_memory']:.1f}MB")
print(f"Feasible: {estimates['feasible']}")
```

## Integration with Simulation Engine

```python
from qbes import SimulationEngine
from qbes.performance import PerformanceProfiler

class ProfiledSimulationEngine(SimulationEngine):
    def __init__(self, config):
        super().__init__(config)
        self.profiler = PerformanceProfiler("QBES")
    
    def run_simulation(self):
        self.profiler.start_session("Full Simulation")
        
        with self.profiler.profile_operation("Initialization"):
            self._initialize()
        
        with self.profiler.profile_operation("MD Simulation"):
            self._run_md()
        
        with self.profiler.profile_operation("Quantum Dynamics"):
            self._run_quantum()
        
        with self.profiler.profile_operation("Analysis"):
            results = self._analyze()
        
        self.profiler.end_session()
        return results
```

## Performance Metrics

The `PerformanceMetrics` class tracks:

```python
@dataclass
class PerformanceMetrics:
    operation: str           # Operation name
    wall_time: float        # Wall-clock time (seconds)
    cpu_time: float         # CPU time (seconds)
    memory_peak: float      # Peak memory (MB)
    memory_increase: float  # Memory increase (MB)
    calls: int             # Number of calls
```

## Best Practices

1. **Profile before optimizing**: Identify real bottlenecks
2. **Use sessions**: Organize profiling by simulation stages
3. **Compare runs**: Track improvements objectively
4. **Check CPU utilization**: Low % suggests I/O or parallelization opportunity
5. **Monitor memory**: Large increases may indicate memory leaks
6. **Regular profiling**: Catch performance regressions early

## Common Bottlenecks

### MD Integration Slow
- **Reduce atoms**: Use smaller simulation box
- **Larger timestep**: If stable (check energy conservation)
- **GPU acceleration**: Use CUDA or OpenCL platforms
- **Cutoff optimization**: Tune nonbonded cutoffs

### Quantum Evolution Slow
- **Sparse matrices**: Use sparse Hamiltonian representation
- **Adaptive timestep**: Let solver optimize dt
- **Lower tolerance**: Relax rtol/atol if acceptable
- **Reduce basis**: Use fewer quantum states

### Hamiltonian Construction Slow
- **Cache results**: Reuse for repeated calculations
- **Precompute**: Calculate coupling terms once
- **Exploit symmetry**: Use molecular symmetry
- **Parallel construction**: Parallelize independent parts

### High Memory Usage
- **Chunked I/O**: Stream data instead of loading all
- **Reduce trajectory frequency**: Save fewer snapshots
- **Compress data**: Use HDF5 with compression
- **Generator patterns**: Use iterators instead of lists

## Troubleshooting

### "No active session" warning
- Start a session with `profiler.start_session()` before profiling operations

### Memory measurements seem wrong
- Ensure `psutil` is installed: `pip install psutil`
- Some OS-level caching may affect measurements

### CPU utilization over 100%
- Normal for multi-threaded operations (shows total across cores)

### Profiling overhead too high
- Use profiling selectively, not for every function
- Profile larger code blocks rather than tiny functions

## Dependencies

- `psutil`: Process and system monitoring
- `tracemalloc`: Memory allocation tracking (Python stdlib)
- `time`: High-precision timing (Python stdlib)

Install with:
```bash
pip install psutil
```

## See Also

- `qbes.benchmarks`: Performance benchmarking suite
- `qbes.validation`: Validation and verification
- `docs/guides/optimization.md`: Optimization guide
- `examples/performance/`: Performance examples
