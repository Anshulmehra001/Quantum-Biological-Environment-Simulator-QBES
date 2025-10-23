# Literature Benchmarks Module

## Overview

The literature benchmarks module provides comparison with published results from quantum biology research. Validate QBES simulations against well-established experimental and theoretical benchmarks.

## Included Benchmarks

### 1. FMO Complex - Engel et al. (2007)
**Reference**: Nature 446, 782-786 (2007)

Famous paper demonstrating long-lived quantum coherence in photosynthetic light harvesting.

- **System**: 7-site FMO complex from green sulfur bacteria
- **Temperature**: 77 K
- **Coherence lifetime**: 660 ± 60 fs
- **Energy transfer time**: 1 ± 0.1 ps
- **Quantum efficiency**: 95 ± 5%

### 2. FMO Complex - Ishizaki & Fleming (2009)
**Reference**: PNAS 106, 17255-17260 (2009)

Theoretical study with optimized spectral density parameters.

- **System**: 7-site FMO complex
- **Temperature**: 300 K (room temperature)
- **Reorganization energy**: 35 cm⁻¹
- **Cutoff frequency**: 166 cm⁻¹
- **Transfer efficiency**: 99%

### 3. Photosystem II - Romero et al. (2014)
**Reference**: Nature Physics 10, 676-682 (2014)

Long-lived quantum coherence at room temperature in PSII reaction center.

- **System**: 4-site PSII reaction center
- **Temperature**: 294 K
- **Coherence lifetime**: 400 ± 50 fs
- **Quantum yield**: 85%
- **Technique**: 2D electronic spectroscopy

### 4. Two-Level System - Analytical
**Reference**: Textbook (Sakurai Quantum Mechanics)

Exact analytical solution for validation.

- **System**: Driven two-level system
- **Temperature**: 0 K
- **Dynamics**: Rabi oscillations with dephasing
- **Use**: Numerical accuracy verification

## Usage

### Basic Validation

```python
from qbes.benchmarks.literature import LiteratureBenchmarks

# Load benchmarks
benchmarks = LiteratureBenchmarks()

# List available benchmarks
print(benchmarks.list_benchmarks())
# ['fmo_engel_2007', 'fmo_ishizaki_2009', 'psii_romero_2014', 'tls_analytical']

# Get specific benchmark
fmo = benchmarks.get_benchmark('fmo_engel_2007')
print(f"System: {fmo.name}")
print(f"Reference: {fmo.reference}")
print(f"Site energies: {fmo.parameters['site_energies']}")
```

### Compare Simulation with Literature

```python
from qbes import SimulationEngine
from qbes.benchmarks.literature import LiteratureBenchmarks

# Run simulation
engine = SimulationEngine(fmo_config)
results = engine.run_simulation()

# Compare with literature
benchmarks = LiteratureBenchmarks()
comparison = benchmarks.compare_with_literature(
    results, 
    'fmo_engel_2007'
)

print(f"Coherence lifetime error: {comparison['coherence_lifetime_error']:.2%}")
print(f"Transfer time error: {comparison['transfer_time_error']:.2%}")
print(f"Match: {comparison['coherence_lifetime_match']}")  # True/False
```

### Validation Function

```python
from qbes.benchmarks.literature import validate_against_literature

# Quick validation
validation = validate_against_literature(
    simulation_results,
    benchmark='fmo_engel_2007'
)

if not validation.is_valid:
    print("Failed literature validation!")
    for error in validation.errors:
        print(f"  {error}")

for warning in validation.warnings:
    print(f"  Warning: {warning}")
```

### Generate Report

```python
benchmarks = LiteratureBenchmarks()

# Compare multiple benchmarks
comparisons = {}
for benchmark_name in ['fmo_engel_2007', 'fmo_ishizaki_2009']:
    comparisons[benchmark_name] = benchmarks.compare_with_literature(
        results,
        benchmark_name
    )

# Generate comprehensive report
report = benchmarks.generate_benchmark_report(comparisons)
print(report)
```

Example output:
```
======================================================================
LITERATURE BENCHMARK COMPARISON
======================================================================

Benchmark: FMO Complex - Engel et al. 2007
Reference: Nature 446, 782-786 (2007)

  coherence_lifetime_error: 0.1523
  coherence_lifetime_match: ✅ PASS
  transfer_time_error: 0.0856
  transfer_time_match: ✅ PASS
  efficiency_error: 0.0234
  efficiency_match: ✅ PASS

Benchmark: FMO Complex - Ishizaki & Fleming 2009
Reference: PNAS 106, 17255-17260 (2009)

  transfer_efficiency_error: 0.0412
  transfer_efficiency_match: ✅ PASS

======================================================================
```

## Running Benchmark Simulations

### Example: FMO Complex

```python
from qbes import SimulationEngine, ConfigurationManager
from qbes.benchmarks.literature import LiteratureBenchmarks

# Get benchmark parameters
benchmarks = LiteratureBenchmarks()
fmo_bench = benchmarks.get_benchmark('fmo_engel_2007')

# Create configuration from benchmark
config = ConfigurationManager.create_default()
config.quantum.num_sites = fmo_bench.parameters['n_sites']
config.quantum.site_energies = fmo_bench.parameters['site_energies'].tolist()
config.quantum.couplings = fmo_bench.parameters['couplings'].tolist()
config.quantum.temperature = fmo_bench.parameters['temperature']

# Run simulation
engine = SimulationEngine(config)
results = engine.run_simulation()

# Compare
comparison = benchmarks.compare_with_literature(results, 'fmo_engel_2007')

# Check agreement
if comparison['coherence_lifetime_match']:
    print("✅ Coherence lifetime matches literature!")
else:
    error = comparison['coherence_lifetime_error']
    print(f"❌ Coherence lifetime off by {error:.1%}")
```

## Benchmark Data Structure

Each benchmark contains:

```python
@dataclass
class LiteratureBenchmark:
    name: str                          # Full citation name
    system: str                        # System type (fmo_complex, etc.)
    reference: str                     # Journal citation
    year: int                          # Publication year
    parameters: Dict                   # System parameters (energies, couplings)
    results: Dict[str, np.ndarray]    # Measured results
    uncertainty: Optional[Dict]        # Experimental uncertainties
    conditions: Optional[Dict]         # Experimental conditions
```

## Comparison Metrics

The comparison returns:

- `coherence_lifetime_error`: Relative error in coherence decay time
- `coherence_lifetime_match`: True if within 30% of literature value
- `transfer_time_error`: Relative error in energy transfer time
- `transfer_time_match`: True if within 30% of literature value
- `efficiency_error`: Absolute error in quantum efficiency
- `efficiency_match`: True if within 10% of literature value

## Adding New Benchmarks

To add a new literature benchmark:

```python
def _my_new_benchmark(self) -> LiteratureBenchmark:
    """Description of the benchmark."""
    return LiteratureBenchmark(
        name="System - Author et al. Year",
        system="system_type",
        reference="Journal Vol, Pages (Year)",
        year=2024,
        parameters={
            'site_energies': np.array([...]),
            'couplings': np.array([[...]]),
            'temperature': 300.0
        },
        results={
            'coherence_lifetime': np.array([value]),
            'transfer_time': np.array([value])
        },
        uncertainty={
            'coherence_lifetime': error_value
        },
        conditions={
            'temperature': 300.0,
            'technique': 'method_name'
        }
    )

# Add to _load_benchmarks():
benchmarks['my_new_benchmark'] = self._my_new_benchmark()
```

## Best Practices

1. **Match conditions**: Ensure temperature, basis size match benchmark
2. **Check units**: Convert energies to cm⁻¹ when comparing
3. **Allow tolerance**: Experimental/theoretical differences expected
4. **Multiple benchmarks**: Validate against several sources
5. **Document deviations**: Note why results may differ

## Interpretation Guidelines

### Good Match (< 30% error)
- Physical model is correct
- Parameters are reasonable
- Numerical methods are accurate

### Moderate Deviation (30-50%)
- Check parameter values carefully
- Verify units and conversions
- Review approximations made

### Large Deviation (> 50%)
- Likely incorrect physical model
- Wrong parameters or units
- Numerical instability
- Different system/conditions

## Common Issues

### "No coherence_lifetime attribute"
- Simulation results must include coherence analysis
- Run coherence decay fitting before comparison

### "Large error in transfer time"
- Check if initial conditions match benchmark
- Verify site energy and coupling values
- Review temperature settings

### "Efficiency mismatch"
- Ensure proper sink/trap modeling
- Check if infinite time limit was reached
- Verify rate calculations

## References

1. **Engel et al.** (2007). Evidence for wavelike energy transfer through quantum coherence in photosynthetic systems. *Nature* 446, 782-786.

2. **Ishizaki & Fleming** (2009). Theoretical examination of quantum coherence in a photosynthetic system at physiological temperature. *PNAS* 106, 17255-17260.

3. **Romero et al.** (2014). Quantum coherence in photosynthesis for efficient solar-energy conversion. *Nature Physics* 10, 676-682.

4. **Cao et al.** (2020). Quantum biology revisited. *Science Advances* 6, eaaz4888.

## See Also

- `qbes.benchmarks.benchmark_runner`: Automated benchmark suite
- `qbes.validation`: Validation framework
- `qbes.performance`: Performance profiling
- `docs/guides/benchmarking.md`: Benchmarking guide
- `examples/benchmarks/`: Benchmark examples
