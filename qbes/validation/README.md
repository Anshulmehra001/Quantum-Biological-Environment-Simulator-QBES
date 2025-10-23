# Enhanced Validation Module

## Overview

The enhanced validation module provides comprehensive validation and verification capabilities for QBES simulations beyond the basic validation framework.

## Features

### Density Matrix Validation
- Hermiticity checks
- Trace normalization
- Positivity (non-negative eigenvalues)
- Bounds verification
- Stability analysis

### Energy Conservation
- Total energy tracking
- Conservation violation detection
- Drift analysis
- Statistical measures

### Thermalization Validation
- Boltzmann distribution comparison
- Thermal equilibrium verification
- Temperature consistency

### Coherence Analysis
- Exponential decay fitting
- Decoherence time extraction
- Coherence lifetime comparison

### Analytical Comparison
- RMSE calculations
- Reference solution validation
- Error metrics

## Usage

### Basic Validation

```python
from qbes.validation import EnhancedValidator, validate_simulation

# Create validator
validator = EnhancedValidator()

# Validate density matrices
dm_valid = validator.validate_density_matrix(density_matrices)

# Validate energy conservation
energy_valid = validator.validate_energy_conservation(trajectory)

# Full validation
metrics = validate_simulation(simulation_results)
print(metrics.generate_report())
```

### With Simulation Engine

```python
from qbes import SimulationEngine
from qbes.validation import validate_simulation

# Run simulation
engine = SimulationEngine(config)
results = engine.run_simulation()

# Validate results
validation = validate_simulation(results)

if not validation.is_valid:
    print("Validation failed!")
    for error in validation.errors:
        print(f"  Error: {error}")
```

### Custom Validation

```python
validator = EnhancedValidator()

# Validate specific aspects
metrics = validator.validate_density_matrix(
    rho_trajectory,
    check_hermiticity=True,
    check_trace=True,
    check_positivity=True,
    check_bounds=True
)

print(f"Hermiticity: {metrics.is_hermitian}")
print(f"Trace error: {metrics.trace_error}")
print(f"Min eigenvalue: {metrics.min_eigenvalue}")
```

## Validation Metrics

The `ValidationMetrics` class contains:

```python
@dataclass
class ValidationMetrics:
    is_valid: bool
    is_hermitian: bool
    trace_error: float
    min_eigenvalue: float
    max_eigenvalue: float
    energy_conservation_error: float
    thermalization_score: float
    coherence_lifetime: Optional[float]
    coherence_fit_r2: Optional[float]
    rmse: Optional[float]
    errors: List[str]
    warnings: List[str]
```

## Best Practices

1. **Always validate after simulation**: Check physical consistency
2. **Review warnings**: Even if `is_valid=True`, warnings may indicate issues
3. **Compare with analytical**: Use known solutions when available
4. **Track trends**: Monitor validation metrics across parameter sweeps
5. **Set appropriate tolerances**: Adjust thresholds for your system

## Integration with Testing

```python
import pytest
from qbes.validation import validate_simulation

def test_simulation_validation():
    results = run_test_simulation()
    validation = validate_simulation(results)
    
    assert validation.is_valid, f"Validation failed: {validation.errors}"
    assert validation.trace_error < 1e-6
    assert validation.min_eigenvalue >= -1e-10
```

## Performance Notes

- Validation adds ~10-20% overhead to simulation time
- Most expensive: eigenvalue calculations for large systems
- Consider validating subset of timesteps for long simulations
- Use `validate_every_n_steps` parameter for efficiency

## Troubleshooting

### "Density matrix not Hermitian"
- Check numerical precision settings
- Verify quantum evolution operators are Hermitian
- Increase integration accuracy (rtol, atol)

### "Trace not normalized"
- Check for numerical drift in long simulations
- Verify normalization in initial conditions
- Consider periodic renormalization

### "Negative eigenvalues detected"
- Usually indicates numerical instability
- Reduce timestep size
- Check Lindblad operators are properly defined
- Increase solver tolerance

### "Large energy conservation error"
- Verify Hamiltonian is time-independent (or properly tracked)
- Check MD integration parameters
- Review thermostat/barostat settings

## See Also

- `qbes.validation.validator`: Basic validation framework
- `qbes.validation.accuracy_calculator`: Accuracy metrics
- `qbes.benchmarks.literature`: Literature comparison
- `docs/guides/validation.md`: Validation guide
