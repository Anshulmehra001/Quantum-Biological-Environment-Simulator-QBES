# QBES Testing Suite

This directory contains the comprehensive testing suite for the Quantum Biological Environment Simulator (QBES).

## 📁 Directory Structure

```
tests/
├── conftest.py                    # Pytest configuration and shared fixtures
├── pytest.ini                     # Pytest settings
├── run_tests.py                   # Test runner script
├── README.md                      # This file
├── unit/                          # Unit tests
│   ├── __init__.py
│   ├── test_quantum_engine.py     # Quantum engine tests
│   ├── test_simulation_engine.py  # Simulation engine tests
│   ├── test_analysis.py           # Analysis module tests
│   └── test_cli.py                # CLI interface tests
├── integration/                   # Integration tests
│   ├── __init__.py
│   └── test_full_simulation.py    # End-to-end simulation tests
├── benchmarks/                    # Performance and accuracy benchmarks
│   ├── __init__.py
│   └── test_performance.py        # Performance benchmark tests
└── fixtures/                      # Test data and fixtures
    ├── __init__.py
    └── test_systems.py            # Predefined test systems
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:
- **Quantum Engine**: State evolution, Hamiltonian construction, coherence calculations
- **Simulation Engine**: Orchestration, progress tracking, checkpointing
- **Analysis Module**: Statistical analysis, coherence lifetime calculation
- **CLI Interface**: Command-line argument parsing and execution

### Integration Tests (`tests/integration/`)
Test component interactions and workflows:
- **Full Simulation**: Complete simulation workflows
- **Configuration**: Config loading and validation
- **Error Handling**: Error propagation and recovery

### Benchmark Tests (`tests/benchmarks/`)
Performance and accuracy validation:
- **Performance**: Execution speed and memory usage
- **Accuracy**: Comparison with analytical solutions
- **Literature Validation**: Comparison with published results

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run quick test suite (unit tests only, no slow tests)
python run_tests.py --suite quick

# Run with coverage
python run_tests.py --coverage
```

### Using pytest directly
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/benchmarks/ -v

# Run tests with specific markers
python -m pytest -m "unit" -v
python -m pytest -m "not slow" -v
python -m pytest -m "benchmark" -v

# Run with coverage
python -m pytest tests/ --cov=qbes --cov-report=html
```

### Test Suites

#### Quick Suite (2-5 minutes)
```bash
python run_tests.py --suite quick
```
- Fast unit tests only
- No slow or benchmark tests
- Good for development workflow

#### Unit Tests (5-10 minutes)
```bash
python run_tests.py --suite unit
```
- All unit tests including slow ones
- Tests individual components
- Good for component development

#### Integration Tests (10-15 minutes)
```bash
python run_tests.py --suite integration
```
- Component interaction tests
- End-to-end workflow tests
- Good for system validation

#### Benchmark Tests (15-30 minutes)
```bash
python run_tests.py --suite benchmark
```
- Performance benchmarks
- Accuracy validation
- Literature comparison
- Good for performance analysis

#### Full Suite (30-45 minutes)
```bash
python run_tests.py --suite all
```
- All test categories
- Complete validation
- Good for release testing

## 🏷️ Test Markers

Tests are marked with pytest markers for easy selection:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.benchmark` - Benchmark tests
- `@pytest.mark.slow` - Tests that take longer to run

### Running Specific Markers
```bash
# Run only unit tests
python -m pytest -m "unit" -v

# Run fast tests only (exclude slow)
python -m pytest -m "not slow" -v

# Run benchmark tests
python -m pytest -m "benchmark" -v

# Combine markers
python -m pytest -m "unit and not slow" -v
```

## 🔧 Test Configuration

### Fixtures
Common test fixtures are defined in `conftest.py`:
- `quantum_engine` - QuantumEngine instance
- `simulation_engine` - SimulationEngine instance
- `simple_density_matrix` - 2x2 test density matrix
- `two_level_hamiltonian` - Simple Hamiltonian
- `temp_output_dir` - Temporary directory for test outputs

### Test Systems
Predefined test systems in `fixtures/test_systems.py`:
- Two-level quantum systems
- Three-level ladder systems
- Photosynthetic complex models
- Enzyme active site models
- Analytical test cases with known solutions

## 📊 Coverage Reporting

Generate coverage reports:
```bash
# HTML coverage report
python -m pytest tests/ --cov=qbes --cov-report=html

# Terminal coverage report
python -m pytest tests/ --cov=qbes --cov-report=term-missing

# XML coverage report (for CI)
python -m pytest tests/ --cov=qbes --cov-report=xml
```

Coverage reports are saved to `htmlcov/index.html`.

## 🐛 Debugging Tests

### Verbose Output
```bash
python -m pytest tests/ -v -s
```

### Run Specific Test
```bash
python -m pytest tests/unit/test_quantum_engine.py::TestQuantumEngine::test_create_two_level_hamiltonian -v
```

### Debug Mode
```bash
python -m pytest tests/ --pdb
```

### Capture Output
```bash
python -m pytest tests/ -v -s --capture=no
```

## ⚡ Parallel Testing

Run tests in parallel (requires pytest-xdist):
```bash
# Auto-detect CPU cores
python -m pytest tests/ -n auto

# Specific number of processes
python -m pytest tests/ -n 4

# Using test runner
python run_tests.py --parallel 4
```

## 🎯 Writing New Tests

### Test Structure
```python
import pytest
from qbes.quantum_engine import QuantumEngine

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self, quantum_engine):
        """Test basic functionality."""
        # Arrange
        input_data = create_test_input()
        
        # Act
        result = quantum_engine.new_method(input_data)
        
        # Assert
        assert result is not None
        assert result.property == expected_value
    
    @pytest.mark.parametrize("param1,param2,expected", [
        (1.0, 2.0, 3.0),
        (2.0, 3.0, 5.0),
    ])
    def test_parametrized(self, quantum_engine, param1, param2, expected):
        """Test with multiple parameter sets."""
        result = quantum_engine.calculate(param1, param2)
        assert result == expected
    
    @pytest.mark.slow
    def test_performance(self, quantum_engine):
        """Test performance (marked as slow)."""
        # Performance test code
        pass
```

### Test Guidelines
1. **Arrange-Act-Assert**: Structure tests clearly
2. **Descriptive Names**: Use descriptive test method names
3. **Single Responsibility**: Test one thing per test method
4. **Use Fixtures**: Leverage shared fixtures from conftest.py
5. **Mark Appropriately**: Use pytest markers for categorization
6. **Test Edge Cases**: Include boundary conditions and error cases
7. **Mock External Dependencies**: Use mocks for external systems

## 📈 Continuous Integration

For CI/CD pipelines:
```bash
# Fast CI tests (< 5 minutes)
python run_tests.py --suite quick --coverage

# Full CI tests (< 30 minutes)
python run_tests.py --suite all --coverage --parallel auto
```

## 🔍 Test Quality Metrics

### Coverage Targets
- **Unit Tests**: >90% line coverage
- **Integration Tests**: >80% feature coverage
- **Overall**: >85% total coverage

### Performance Targets
- **Unit Tests**: <10 seconds per test
- **Integration Tests**: <60 seconds per test
- **Benchmark Tests**: <300 seconds per test

## 📞 Support

For testing issues:
- Check test output for specific error messages
- Review test documentation and examples
- Run tests with verbose output (`-v -s`)
- Contact: aniketmehra715@gmail.com

---

**Repository**: https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-  
**License**: Creative Commons BY-NC-SA 4.0