# Contributing to QBES

Thank you for your interest in contributing to the Quantum Biological Environment Simulator (QBES)! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [License](#license)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-
   cd QBES
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv qbes_env
   source qbes_env/bin/activate  # Linux/Mac
   qbes_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import qbes; print('QBES installed successfully')"
   python -m pytest tests/ -v
   ```

## Contributing Guidelines

### Types of Contributions

We welcome the following types of contributions:

- 🐛 **Bug Reports**: Report issues or unexpected behavior
- 🚀 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve or add documentation
- 🧪 **Tests**: Add or improve test coverage
- 🔧 **Code**: Fix bugs or implement new features
- 🎨 **Examples**: Add new simulation examples

### Before Contributing

1. **Check Existing Issues**: Look for existing issues or discussions
2. **Create an Issue**: For significant changes, create an issue first
3. **Discuss**: Engage with maintainers about your proposed changes

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   python -m pytest tests/ -v
   python test_project.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new quantum coherence analysis"
   # or
   git commit -m "fix: resolve numerical instability in Lindblad solver"
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Code Structure

```python
def calculate_coherence_lifetime(state_trajectory: List[DensityMatrix]) -> float:
    """
    Calculate quantum coherence lifetime from state evolution.
    
    Args:
        state_trajectory: List of density matrices over time
        
    Returns:
        Coherence lifetime in simulation time units
        
    Raises:
        ValueError: If trajectory is empty or invalid
    """
    # Implementation here
    pass
```

### Commit Message Format

Use conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `style:` - Code style changes
- `chore:` - Maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_quantum_engine.py -v

# Run with coverage
python -m pytest tests/ --cov=qbes --cov-report=html
```

### Writing Tests

- Add tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Include edge cases and boundary conditions

Example test structure:
```python
def test_quantum_state_evolution():
    """Test quantum state evolution with Lindblad equation."""
    # Arrange
    initial_state = create_test_state()
    hamiltonian = create_test_hamiltonian()
    
    # Act
    evolved_state = quantum_engine.evolve_state(
        initial_state, time_step=0.01, hamiltonian=hamiltonian
    )
    
    # Assert
    assert evolved_state.time == initial_state.time + 0.01
    assert np.isclose(np.trace(evolved_state.matrix), 1.0)
```

### Test Organization

Tests are organized in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and fixtures
├── unit/                          # Unit tests
│   ├── test_quantum_engine.py
│   ├── test_simulation_engine.py
│   └── test_analysis.py
├── integration/                   # Integration tests
│   ├── test_full_simulation.py
│   └── test_cli_integration.py
├── benchmarks/                    # Benchmark tests
│   ├── test_performance.py
│   └── test_accuracy.py
└── fixtures/                      # Test data and fixtures
    ├── test_systems.py
    └── sample_data/
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **User Documentation**: Guides and tutorials in `docs/`
3. **API Documentation**: Detailed API reference
4. **Examples**: Practical usage examples

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date with code changes
- Use proper Markdown formatting

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html
```

## Submitting Changes

### Pull Request Process

1. **Ensure Tests Pass**
   ```bash
   python -m pytest tests/ -v
   python test_project.py
   ```

2. **Update Documentation**
   - Update relevant documentation
   - Add docstrings to new functions
   - Update CHANGELOG.md if applicable

3. **Create Pull Request**
   - Use a descriptive title
   - Provide detailed description of changes
   - Reference related issues
   - Include screenshots if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated

## Related Issues
Fixes #123
```

## Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Verify functionality works as expected
4. **Documentation**: Ensure documentation is complete and accurate

## Getting Help

- **Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact aniketmehra715@gmail.com for direct communication

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

## License

By contributing to QBES, you agree that your contributions will be licensed under the Creative Commons BY-NC-SA 4.0 license. See [LICENSE](LICENSE) for details.

---

**Repository**: https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-  
**Developer**: Aniket Mehra  
**License**: Creative Commons BY-NC-SA 4.0

Thank you for contributing to QBES! 🧬⚛️