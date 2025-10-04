# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when using QBES.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Simulation Errors](#simulation-errors)
- [Performance Issues](#performance-issues)
- [Results and Analysis Problems](#results-and-analysis-problems)
- [Platform-Specific Issues](#platform-specific-issues)
- [Frequently Asked Questions](#frequently-asked-questions)

## Installation Issues

### ImportError: No module named 'qbes'

**Problem**: QBES is not properly installed or not in Python path.

**Solutions**:
```bash
# Reinstall in development mode
pip install -e .

# Check if QBES is in Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify installation
python -c "import qbes; print(qbes.__version__)"
```

### Dependency Installation Failures

#### OpenMM Installation Issues

**Problem**: OpenMM fails to install or import.

**Solutions**:
```bash
# Use conda for OpenMM (recommended)
conda install -c conda-forge openmm

# Alternative: use conda-forge for all scientific packages
conda install -c conda-forge numpy scipy matplotlib openmm mdtraj

# Verify OpenMM installation
python -c "import openmm; print('OpenMM version:', openmm.version.version)"
```

#### QuTiP Compilation Errors

**Problem**: QuTiP fails to compile from source.

**Solutions**:
```bash
# Install pre-compiled QuTiP
conda install -c conda-forge qutip

# Or use pip with pre-compiled wheels
pip install qutip --only-binary=all

# For development version
pip install git+https://github.com/qutip/qutip.git
```

#### CUDA/GPU Support Issues

**Problem**: GPU acceleration not working.

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install appropriate CUDA toolkit
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x

# Verify GPU support
python -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

### Virtual Environment Issues

**Problem**: Conflicts between system and virtual environment packages.

**Solutions**:
```bash
# Create clean environment
conda create -n qbes_clean python=3.9
conda activate qbes_clean

# Install only necessary packages
conda install -c conda-forge numpy scipy matplotlib openmm qutip
pip install -e .

# Or use venv
python -m venv qbes_clean
source qbes_clean/bin/activate  # Linux/Mac
# qbes_clean\Scripts\activate  # Windows
pip install -e .
```

## Configuration Problems

### YAML Parsing Errors

**Problem**: Configuration file has syntax errors.

**Common Issues**:
```yaml
# WRONG: Inconsistent indentation
system:
  pdb_file: "structure.pdb"
   force_field: "amber14"  # Wrong indentation

# CORRECT:
system:
  pdb_file: "structure.pdb"
  force_field: "amber14"

# WRONG: Missing quotes for strings with special characters
pdb_file: path/to/file with spaces.pdb

# CORRECT:
pdb_file: "path/to/file with spaces.pdb"
```

**Solutions**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Use QBES validation
qbes validate config.yaml
```

### Parameter Validation Errors

**Problem**: Configuration parameters are outside valid ranges.

**Common Issues and Solutions**:

```yaml
# Temperature must be positive
simulation:
  temperature: -100.0  # WRONG
  temperature: 300.0   # CORRECT

# Time step must be appropriate for system
simulation:
  time_step: 1.0       # WRONG: Too large
  time_step: 1.0e-15   # CORRECT: 1 femtosecond

# Coupling strength should be reasonable
noise_model:
  coupling_strength: 100.0  # WRONG: Too strong
  coupling_strength: 2.0    # CORRECT
```

### File Path Issues

**Problem**: PDB files or other input files not found.

**Solutions**:
```bash
# Use absolute paths
system:
  pdb_file: "/full/path/to/structure.pdb"

# Or relative paths from working directory
system:
  pdb_file: "./structures/structure.pdb"

# Check file exists
ls -la path/to/structure.pdb

# Verify file permissions
chmod 644 structure.pdb
```

## Simulation Errors

### Memory Errors

**Problem**: Simulation runs out of memory.

**Symptoms**:
- `MemoryError` exceptions
- System becomes unresponsive
- Killed by OS (Linux: "Killed")

**Solutions**:

1. **Reduce System Size**:
```yaml
quantum_subsystem:
  max_quantum_atoms: 50  # Reduce from larger number
  selection_method: "active_site"  # More selective
```

2. **Optimize Memory Usage**:
```yaml
output:
  save_trajectory: false  # Don't save full trajectory
  checkpoint_interval: 10000  # Less frequent checkpoints
```

3. **Use Sparse Representations**:
```python
# In custom code
import scipy.sparse as sp
# Use sparse matrices for large Hamiltonians
```

4. **Increase Virtual Memory**:
```bash
# Linux: Increase swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Numerical Instability

**Problem**: Simulation produces unphysical results.

**Symptoms**:
- Negative probabilities
- Norm violations
- Exponentially growing values
- NaN or Inf values

**Solutions**:

1. **Reduce Time Step**:
```yaml
simulation:
  time_step: 1.0e-16  # Smaller time step
```

2. **Use Adaptive Integration**:
```yaml
simulation:
  integration_method: "adaptive_runge_kutta"
  error_tolerance: 1.0e-8
```

3. **Check Initial Conditions**:
```python
# Verify initial state is normalized
initial_state = results.initial_state
norm = np.trace(initial_state @ initial_state.conj().T)
print(f"Initial state norm: {norm}")  # Should be 1.0
```

### Convergence Issues

**Problem**: Simulation doesn't converge or takes too long.

**Solutions**:

1. **Check Convergence Criteria**:
```yaml
simulation:
  convergence_threshold: 1.0e-6
  max_iterations: 100000
```

2. **Use Better Initial Guess**:
```yaml
quantum_subsystem:
  initial_state: "thermal_equilibrium"  # Better than random
```

3. **Optimize Parameters**:
```yaml
noise_model:
  coupling_strength: 1.0  # Reduce if too strong
```

## Performance Issues

### Slow Simulation Speed

**Problem**: Simulations take much longer than expected.

**Diagnostic Steps**:
```python
# Profile your simulation
import cProfile
import pstats

# Run with profiling
cProfile.run('engine.run_simulation()', 'simulation_profile.prof')

# Analyze results
stats = pstats.Stats('simulation_profile.prof')
stats.sort_stats('cumulative').print_stats(20)
```

**Solutions**:

1. **Enable Compiler Optimizations**:
```bash
# Set environment variables
export NUMBA_DISABLE_JIT=0
export OMP_NUM_THREADS=8  # Use multiple cores
```

2. **Use GPU Acceleration**:
```yaml
simulation:
  use_gpu: true
  gpu_device: 0
```

3. **Optimize System Size**:
```yaml
quantum_subsystem:
  max_quantum_atoms: 100  # Reduce if possible
  basis_truncation: "adaptive"
```

### High Memory Usage

**Problem**: Simulation uses more memory than available.

**Solutions**:

1. **Monitor Memory Usage**:
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Call periodically during simulation
```

2. **Use Memory-Efficient Options**:
```yaml
output:
  save_trajectory: false
  save_intermediate: false
  compression: "gzip"
```

3. **Implement Checkpointing**:
```yaml
output:
  save_checkpoints: true
  checkpoint_interval: 1000
  max_checkpoints: 5  # Keep only recent checkpoints
```

## Results and Analysis Problems

### Missing Output Files

**Problem**: Expected output files are not generated.

**Solutions**:

1. **Check Output Directory**:
```bash
# Verify directory exists and is writable
ls -la output_directory/
touch output_directory/test_file  # Test write permissions
```

2. **Check Configuration**:
```yaml
output:
  directory: "./output"  # Ensure path is correct
  save_trajectory: true  # Enable desired outputs
  save_plots: true
```

3. **Check for Errors**:
```bash
# Look for error logs
ls output_directory/error_*.log
cat output_directory/error_*.log
```

### Incorrect Results

**Problem**: Results don't match expectations or literature values.

**Diagnostic Steps**:

1. **Run Benchmark Tests**:
```bash
qbes benchmark run-benchmarks --output-dir ./validation
```

2. **Check Units and Scaling**:
```python
# Verify units are consistent
print(f"Time units: {results.time_units}")
print(f"Energy units: {results.energy_units}")
```

3. **Compare with Simple Systems**:
```bash
# Run two-level system benchmark
qbes generate-config simple_test.yaml --template default
# Modify for simple two-level system
qbes run simple_test.yaml
```

### Visualization Problems

**Problem**: Plots are not generated or look incorrect.

**Solutions**:

1. **Check Matplotlib Backend**:
```python
import matplotlib
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Set appropriate backend
matplotlib.use('Agg')  # For headless systems
```

2. **Install Additional Dependencies**:
```bash
pip install seaborn plotly
```

3. **Check Plot Configuration**:
```yaml
output:
  plot_format: "png"  # or "pdf", "svg"
  plot_dpi: 300
  plot_style: "scientific"
```

## Platform-Specific Issues

### Linux Issues

#### Permission Denied Errors
```bash
# Fix file permissions
chmod +x scripts/*.py
sudo chown -R $USER:$USER ~/.local/

# Fix library paths
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### Missing System Libraries
```bash
# Install development packages
sudo apt-get install python3-dev build-essential
sudo apt-get install libfftw3-dev liblapack-dev libblas-dev
```

### macOS Issues

#### Xcode Command Line Tools
```bash
# Install if missing
xcode-select --install

# Update if outdated
softwareupdate --install -a
```

#### Homebrew Package Conflicts
```bash
# Clean up Homebrew
brew cleanup
brew doctor

# Reinstall problematic packages
brew uninstall --ignore-dependencies openmm
brew install openmm
```

### Windows Issues

#### Long Path Names
```bash
# Enable long path support in Windows settings
# Or use WSL for Linux compatibility
wsl --install -d Ubuntu
```

#### Visual C++ Build Tools
```bash
# Install Microsoft Visual C++ Build Tools
# Download from Microsoft website
# Or use conda for pre-compiled packages
conda install -c conda-forge openmm qutip
```

## Frequently Asked Questions

### General Usage

**Q: How do I choose appropriate simulation parameters?**

A: Start with template configurations and adjust based on your system:
- **Time step**: 1 fs for most systems, 0.1 fs for fast dynamics
- **Simulation time**: 1-10 ps for coherence studies
- **Temperature**: Match experimental conditions
- **System size**: Keep quantum region < 200 atoms

**Q: How long should simulations take?**

A: Typical times:
- Simple systems (< 50 atoms): Minutes to hours
- Medium systems (50-200 atoms): Hours to days
- Large systems (> 200 atoms): Days to weeks

**Q: How do I validate my results?**

A: 
1. Run benchmark tests first
2. Compare with analytical solutions for simple cases
3. Check energy conservation
4. Verify physical reasonableness of results

### Scientific Questions

**Q: What biological systems can QBES simulate?**

A: QBES is designed for:
- Photosynthetic complexes
- Enzyme active sites
- DNA/RNA dynamics
- Membrane proteins
- Any system with quantum coherence effects

**Q: How accurate are the results?**

A: Accuracy depends on:
- Quality of molecular structure
- Appropriateness of noise model
- Convergence of simulation parameters
- Validation against benchmarks

**Q: Can I compare results with experiments?**

A: Yes, but consider:
- Experimental conditions (temperature, environment)
- Time scales accessible to experiments
- Observable quantities that can be measured

### Technical Questions

**Q: Can I run QBES on a cluster?**

A: Yes, QBES supports:
- MPI parallel processing
- Job submission systems (SLURM, PBS)
- GPU acceleration
- Checkpoint/restart for long jobs

**Q: How do I extend QBES functionality?**

A: You can:
- Create custom analysis modules
- Implement new noise models
- Add visualization tools
- Contribute to the main codebase

**Q: What file formats does QBES support?**

A: Input formats:
- PDB (molecular structures)
- YAML (configuration)
- Various MD trajectory formats

Output formats:
- JSON (results summary)
- HDF5 (trajectory data)
- CSV (time series)
- PNG/PDF (plots)

## Getting Additional Help

If this troubleshooting guide doesn't resolve your issue:

1. **Check Documentation**:
   - [User Guide](user_guide.md)
   - [API Reference](api_reference.md)
   - [Theory and Methods](theory.md)

2. **Search Existing Issues**:
   - [GitHub Issues](https://github.com/your-org/qbes/issues)
   - [GitHub Discussions](https://github.com/your-org/qbes/discussions)

3. **Create New Issue**:
   Include:
   - Operating system and version
   - Python version
   - QBES version
   - Complete error message
   - Minimal example to reproduce problem
   - Configuration file (if relevant)

4. **Contact Support**:
   - Email: qbes-support@example.com
   - Include system information and error details

## Contributing to Troubleshooting

Help improve this guide by:
- Reporting new issues and solutions
- Suggesting clarifications
- Adding platform-specific solutions
- Sharing optimization tips

Your contributions help the entire QBES community!