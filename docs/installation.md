# Installation Guide

This guide provides detailed instructions for installing the Quantum Biological Environment Simulator (QBES) on various platforms.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Detailed Installation](#detailed-installation)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Dependency Management](#dependency-management)
- [GPU Support](#gpu-support)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8 GB
- **Storage**: 2 GB free space
- **Internet**: Required for dependency installation

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 20.04+) or macOS (11.0+)
- **Python**: 3.9 or 3.10
- **RAM**: 32 GB or more
- **CPU**: Multi-core processor (8+ cores recommended)
- **Storage**: SSD with 10+ GB free space
- **GPU**: CUDA-compatible GPU (optional, for acceleration)

### Supported Python Versions

QBES is tested and supported on:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

## Quick Installation

For most users, the following commands will install QBES:

```bash
# Clone the repository
git clone https://github.com/your-org/qbes.git
cd qbes

# Install in development mode
pip install -e .

# Verify installation
qbes --version
```

If you encounter issues, see the [Detailed Installation](#detailed-installation) section below.

## Detailed Installation

### Step 1: Python Environment Setup

We strongly recommend using a virtual environment to avoid conflicts with other packages:

#### Using conda (recommended)

```bash
# Create a new conda environment
conda create -n qbes python=3.9
conda activate qbes

# Install conda-forge packages first (for better compatibility)
conda install -c conda-forge numpy scipy matplotlib h5py pyyaml
```

#### Using venv

```bash
# Create virtual environment
python -m venv qbes_env

# Activate environment (Linux/macOS)
source qbes_env/bin/activate

# Activate environment (Windows)
qbes_env\Scripts\activate
```

### Step 2: Clone Repository

```bash
git clone https://github.com/your-org/qbes.git
cd qbes
```

### Step 3: Install Dependencies

#### Option A: Automatic Installation (Recommended)

```bash
# Install QBES and all dependencies
pip install -e .
```

#### Option B: Manual Dependency Installation

If automatic installation fails, install dependencies manually:

```bash
# Core scientific computing
pip install numpy>=1.20.0 scipy>=1.7.0 matplotlib>=3.3.0

# Quantum mechanics
pip install qutip>=4.6.0

# Molecular dynamics
pip install openmm>=7.6.0 mdtraj>=1.9.0 biopython>=1.78

# Data handling
pip install pandas>=1.3.0 h5py>=3.1.0 pyyaml>=5.4.0

# Additional dependencies
pip install numba>=0.54.0 seaborn>=0.11.0 plotly>=5.0.0
pip install tqdm>=4.60.0 click>=8.0.0 psutil>=5.8.0
pip install astropy>=4.2.0 joblib>=1.0.0

# Install QBES
pip install -e .
```

### Step 4: Verification

```bash
# Check QBES installation
qbes --version

# Run basic tests
python -c "import qbes; print('QBES imported successfully')"

# Run benchmark suite (optional but recommended)
qbes benchmark run-benchmarks --output-dir ./test_benchmarks
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

#### Prerequisites

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install python3-dev python3-pip git build-essential
sudo apt install libopenmm-dev libfftw3-dev liblapack-dev libblas-dev

# For GPU support (optional)
sudo apt install nvidia-cuda-toolkit
```

#### Installation

```bash
# Install QBES
git clone https://github.com/your-org/qbes.git
cd qbes
pip install -e .
```

### macOS

#### Prerequisites

Install Xcode command line tools:

```bash
xcode-select --install
```

Install Homebrew (if not already installed):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install system dependencies:

```bash
# Install Python and development tools
brew install python@3.9 git cmake

# Install scientific libraries
brew install openblas lapack fftw

# For OpenMM (molecular dynamics)
brew install openmm
```

#### Installation

```bash
# Install QBES
git clone https://github.com/your-org/qbes.git
cd qbes
pip install -e .
```

### Windows

#### Prerequisites

1. Install Python 3.9+ from [python.org](https://python.org)
2. Install Git from [git-scm.com](https://git-scm.com)
3. Install Microsoft Visual C++ Build Tools

#### Using Windows Subsystem for Linux (WSL) - Recommended

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu

# Follow Linux installation instructions within WSL
```

#### Native Windows Installation

```cmd
# Clone repository
git clone https://github.com/your-org/qbes.git
cd qbes

# Create virtual environment
python -m venv qbes_env
qbes_env\Scripts\activate

# Install dependencies
pip install -e .
```

## Dependency Management

### Core Dependencies

QBES requires several scientific computing libraries:

- **NumPy**: Numerical computing foundation
- **SciPy**: Scientific computing algorithms
- **QuTiP**: Quantum information processing
- **OpenMM**: Molecular dynamics simulations
- **matplotlib**: Plotting and visualization

### Optional Dependencies

- **CuPy**: GPU acceleration (requires CUDA)
- **MPI4Py**: Parallel computing support
- **Jupyter**: Interactive notebooks
- **Sphinx**: Documentation generation

### Dependency Conflicts

If you encounter dependency conflicts:

1. Use a fresh virtual environment
2. Install dependencies in the recommended order
3. Use conda for scientific packages when possible
4. Check for version compatibility issues

## GPU Support

QBES can optionally use GPU acceleration for certain calculations.

### NVIDIA GPU (CUDA)

#### Prerequisites

1. Install NVIDIA drivers
2. Install CUDA toolkit (version 11.0+)
3. Verify CUDA installation:

```bash
nvidia-smi
nvcc --version
```

#### Installation

```bash
# Install CuPy for GPU acceleration
pip install cupy-cuda11x  # Replace 11x with your CUDA version

# Verify GPU support
python -c "import cupy; print('GPU support available')"
```

### AMD GPU (ROCm)

AMD GPU support is experimental. Contact the development team for assistance.

## Verification

### Basic Verification

```bash
# Check QBES version
qbes --version

# Test Python import
python -c "import qbes; print('QBES version:', qbes.__version__)"

# Check CLI functionality
qbes --help
```

### Comprehensive Testing

```bash
# Run unit tests (if pytest is installed)
python -m pytest tests/ -v

# Run benchmark suite
qbes benchmark run-benchmarks --output-dir ./verification_results

# Test configuration generation
qbes generate-config test_config.yaml
qbes validate test_config.yaml
```

### Performance Testing

```bash
# Run performance benchmarks
qbes benchmark run-benchmarks --scaling-test --output-dir ./performance_test

# Check system resources
python -c "
import psutil
print(f'CPU cores: {psutil.cpu_count()}')
print(f'RAM: {psutil.virtual_memory().total / 1e9:.1f} GB')
"
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'qbes'

**Solution**: Ensure you installed QBES in development mode:
```bash
pip install -e .
```

#### OpenMM installation fails

**Solution**: Install OpenMM via conda:
```bash
conda install -c conda-forge openmm
```

#### QuTiP compilation errors

**Solution**: Install pre-compiled QuTiP:
```bash
conda install -c conda-forge qutip
```

#### Memory errors during simulation

**Solutions**:
1. Reduce system size in configuration
2. Increase virtual memory/swap space
3. Use a machine with more RAM

#### Slow performance

**Solutions**:
1. Enable GPU acceleration (if available)
2. Reduce simulation time or time step
3. Use parallel processing options
4. Optimize system configuration

### Platform-Specific Issues

#### Linux: Permission denied errors

```bash
# Fix permissions
sudo chown -R $USER:$USER ~/.local/
```

#### macOS: SSL certificate errors

```bash
# Update certificates
/Applications/Python\ 3.9/Install\ Certificates.command
```

#### Windows: Long path issues

Enable long path support in Windows settings or use WSL.

### Getting Help

If you continue to experience issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/your-org/qbes/issues)
3. Create a new issue with:
   - Your operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce the problem

### Automated Installation Script

For convenience, QBES includes an automated installation script:

```bash
# Run automated installation
python scripts/install_qbes.py

# Check installation
python scripts/verify_installation.py
```

This script will:
- Detect your system configuration
- Install appropriate dependencies
- Verify the installation
- Run basic tests

## Next Steps

After successful installation:

1. Read the [User Guide](user_guide.md)
2. Try the [Tutorial](tutorial.md)
3. Run example simulations
4. Explore the [API Reference](api_reference.md)

## Updates and Maintenance

### Updating QBES

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -e . --upgrade

# Verify update
qbes --version
```

### Uninstalling QBES

```bash
# Uninstall QBES
pip uninstall qbes

# Remove virtual environment (if used)
conda env remove -n qbes  # for conda
rm -rf qbes_env/          # for venv
```