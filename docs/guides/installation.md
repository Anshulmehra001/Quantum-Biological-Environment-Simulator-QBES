# QBES Installation Guide

## Prerequisites
- Python 3.8+ (Python 3.9+ recommended)
- 8GB+ RAM (16GB+ recommended)
- 2GB+ free disk space

## Quick Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-
cd QBES
```

### Step 2: Install QBES
```bash
# Standard installation
pip install -e .

# With all dependencies
pip install -e .[dev,gpu,visualization]
```

### Step 3: Verify Installation
```bash
# Check QBES installation
python -c "import qbes; print('QBES version:', qbes.__version__)"

# Run quick validation
python -m qbes.cli validate --suite quick
```

## Virtual Environment Setup (Recommended)

```bash
# Create virtual environment
python -m venv qbes_env
source qbes_env/bin/activate  # Linux/Mac
qbes_env\Scripts\activate     # Windows

# Install QBES
pip install -e .
```

## Troubleshooting

### Common Issues
- **Import Error**: Check Python path and reinstall with `pip install -e . --force-reinstall`
- **Missing Dependencies**: Install with `pip install -r requirements.txt`
- **Permission Errors**: Use virtual environment

### System Requirements
- **Minimum**: Python 3.8+, 8GB RAM, 2GB storage
- **Recommended**: Python 3.9+, 32GB RAM, SSD storage, 8+ CPU cores

## Verification
After installation, run:
```bash
python -c "import qbes; print('✅ QBES installed successfully')"
```