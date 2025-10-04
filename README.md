# Quantum Biological Environment Simulator (QBES)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![Production Ready](https://img.shields.io/badge/status-production--ready-green.svg)]()

**Production-ready scientific software toolkit** for simulating quantum mechanics within noisy biological environments. QBES enables researchers to study quantum effects in biological systems such as photosynthesis, enzyme catalysis, and neural processes using scientifically rigorous open quantum systems theory.

## ⚠️ Important: Educational Website vs Research Software

This repository contains **TWO SEPARATE COMPONENTS**:

### 🔬 **QBES Research Software** (This Directory)
- **Purpose**: Production-ready scientific research tool
- **Location**: Main directory (`/qbes/`, `/tests/`, `/docs/`)
- **Users**: Researchers, scientists, academic institutions
- **Status**: Complete, validated, ready for scientific use

### 🎓 **Educational Website** (`/website/` directory)
- **Purpose**: Learning platform and project demonstration
- **Location**: `/website/` subdirectory only
- **Users**: Students, educators, general public
- **Status**: Interactive educational tool (NOT for research)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/qbes.git
cd qbes

# Install in development mode
pip install -e .

# Verify installation
qbes --version
```

### Basic Usage

```bash
# Generate a configuration file for your system
qbes generate-config my_simulation.yaml --template photosystem

# Validate your configuration
qbes validate my_simulation.yaml

# Run the simulation
qbes run my_simulation.yaml --verbose

# Monitor progress (in another terminal)
qbes status ./output

# Run benchmarks to verify installation
qbes benchmark run-benchmarks --output-dir ./benchmark_results
```

### Python API Example

```python
from qbes import ConfigurationManager, SimulationEngine

# Load configuration
config_manager = ConfigurationManager()
config = config_manager.load_config("configs/default_config.yaml")

# Initialize and run simulation
engine = SimulationEngine()
engine.initialize_simulation(config)
results = engine.run_simulation()

# Analyze results
print(f"Coherence lifetime: {results.coherence_lifetime:.2e} s")
print(f"Decoherence rate: {results.decoherence_rate:.2e} Hz")
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 8 GB (16 GB recommended for large systems)
- **Storage**: 2 GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 32 GB or more
- **CPU**: Multi-core processor (8+ cores recommended)
- **GPU**: CUDA-compatible GPU (optional, for acceleration)
- **Storage**: SSD with 10+ GB free space

### Dependencies

Core scientific libraries:
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- QuTiP ≥ 4.6.0 (quantum calculations)
- OpenMM ≥ 7.6.0 (molecular dynamics)
- matplotlib ≥ 3.3.0 (visualization)

See [`requirements.txt`](requirements.txt) for complete dependency list.

## 🔬 Key Features

### Scientific Rigor
- **Open Quantum Systems Theory**: Based on Lindblad master equation formalism
- **Validated Noise Models**: Scientifically validated biological environment models
- **Benchmark Suite**: Comprehensive validation against analytical solutions and literature
- **Uncertainty Quantification**: Statistical analysis and confidence intervals

### Hybrid QM/MM Approach
- **Quantum Subsystem**: Accurate quantum mechanical treatment of active regions
- **Classical Environment**: Molecular dynamics for realistic biological environments
- **Seamless Integration**: Automated coupling between quantum and classical components

### User-Friendly Interface
- **Command-Line Tools**: Intuitive CLI for all operations
- **Configuration Templates**: Pre-configured setups for common biological systems
- **Progress Monitoring**: Real-time simulation progress and status checking
- **Automated Installation**: Dependency detection and environment setup

### Comprehensive Analysis
- **Quantum Coherence Measures**: Coherence lifetimes, quantum discord, entanglement
- **Statistical Analysis**: Uncertainty quantification and significance testing
- **Publication-Ready Plots**: Automated generation of scientific figures
- **Data Export**: Multiple formats (HDF5, JSON, CSV) for further analysis

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[User Guide](docs/user_guide.md)** - Complete usage documentation
- **[Tutorial](docs/tutorial.md)** - Step-by-step examples
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Theory and Methods](docs/theory.md)** - Mathematical foundations
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## 🧪 Example Systems

QBES includes pre-configured templates for common biological systems:

### Photosynthetic Complexes
```bash
qbes generate-config photosystem.yaml --template photosystem
# Simulates quantum coherence in light-harvesting complexes
```

### Enzyme Active Sites
```bash
qbes generate-config enzyme.yaml --template enzyme
# Models quantum tunneling in enzymatic reactions
```

### Membrane Proteins
```bash
qbes generate-config membrane.yaml --template membrane
# Studies quantum effects in membrane-bound systems
```

## 🔧 Command-Line Interface

QBES provides a comprehensive CLI with the following commands:

### Simulation Management
- `qbes run <config>` - Run a simulation
- `qbes resume <checkpoint>` - Resume from checkpoint
- `qbes status <output_dir>` - Check simulation status
- `qbes monitor-sim <output_dir>` - Real-time monitoring

### Configuration
- `qbes generate-config <file>` - Generate configuration template
- `qbes validate <config>` - Validate configuration file

### Benchmarking
- `qbes benchmark run-benchmarks` - Run validation suite
- `qbes benchmark analyze <results_dir>` - Analyze benchmark results

Run `qbes --help` or `qbes <command> --help` for detailed usage information.

## 🧬 Supported Biological Systems

QBES is designed to handle a wide range of biological systems:

- **Photosynthetic Complexes**: Light-harvesting complexes, reaction centers
- **Enzyme Active Sites**: Catalytic mechanisms, tunneling effects
- **DNA/RNA**: Base pair dynamics, charge transfer
- **Membrane Proteins**: Ion channels, transporters
- **Protein-Protein Interactions**: Binding dynamics, allosteric effects

## 📊 Validation and Benchmarks

QBES includes extensive validation against:

- **Analytical Solutions**: Simple quantum systems with known results
- **Literature Benchmarks**: Published experimental and theoretical data
- **Cross-Validation**: Comparison with other simulation packages
- **Performance Tests**: Computational scaling and efficiency

Run the benchmark suite to verify your installation:

```bash
qbes benchmark run-benchmarks --verbose
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Documentation guidelines
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use QBES in your research, please cite:

```bibtex
@software{qbes2024,
  title={Quantum Biological Environment Simulator (QBES)},
  author={[Authors]},
  year={2024},
  url={https://github.com/your-org/qbes},
  version={0.1.0}
}
```

## 🆘 Support

- **Documentation**: Check the [`docs/`](docs/) directory
- **Issues**: Report bugs via [GitHub Issues](https://github.com/your-org/qbes/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/your-org/qbes/discussions)
- **Email**: Contact us at qbes-support@example.com

## 📚 Educational Resources

### Interactive Learning Website
For educational purposes, we provide an interactive website that explains quantum biology concepts and demonstrates the QBES project:

```bash
# Launch educational website
python start_website.py
```

**Note**: The educational website is separate from this research software and is intended for learning purposes only.

## 🙏 Acknowledgments

QBES development is supported by:
- [Funding agencies]
- [Collaborating institutions]
- [Open source community]

Special thanks to the developers of QuTiP, OpenMM, and other scientific Python libraries that make this work possible.