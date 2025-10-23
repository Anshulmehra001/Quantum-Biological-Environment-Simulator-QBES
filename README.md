# QBES - Quantum Biological Environment Simulator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Version](https://img.shields.io/badge/version-1.2.0--dev-orange.svg)]()
[![Status](https://img.shields.io/badge/status-development-yellow.svg)]()

**Quantum biology simulation platform - Development Version**

QBES is a quantum biological environment simulator designed to study quantum mechanical effects in biological systems. This is a **development version** created as an academic project, combining quantum mechanics principles with biological system modeling for research and educational purposes.

⚠️ **Development Status**: This software is currently in active development and testing phase. While functional, it should be considered experimental and not yet ready for production research use.

## Quick Start

### Installation
```bash
git clone https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-
cd QBES
pip install -e .
```

### Verify Installation
```bash
python -c "import qbes; print('QBES version:', qbes.__version__)"
```

### Run Interactive Demo
```bash
python qbes_interactive.py
# Choose: 1 (Quick Start) → 1 (Photosynthesis) → y (Run)
```

### Your First Simulation
```bash
# Generate configuration
python -m qbes.cli generate-config my_simulation.yaml --template photosystem

# Run simulation
python -m qbes.cli run my_simulation.yaml --verbose

# View results
python -m qbes.cli view ./photosystem_output
```

## Key Features

- **Quantum Mechanics**: Implementation of open quantum systems theory (Lindblad equations)
- **Biological Modeling**: Basic protein and membrane environment models
- **Interactive Interface**: User-friendly menu-driven interface
- **Educational Focus**: Designed for learning quantum biology concepts
- **Development Platform**: Foundation for future quantum biology research tools

## Development Status

This is an **academic development project** with the following current capabilities:
- ✅ Basic quantum system simulation
- ✅ Simple biological environment models
- ✅ Interactive user interface
- ✅ Demonstration examples
- 🔄 **In Development**: Advanced validation and benchmarking
- 🔄 **In Development**: Production-level accuracy verification
- 🔄 **Future**: Full scientific research capabilities

## Applications

- **Photosynthesis Research**: Quantum coherence in light-harvesting complexes
- **Enzyme Catalysis**: Quantum tunneling effects in biological reactions
- **Drug Discovery**: Quantum effects in protein-drug interactions
- **Membrane Biology**: Quantum phenomena in ion channels and transporters

## Documentation

- **[Installation Guide](docs/guides/installation.md)** - Setup instructions
- **[Getting Started](docs/guides/getting-started.md)** - Quick start tutorial
- **[User Guide](docs/guides/user-guide.md)** - Complete usage documentation
- **[API Reference](docs/technical/api-reference.md)** - Programming interface
- **[Theory](docs/technical/theory.md)** - Mathematical foundations
- **[Examples](docs/examples/)** - Practical simulation examples

## System Requirements

- **Python**: 3.8+ (3.9+ recommended)
- **Memory**: 8GB+ (16GB+ recommended)
- **Storage**: 2GB+ free space
- **OS**: Windows, macOS, or Linux

## License

QBES is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0):

- ✅ **Share**: Copy and redistribute the material in any medium or format
- ✅ **Adapt**: Remix, transform, and build upon the material
- ✅ **Attribution**: Give appropriate credit with proper citation
- ✅ **Academic Use**: Permitted for educational and research purposes
- ❌ **Commercial Use**: Prohibited without explicit written permission
- 🔄 **ShareAlike**: Derivative works must use the same license

For commercial licensing inquiries, please contact aniketmehra715@gmail.com.

See [LICENSE](LICENSE) file for complete terms and conditions.

## Citation

If you use QBES in your research, please cite:

```bibtex
@software{qbes2025,
  title={QBES: Quantum Biological Environment Simulator},
  author={Aniket Mehra},
  year={2025},
  url={https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-},
  version={1.2.0-dev},
  license={CC BY-NC-SA 4.0},
  note={Development version - Academic project}
}
```

## Support

- **Documentation**: Complete guides in [docs/](docs/) directory
- **Examples**: Practical tutorials in [docs/examples/](docs/examples/)
- **Issues**: Contact aniketmehra715@gmail.com for technical support

## About the Developer

**Aniket Mehra**  
Student at LNCT University, India  
Email: aniketmehra715@gmail.com  

This project was developed independently as an academic exploration of quantum biology simulation. The university did not provide direct support for this project.

---

**QBES v1.2.0-dev** - Quantum biology simulation platform (Development Version)  
Copyright © 2025 Aniket Mehra. Licensed under CC BY-NC-SA 4.0.