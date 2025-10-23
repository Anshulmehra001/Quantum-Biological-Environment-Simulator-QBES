# QBES Example Systems

This directory contains example molecular systems and configurations for learning and testing QBES functionality. Each example includes:

- PDB structure file
- Configuration file
- Expected results
- Analysis script
- Documentation

## Available Examples

### 1. Water Box (`water_box/`)
**Difficulty:** Beginner  
**System:** 10 water molecules in a periodic box  
**Purpose:** Learn basic QBES concepts and quantum coherence in simple systems  
**Runtime:** ~2 minutes  

### 2. Benzene Ring (`benzene/`)
**Difficulty:** Beginner  
**System:** Single benzene molecule in vacuum  
**Purpose:** Understand aromatic π-electron systems and quantum delocalization  
**Runtime:** ~5 minutes  

### 3. Photosystem II (`photosystem_ii/`)
**Difficulty:** Intermediate  
**System:** Simplified PSII reaction center with chlorophyll molecules  
**Purpose:** Study quantum coherence in photosynthetic energy transfer  
**Runtime:** ~30 minutes  

### 4. Enzyme Active Site (`enzyme_catalysis/`)
**Difficulty:** Intermediate  
**System:** Simplified enzyme active site with substrate  
**Purpose:** Explore quantum tunneling in enzymatic reactions  
**Runtime:** ~20 minutes  

### 5. DNA Base Pair (`dna_base_pair/`)
**Difficulty:** Advanced  
**System:** Adenine-thymine base pair with surrounding water  
**Purpose:** Investigate quantum effects in DNA dynamics  
**Runtime:** ~45 minutes  

## Quick Start

1. **Choose an example:**
   ```bash
   cd examples/water_box/
   ```

2. **Run the simulation:**
   ```bash
   qbes run config.yaml --verbose
   ```

3. **Analyze results:**
   ```bash
   python analyze_results.py
   ```

4. **View plots:**
   ```bash
   ls plots/
   ```

## Example Structure

Each example directory contains:

```
example_name/
├── README.md              # Detailed description and instructions
├── system.pdb             # Molecular structure file
├── config.yaml            # QBES configuration
├── expected_results.json  # Reference results for validation
├── analyze_results.py     # Custom analysis script
└── plots/                 # Generated plots and figures
    ├── coherence_evolution.png
    ├── population_dynamics.png
    └── energy_analysis.png
```

## Educational Progression

**For Beginners:**
1. Start with `water_box/` to learn basic concepts
2. Try `benzene/` to understand aromatic systems
3. Read the tutorial and user guide

**For Intermediate Users:**
1. Explore `photosystem_ii/` for biological relevance
2. Study `enzyme_catalysis/` for reaction dynamics
3. Modify configurations to see parameter effects

**For Advanced Users:**
1. Analyze `dna_base_pair/` for complex systems
2. Create your own examples based on these templates
3. Contribute new examples to the community

## Validation

All examples include expected results for validation:

```bash
# Run example and compare with expected results
cd examples/water_box/
qbes run config.yaml
python validate_results.py
```

Expected output:
```
✅ Coherence lifetime: 245.7 fs (expected: 245.7 ± 25.0 fs)
✅ Final purity: 0.234 (expected: 0.234 ± 0.05)
✅ Energy conservation: 0.007% (expected: < 1.0%)
```

## Creating Your Own Examples

Use the template in `template_example/` to create new examples:

1. Copy the template directory
2. Replace the PDB file with your system
3. Modify the configuration file
4. Run the simulation and generate expected results
5. Create analysis scripts and documentation

## Troubleshooting

**Issue:** Simulation fails with memory error  
**Solution:** Use smaller systems or reduce `max_quantum_atoms`

**Issue:** Results don't match expected values  
**Solution:** Check system requirements and dependency versions

**Issue:** Plots not generated  
**Solution:** Ensure matplotlib is installed and display is available

## Contributing Examples

We welcome contributions of new example systems! Please:

1. Follow the standard directory structure
2. Include comprehensive documentation
3. Validate results on multiple systems
4. Submit via GitHub pull request

## Support

- Check individual example README files for specific help
- See the main QBES documentation for general issues
- Ask questions in GitHub Discussions
- Report bugs in GitHub Issues