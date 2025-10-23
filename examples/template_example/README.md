# Template Example

This directory provides a template for creating new QBES examples. Copy this directory and modify the files to create your own example system.

## Directory Structure

```
template_example/
├── README.md              # This file - describe your system
├── system.pdb             # Molecular structure file
├── config.yaml            # QBES configuration
├── expected_results.json  # Reference results for validation
├── analyze_results.py     # Custom analysis script
└── validate_results.py    # Results validation script
```

## Creating a New Example

### Step 1: Copy Template

```bash
cp -r examples/template_example examples/my_new_example
cd examples/my_new_example
```

### Step 2: Modify Files

1. **Replace `system.pdb`** with your molecular structure
2. **Edit `config.yaml`** for your system parameters
3. **Update `README.md`** with system description
4. **Modify analysis scripts** as needed

### Step 3: Generate Expected Results

```bash
# Run simulation to generate reference data
qbes run config.yaml

# Copy results to expected_results.json
cp results/simulation_results.json expected_results.json

# Edit expected_results.json to add tolerances and descriptions
```

### Step 4: Test and Document

```bash
# Test the complete workflow
qbes run config.yaml
python analyze_results.py
python validate_results.py

# Update README.md with findings
```

## Template Configuration

The template `config.yaml` includes:
- Basic system setup
- Standard simulation parameters
- Common analysis options
- Typical output settings

Modify these sections for your specific system:

### System Section
```yaml
system:
  pdb_file: "your_system.pdb"          # Your PDB file
  force_field: "appropriate_ff"        # Choose appropriate force field
  # ... other system parameters
```

### Quantum Subsystem
```yaml
quantum_subsystem:
  selection_method: "custom"           # Adjust selection method
  custom_selection: "your_selection"   # Define your quantum region
  max_quantum_atoms: N                 # Set appropriate size
```

### Noise Model
```yaml
noise_model:
  type: "appropriate_type"             # Choose noise model for your system
  # ... adjust parameters for your environment
```

## Analysis Script Template

The `analyze_results.py` template includes:
- Basic result loading
- Standard coherence analysis
- Energy conservation checks
- Plot generation
- Comparison with expected results

Customize for your system:
- Add system-specific analyses
- Create custom plots
- Include relevant physical interpretations

## Validation Script Template

The `validate_results.py` template provides:
- Automated result validation
- Tolerance checking
- Quality assessment
- Detailed reporting

## Documentation Guidelines

Your README.md should include:

### Header Information
- Difficulty level (Beginner/Intermediate/Advanced)
- System description (brief)
- Scientific purpose
- Estimated runtime

### Scientific Background
- Physical/chemical context
- Relevant quantum effects
- Biological significance (if applicable)

### System Details
- Molecular composition
- Quantum subsystem definition
- Environmental conditions
- Key parameters

### Expected Results
- Key metrics with physical interpretation
- Comparison with experimental data (if available)
- Literature references

### Learning Objectives
- What users should learn
- Key concepts demonstrated
- Skills developed

### Exercises
- Parameter variations to try
- Questions to explore
- Extensions to consider

## Best Practices

### System Selection
- Start with well-characterized systems
- Choose appropriate size (10-200 atoms)
- Ensure physical relevance
- Consider computational cost

### Configuration
- Use realistic parameters
- Include appropriate error checking
- Document parameter choices
- Test different conditions

### Analysis
- Focus on key physical quantities
- Provide clear interpretations
- Include uncertainty estimates
- Compare with known results

### Documentation
- Write for your target audience
- Include clear instructions
- Provide troubleshooting tips
- Add relevant references

## Contributing Examples

To contribute your example to QBES:

1. **Test thoroughly** on multiple systems
2. **Follow the template structure**
3. **Include comprehensive documentation**
4. **Validate against known results**
5. **Submit via GitHub pull request**

## Support

For help creating examples:
- Check existing examples for guidance
- Ask questions in GitHub Discussions
- Review the QBES documentation
- Contact the development team