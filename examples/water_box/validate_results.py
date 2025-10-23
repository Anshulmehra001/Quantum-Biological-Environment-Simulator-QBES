#!/usr/bin/env python3
"""
Water Box Example - Results Validation Script

This script validates simulation results against expected values
and provides detailed feedback on simulation quality.
"""

import json
import numpy as np
from pathlib import Path
import sys

def load_results():
    """Load simulation and expected results."""
    # Load simulation results
    sim_file = Path("results/simulation_results.json")
    if sim_file.exists():
        with open(sim_file, 'r') as f:
            sim_results = json.load(f)
    else:
        print("❌ Simulation results not found. Run the simulation first:")
        print("   qbes run config.yaml")
        return None, None
    
    # Load expected results
    exp_file = Path("expected_results.json")
    if exp_file.exists():
        with open(exp_file, 'r') as f:
            exp_results = json.load(f)
    else:
        print("❌ Expected results file not found.")
        return sim_results, None
    
    return sim_results, exp_results

def validate_metric(name, calculated, expected_data, unit=""):
    """Validate a single metric against expected value."""
    expected_value = expected_data['value']
    tolerance = expected_data['tolerance']
    description = expected_data.get('description', '')
    
    # Calculate relative error
    if expected_value != 0:
        rel_error = abs(calculated - expected_value) / abs(expected_value)
        abs_error = abs(calculated - expected_value)
    else:
        rel_error = abs(calculated)
        abs_error = abs(calculated)
    
    # Check if within tolerance
    within_tolerance = abs_error <= tolerance
    
    # Format output
    status = "✅" if within_tolerance else "❌"
    
    if unit:
        print(f"{status} {name}: {calculated:.3g} {unit}")
        print(f"   Expected: {expected_value:.3g} ± {tolerance:.3g} {unit}")
    else:
        print(f"{status} {name}: {calculated:.3g}")
        print(f"   Expected: {expected_value:.3g} ± {tolerance:.3g}")
    
    if not within_tolerance:
        print(f"   ⚠️  Error: {abs_error:.3g} (tolerance: {tolerance:.3g})")
        if rel_error > 0.1:
            print(f"   ⚠️  Large relative error: {rel_error*100:.1f}%")
    
    if description:
        print(f"   📝 {description}")
    
    print()
    
    return within_tolerance

def validate_coherence_analysis(sim_results, exp_results):
    """Validate quantum coherence analysis results."""
    print("🔬 QUANTUM COHERENCE VALIDATION")
    print("-" * 40)
    
    if exp_results is None:
        print("⚠️  No expected results available")
        return True
    
    coherence_exp = exp_results.get('quantum_coherence', {})
    passed_tests = []
    
    # Extract coherence results from simulation
    # Note: This assumes the simulation results contain these fields
    # In practice, you might need to calculate these from time-series data
    
    # Coherence lifetime
    if 'coherence_lifetime' in sim_results:
        lifetime_calc = sim_results['coherence_lifetime']
        if 'coherence_lifetime' in coherence_exp:
            passed = validate_metric(
                "Coherence Lifetime", 
                lifetime_calc, 
                coherence_exp['coherence_lifetime'], 
                "s"
            )
            passed_tests.append(passed)
    
    # Decoherence rate
    if 'decoherence_rate' in sim_results:
        rate_calc = sim_results['decoherence_rate']
        if 'decoherence_rate' in coherence_exp:
            passed = validate_metric(
                "Decoherence Rate", 
                rate_calc, 
                coherence_exp['decoherence_rate'], 
                "Hz"
            )
            passed_tests.append(passed)
    
    # Final purity
    if 'final_purity' in sim_results:
        purity_calc = sim_results['final_purity']
        if 'final_purity' in coherence_exp:
            passed = validate_metric(
                "Final Purity", 
                purity_calc, 
                coherence_exp['final_purity']
            )
            passed_tests.append(passed)
    
    return all(passed_tests) if passed_tests else True

def validate_energy_analysis(sim_results, exp_results):
    """Validate energy conservation and thermodynamics."""
    print("⚡ ENERGY ANALYSIS VALIDATION")
    print("-" * 40)
    
    if exp_results is None:
        print("⚠️  No expected results available")
        return True
    
    energy_exp = exp_results.get('energy_analysis', {})
    passed_tests = []
    
    # Energy conservation error
    if 'energy_conservation_error' in sim_results:
        error_calc = sim_results['energy_conservation_error']
        if 'energy_conservation_error' in energy_exp:
            passed = validate_metric(
                "Energy Conservation Error", 
                error_calc, 
                energy_exp['energy_conservation_error']
            )
            passed_tests.append(passed)
    
    # Initial energy
    if 'initial_energy' in sim_results:
        energy_calc = sim_results['initial_energy']
        if 'initial_energy' in energy_exp:
            passed = validate_metric(
                "Initial Energy", 
                energy_calc, 
                energy_exp['initial_energy'], 
                "kcal/mol"
            )
            passed_tests.append(passed)
    
    # Final energy
    if 'final_energy' in sim_results:
        energy_calc = sim_results['final_energy']
        if 'final_energy' in energy_exp:
            passed = validate_metric(
                "Final Energy", 
                energy_calc, 
                energy_exp['final_energy'], 
                "kcal/mol"
            )
            passed_tests.append(passed)
    
    return all(passed_tests) if passed_tests else True

def validate_simulation_quality(sim_results, exp_results):
    """Validate overall simulation quality metrics."""
    print("🎯 SIMULATION QUALITY VALIDATION")
    print("-" * 40)
    
    passed_tests = []
    
    # Check basic simulation completion
    if 'simulation_completed' in sim_results:
        completed = sim_results['simulation_completed']
        status = "✅" if completed else "❌"
        print(f"{status} Simulation Completion: {completed}")
        passed_tests.append(completed)
    
    # Check timesteps completed
    if 'timesteps_completed' in sim_results:
        timesteps = sim_results['timesteps_completed']
        expected_timesteps = 1000  # Based on config: 1ps / 1fs = 1000 steps
        
        if timesteps >= expected_timesteps * 0.9:  # Allow 10% tolerance
            print(f"✅ Timesteps Completed: {timesteps} (expected: ~{expected_timesteps})")
            passed_tests.append(True)
        else:
            print(f"❌ Timesteps Completed: {timesteps} (expected: ~{expected_timesteps})")
            print("   ⚠️  Simulation may have terminated early")
            passed_tests.append(False)
    
    # Check numerical stability
    if 'numerical_stability' in sim_results:
        stability = sim_results['numerical_stability']
        stable = stability in ['good', 'excellent']
        status = "✅" if stable else "❌"
        print(f"{status} Numerical Stability: {stability}")
        passed_tests.append(stable)
    
    print()
    return all(passed_tests) if passed_tests else True

def check_file_outputs():
    """Check that expected output files were generated."""
    print("📁 OUTPUT FILES VALIDATION")
    print("-" * 40)
    
    results_dir = Path("results")
    expected_files = [
        "simulation_results.json",
        "time_evolution_data.csv",
        "results_summary.txt"
    ]
    
    optional_files = [
        "analysis_report.txt",
        "plots/coherence_evolution.png",
        "plots/population_dynamics.png",
        "plots/energy_conservation.png"
    ]
    
    all_present = True
    
    # Check required files
    for filename in expected_files:
        filepath = results_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename} ({size} bytes)")
        else:
            print(f"❌ {filename} (missing)")
            all_present = False
    
    # Check optional files
    for filename in optional_files:
        filepath = results_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"✅ {filename} ({size} bytes)")
        else:
            print(f"⚪ {filename} (optional, not found)")
    
    print()
    return all_present

def generate_validation_report(coherence_valid, energy_valid, quality_valid, files_valid):
    """Generate overall validation report."""
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    validations = [
        ("Quantum Coherence Analysis", coherence_valid),
        ("Energy Conservation", energy_valid),
        ("Simulation Quality", quality_valid),
        ("Output Files", files_valid)
    ]
    
    passed_count = sum(1 for _, valid in validations if valid)
    total_count = len(validations)
    
    for name, valid in validations:
        status = "✅ PASS" if valid else "❌ FAIL"
        print(f"{status} {name}")
    
    print(f"\nOverall Result: {passed_count}/{total_count} validations passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("Your water box simulation is working correctly.")
        print("\nNext steps:")
        print("• Try modifying parameters in config.yaml")
        print("• Run the benzene example for aromatic systems")
        print("• Explore the photosystem example for biological systems")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} VALIDATION(S) FAILED")
        print("Possible issues:")
        print("• Check system requirements and dependencies")
        print("• Verify QBES installation with: qbes info")
        print("• Try running with different parameters")
        print("• Check the troubleshooting guide")
        return False

def main():
    """Main validation function."""
    print("Water Box Example - Results Validation")
    print("=" * 50)
    
    # Load results
    sim_results, exp_results = load_results()
    if sim_results is None:
        return False
    
    # Perform validations
    coherence_valid = validate_coherence_analysis(sim_results, exp_results)
    energy_valid = validate_energy_analysis(sim_results, exp_results)
    quality_valid = validate_simulation_quality(sim_results, exp_results)
    files_valid = check_file_outputs()
    
    # Generate report
    overall_valid = generate_validation_report(
        coherence_valid, energy_valid, quality_valid, files_valid
    )
    
    # Exit with appropriate code
    sys.exit(0 if overall_valid else 1)

if __name__ == "__main__":
    main()