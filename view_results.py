#!/usr/bin/env python3
"""
QBES Results Viewer - View and analyze simulation results
"""

import os
import json
import sys
from pathlib import Path

def view_simulation_results(results_dir="simulation_results"):
    """View QBES simulation results in multiple formats."""
    
    print("🔍 QBES Results Viewer")
    print("=" * 50)
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Run a simulation first or use: python create_sample_results.py")
        return False
    
    print(f"📁 Results Directory: {results_path}")
    print(f"📊 Available Files:")
    
    files = list(results_path.glob("*"))
    for file in files:
        size = file.stat().st_size
        print(f"   • {file.name} ({size:,} bytes)")
    
    print("\n" + "=" * 50)
    
    # 1. Show Summary Report
    summary_file = results_path / "simulation_summary.txt"
    if summary_file.exists():
        print("📋 SIMULATION SUMMARY")
        print("=" * 50)
        with open(summary_file, 'r') as f:
            content = f.read()
        print(content)
        print("=" * 50)
    
    # 2. Show JSON Results (key values)
    json_file = results_path / "simulation_results.json"
    if json_file.exists():
        print("\n📊 KEY NUMERICAL RESULTS")
        print("=" * 50)
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # System info
        if 'simulation_info' in data:
            info = data['simulation_info']
            print("System Configuration:")
            print(f"  • System Type: {info.get('system_type', 'Unknown')}")
            print(f"  • Temperature: {info.get('temperature', 'Unknown')} K")
            print(f"  • Noise Model: {info.get('noise_model', 'Unknown')}")
            print(f"  • Simulation Time: {info.get('simulation_time', 'Unknown')} ps")
            print(f"  • Total Steps: {info.get('total_steps', 'Unknown')}")
        
        # Key results
        if 'results' in data:
            results = data['results']
            print(f"\nQuantum Properties:")
            print(f"  • Coherence Lifetime: {results.get('coherence_lifetime', 'Unknown')} ps")
            print(f"  • Final Coherence: {results.get('final_coherence', 'Unknown'):.3f}")
            print(f"  • Final Purity: {results.get('final_purity', 'Unknown'):.3f}")
        
        # Analysis
        if 'analysis' in data:
            analysis = data['analysis']
            print(f"\nDecoherence Analysis:")
            print(f"  • Decoherence Rate: {analysis.get('decoherence_rate', 'Unknown')} ps^-1")
            print(f"  • Initial Purity: {analysis.get('initial_purity', 'Unknown'):.3f}")
            print(f"  • Purity Loss: {analysis.get('purity_loss', 'Unknown'):.3f}")
            print(f"  • Coherence Decay: {analysis.get('coherence_decay', 'Unknown'):.3f}")
        
        print("=" * 50)
    
    # 3. Show Analysis Report
    analysis_file = results_path / "analysis_report.txt"
    if analysis_file.exists():
        print("\n🔬 DETAILED ANALYSIS")
        print("=" * 50)
        with open(analysis_file, 'r') as f:
            content = f.read()
        print(content)
        print("=" * 50)
    
    # 4. Show CSV Data Preview
    csv_file = results_path / "time_evolution_data.csv"
    if csv_file.exists():
        print("\n📈 TIME EVOLUTION DATA (First 10 points)")
        print("=" * 50)
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Show header and first 10 data points
        for i, line in enumerate(lines[:11]):
            print(f"   {line.strip()}")
        
        if len(lines) > 11:
            print(f"   ... ({len(lines)-11} more data points)")
        
        print("=" * 50)
    
    # 5. Plotting Instructions
    print("\n📊 VISUALIZATION OPTIONS")
    print("=" * 50)
    print("To create plots from your data:")
    print("")
    print("Option 1 - Python/Matplotlib:")
    print("   python plot_results.py")
    print("")
    print("Option 2 - Excel/Spreadsheet:")
    print(f"   Open: {csv_file}")
    print("   Create charts from Time_ps vs Coherence columns")
    print("")
    print("Option 3 - Online Tools:")
    print("   Upload CSV to: plot.ly, Google Sheets, or similar")
    print("")
    print("Option 4 - QBES Website:")
    print("   Open: website/qbes_website.html")
    print("   Use the Interactive Demo section")
    
    print("=" * 50)
    
    # 6. File Locations Summary
    print("\n📁 FILE SUMMARY")
    print("=" * 50)
    print("Your simulation results are saved in multiple formats:")
    print("")
    print("📋 Human-Readable:")
    print(f"   • {summary_file.name} - Main results summary")
    print(f"   • {analysis_file.name} - Detailed scientific analysis")
    print("")
    print("💾 Data Files:")
    print(f"   • {json_file.name} - Complete numerical data (JSON)")
    print(f"   • {csv_file.name} - Time series data (CSV)")
    print("")
    print("🎯 Recommended Reading Order:")
    print("   1. Read simulation_summary.txt first")
    print("   2. Review analysis_report.txt for details")
    print("   3. Use CSV file for plotting/further analysis")
    
    return True

def create_simple_plot():
    """Create a simple text-based plot of the results."""
    
    csv_file = Path("simulation_results/time_evolution_data.csv")
    if not csv_file.exists():
        print("❌ No CSV data found. Run create_sample_results.py first.")
        return
    
    print("\n📊 SIMPLE TEXT PLOT - Coherence vs Time")
    print("=" * 60)
    
    # Read CSV data
    times = []
    coherences = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    for line in lines[::10]:  # Every 10th point
        parts = line.strip().split(',')
        times.append(float(parts[0]))
        coherences.append(float(parts[3]))
    
    # Create simple ASCII plot
    max_coherence = max(coherences)
    width = 50
    
    print("Coherence")
    print("    ^")
    
    for i in range(len(times)):
        t = times[i]
        c = coherences[i]
        bar_length = int((c / max_coherence) * width)
        bar = "█" * bar_length
        print(f"{c:.2f}|{bar}")
    
    print("    " + "-" * (width + 5))
    print(f"    0{'':>10}{times[-1]/2:>10.1f}{'':>10}{times[-1]:>10.1f}")
    print(f"{'':>15}Time (ps)")
    
    print("\n📈 Interpretation:")
    print("   • Coherence starts at 1.0 (perfect quantum coherence)")
    print("   • Decays exponentially due to biological environment")
    print("   • Reaches near zero after ~30 ps (3 lifetimes)")
    print("   • This is typical for quantum effects in biology")

if __name__ == "__main__":
    print("🚀 QBES Results Analysis")
    print("=" * 60)
    
    # Check if results exist
    if not Path("simulation_results").exists():
        print("📊 No results found. Creating sample results...")
        os.system("python create_sample_results.py")
        print()
    
    # View results
    success = view_simulation_results()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("=" * 50)
        print("1. 📖 Read the summary files above")
        print("2. 📊 Create plots using the CSV data")
        print("3. 🌐 Try the interactive website demo")
        print("4. 🔬 Run more simulations with different parameters")
        print("5. 📈 Compare results across different conditions")
        
        # Show simple plot
        create_simple_plot()
        
        print("\n✨ Your quantum biology simulation is complete!")
    else:
        print("\n❌ Could not load results. Please check the files.")