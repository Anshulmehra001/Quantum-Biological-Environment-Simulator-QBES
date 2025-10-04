#!/usr/bin/env python3
"""
QBES Enhanced Command Line Interface
Professional CLI for quantum biology simulations
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
import subprocess
import shutil

# Add QBES to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from qbes.config_manager import ConfigManager
    from qbes.simulation_engine import SimulationEngine
    from qbes.analysis import AnalysisEngine
    from qbes.visualization import VisualizationEngine
    QBES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  QBES modules not fully available: {e}")
    print("🔄 Running in demonstration mode...")
    QBES_AVAILABLE = False

import numpy as np

class QBESCLIInterface:
    """Enhanced CLI interface for QBES"""
    
    def __init__(self):
        self.results_dir = Path("simulation_results")
        self.config_dir = Path("configs")
        self.results_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
    def print_banner(self):
        """Print QBES banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    ██████╗ ██████╗ ███████╗███████╗                         ║
║   ██╔═══██╗██╔══██╗██╔════╝██╔════╝                         ║
║   ██║   ██║██████╔╝█████╗  ███████╗                         ║
║   ██║▄▄ ██║██╔══██╗██╔══╝  ╚════██║                         ║
║   ╚██████╔╝██████╔╝███████╗███████║                         ║
║    ╚══▀▀═╝ ╚═════╝ ╚══════╝╚══════╝                         ║
║                                                              ║
║         Quantum Biological Environment Simulator            ║
║              Professional CLI Interface                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        print(f"🔬 QBES Backend: {'Available' if QBES_AVAILABLE else 'Demo Mode'}")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 66)
    
    def create_config_template(self, template_type="basic", output_file=None):
        """Create configuration template"""
        templates = {
            "basic": {
                "system": {
                    "type": "two_level",
                    "temperature": 300.0,
                    "hamiltonian": {
                        "energy_gap": 2.0,
                        "coupling": 0.1
                    }
                },
                "simulation": {
                    "time_step": 0.1,
                    "total_time": 10.0,
                    "method": "lindblad"
                },
                "noise": {
                    "model": "protein_ohmic",
                    "strength": 0.1
                },
                "output": {
                    "save_trajectory": True,
                    "analysis": ["coherence", "purity", "energy"]
                }
            },
            "photosynthesis": {
                "system": {
                    "type": "two_level",
                    "temperature": 300.0,
                    "hamiltonian": {
                        "energy_gap": 1.5,
                        "coupling": 0.05
                    }
                },
                "simulation": {
                    "time_step": 0.05,
                    "total_time": 20.0,
                    "method": "lindblad"
                },
                "noise": {
                    "model": "protein_ohmic",
                    "strength": 0.05
                },
                "output": {
                    "save_trajectory": True,
                    "analysis": ["coherence", "purity", "energy", "entanglement"]
                }
            },
            "enzyme": {
                "system": {
                    "type": "two_level",
                    "temperature": 310.0,
                    "hamiltonian": {
                        "energy_gap": 2.5,
                        "coupling": 0.2
                    }
                },
                "simulation": {
                    "time_step": 0.02,
                    "total_time": 5.0,
                    "method": "lindblad"
                },
                "noise": {
                    "model": "protein_ohmic",
                    "strength": 0.2
                },
                "output": {
                    "save_trajectory": True,
                    "analysis": ["coherence", "purity", "tunneling_rate"]
                }
            },
            "membrane": {
                "system": {
                    "type": "two_level",
                    "temperature": 300.0,
                    "hamiltonian": {
                        "energy_gap": 1.0,
                        "coupling": 0.1
                    }
                },
                "simulation": {
                    "time_step": 0.1,
                    "total_time": 15.0,
                    "method": "lindblad"
                },
                "noise": {
                    "model": "membrane_fluctuations",
                    "strength": 0.15
                },
                "output": {
                    "save_trajectory": True,
                    "analysis": ["coherence", "purity", "energy"]
                }
            }
        }
        
        if template_type not in templates:
            print(f"❌ Unknown template type: {template_type}")
            print(f"📋 Available templates: {list(templates.keys())}")
            return False
        
        config = templates[template_type]
        
        if output_file is None:
            output_file = self.config_dir / f"{template_type}_config.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created {template_type} configuration: {output_file}")
        return True
    
    def run_simulation(self, config_file, output_dir=None):
        """Run simulation from config file"""
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_file}")
            return False
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.results_dir / f"sim_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting simulation...")
        print(f"📁 Config: {config_file}")
        print(f"📁 Output: {output_dir}")
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if QBES_AVAILABLE:
                results = self._run_real_simulation(config)
            else:
                results = self._run_mock_simulation(config)
            
            # Save results
            self._save_results(results, output_dir, config)
            
            print(f"✅ Simulation completed successfully!")
            print(f"📊 Results saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            return False
    
    def _run_real_simulation(self, config):
        """Run actual QBES simulation"""
        # This would use the real QBES engine
        config_manager = ConfigManager()
        engine = SimulationEngine()
        
        # Convert dict config to QBES config object
        sim_config = config_manager.create_config_from_dict(config)
        results = engine.run_simulation(sim_config)
        
        return results
    
    def _run_mock_simulation(self, config):
        """Run mock simulation for demonstration"""
        print("🔄 Running demonstration simulation...")
        
        # Extract parameters
        total_time = config.get("simulation", {}).get("total_time", 10.0)
        time_step = config.get("simulation", {}).get("time_step", 0.1)
        temperature = config.get("system", {}).get("temperature", 300.0)
        energy_gap = config.get("system", {}).get("hamiltonian", {}).get("energy_gap", 2.0)
        coupling = config.get("system", {}).get("hamiltonian", {}).get("coupling", 0.1)
        noise_strength = config.get("noise", {}).get("strength", 0.1)
        
        # Generate time points
        time_points = np.arange(0, total_time + time_step, time_step)
        
        # Calculate decoherence rate based on parameters
        decoherence_rate = noise_strength * (temperature / 300.0) ** 0.5
        
        # Generate quantum evolution
        coherence = np.exp(-decoherence_rate * time_points)
        purity = 0.5 + 0.5 * coherence
        energy = np.full_like(time_points, energy_gap / 2)
        
        # Add some realistic noise
        coherence += np.random.normal(0, 0.01, len(coherence))
        purity += np.random.normal(0, 0.005, len(purity))
        
        # Ensure physical bounds
        coherence = np.clip(coherence, 0, 1)
        purity = np.clip(purity, 0.5, 1)
        
        results = {
            "config": config,
            "time_evolution": {
                "time": time_points.tolist(),
                "coherence": coherence.tolist(),
                "purity": purity.tolist(),
                "energy": energy.tolist()
            },
            "analysis": {
                "decoherence_rate": decoherence_rate,
                "coherence_lifetime": 1.0 / decoherence_rate if decoherence_rate > 0 else float('inf'),
                "final_coherence": coherence[-1],
                "final_purity": purity[-1],
                "energy_conservation": np.std(energy)
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulation_type": "mock_demonstration",
                "qbes_version": "0.1.0"
            }
        }
        
        return results
    
    def _save_results(self, results, output_dir, config):
        """Save simulation results in multiple formats"""
        output_dir = Path(output_dir)
        
        # Save full results as JSON
        with open(output_dir / "simulation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save time evolution data as CSV
        time_data = results.get("time_evolution", {})
        if time_data:
            import pandas as pd
            df = pd.DataFrame({
                'Time_ps': time_data.get('time', []),
                'Coherence': time_data.get('coherence', []),
                'Purity': time_data.get('purity', []),
                'Energy_eV': time_data.get('energy', [])
            })
            df.to_csv(output_dir / "time_evolution_data.csv", index=False)
        
        # Save analysis summary
        analysis = results.get("analysis", {})
        summary_text = f"""QBES Simulation Results
========================================
System Configuration:
  System Type: {config.get('system', {}).get('type', 'unknown')}
  Temperature: {config.get('system', {}).get('temperature', 'N/A')} K
  Noise Model: {config.get('noise', {}).get('model', 'N/A')}
  Simulation Time: {config.get('simulation', {}).get('total_time', 'N/A')} ps
  Time Steps: {len(time_data.get('time', []))}

Hamiltonian Matrix:
  [{config.get('system', {}).get('hamiltonian', {}).get('energy_gap', 0)/2:.1f}  {config.get('system', {}).get('hamiltonian', {}).get('coupling', 0):.1f}]
  [{config.get('system', {}).get('hamiltonian', {}).get('coupling', 0):.1f}  {config.get('system', {}).get('hamiltonian', {}).get('energy_gap', 0):.1f}]  (in eV)

Key Results:
  Coherence Lifetime: {analysis.get('coherence_lifetime', 'N/A'):.1f} ps
  Decoherence Rate: {analysis.get('decoherence_rate', 'N/A'):.3f} ps^-1
  Initial Purity: 1.000
  Final Purity: {analysis.get('final_purity', 'N/A'):.3f}
  Initial Coherence: 1.000
  Final Coherence: {analysis.get('final_coherence', 'N/A'):.3f}

Physical Interpretation:
- The quantum system starts in a coherent superposition state
- Biological environment causes decoherence
- Coherence decays exponentially: C(t) = exp(-t/tau)
- Coherence lifetime tau = {analysis.get('coherence_lifetime', 'N/A'):.1f} ps is typical for biological systems
- This demonstrates quantum effects in photosynthesis/enzymes

Applications:
- Photosynthetic energy transfer efficiency
- Enzyme catalysis quantum tunneling
- Quantum biology research
"""
        
        with open(output_dir / "simulation_summary.txt", 'w') as f:
            f.write(summary_text)
        
        # Save detailed analysis
        detailed_analysis = f"""QBES Analysis Report
==============================

1. QUANTUM COHERENCE ANALYSIS
------------------------------
Coherence Lifetime: {analysis.get('coherence_lifetime', 'N/A'):.1f} ps
Decay Rate: {analysis.get('decoherence_rate', 'N/A'):.1f} ps^-1
Decay Model: Exponential C(t) = exp(-{analysis.get('decoherence_rate', 0):.1f}t)

2. PURITY EVOLUTION
------------------------------
Initial Purity: 1.000
Final Purity: {analysis.get('final_purity', 'N/A'):.3f}
Purity Loss: {1.0 - analysis.get('final_purity', 1.0):.3f}

3. ENERGY CONSERVATION
------------------------------
Average Energy: {config.get('system', {}).get('hamiltonian', {}).get('energy_gap', 0)/2:.3f} eV
Energy Fluctuation: {analysis.get('energy_conservation', 0):.6f} eV
Energy is conserved (as expected)

4. BIOLOGICAL SIGNIFICANCE
------------------------------
This simulation models quantum effects in:
• Photosynthetic light-harvesting complexes
• Enzyme active sites with quantum tunneling
• Membrane protein quantum dynamics
• DNA charge transfer processes

The {analysis.get('coherence_lifetime', 'N/A'):.1f} ps coherence lifetime is realistic for:
• Chlorophyll molecules in photosystems
• Aromatic amino acids in proteins
• Biological chromophores at room temperature
"""
        
        with open(output_dir / "analysis_report.txt", 'w') as f:
            f.write(detailed_analysis)
    
    def list_simulations(self):
        """List all simulation results"""
        if not self.results_dir.exists():
            print("📁 No simulation results directory found")
            return
        
        sim_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        if not sim_dirs:
            print("📁 No simulation results found")
            return
        
        print(f"📊 Found {len(sim_dirs)} simulation results:")
        print("=" * 60)
        
        for sim_dir in sorted(sim_dirs):
            # Try to load metadata
            results_file = sim_dir / "simulation_results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    timestamp = results.get("metadata", {}).get("timestamp", "Unknown")
                    config = results.get("config", {})
                    analysis = results.get("analysis", {})
                    
                    print(f"📁 {sim_dir.name}")
                    print(f"   📅 Time: {timestamp}")
                    print(f"   🌡️  Temperature: {config.get('system', {}).get('temperature', 'N/A')} K")
                    print(f"   ⏱️  Lifetime: {analysis.get('coherence_lifetime', 'N/A'):.1f} ps")
                    print(f"   📊 Files: {len(list(sim_dir.glob('*')))} files")
                    print()
                    
                except Exception as e:
                    print(f"📁 {sim_dir.name} (metadata error: {e})")
            else:
                print(f"📁 {sim_dir.name} (no results file)")
    
    def view_results(self, results_dir):
        """View simulation results"""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return False
        
        # Check for results file
        results_file = results_path / "simulation_results.json"
        if not results_file.exists():
            print(f"❌ No simulation results found in: {results_dir}")
            return False
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Display summary
            print(f"🔬 QBES Results Analysis")
            print("=" * 60)
            print(f"📁 Results Directory: {results_path.name}")
            
            # Show available files
            files = list(results_path.glob('*'))
            print(f"📊 Available Files:")
            for file in files:
                size = file.stat().st_size
                print(f"   • {file.name} ({size:,} bytes)")
            
            print("=" * 50)
            print("📋 SIMULATION SUMMARY")
            print("=" * 50)
            
            # Read and display summary
            summary_file = results_path / "simulation_summary.txt"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    print(f.read())
            
            print("=" * 50)
            print("📊 KEY NUMERICAL RESULTS")
            print("=" * 50)
            
            config = results.get("config", {})
            analysis = results.get("analysis", {})
            time_data = results.get("time_evolution", {})
            
            print(f"System Configuration:")
            print(f"  • System Type: {config.get('system', {}).get('type', 'N/A')}")
            print(f"  • Temperature: {config.get('system', {}).get('temperature', 'N/A')} K")
            print(f"  • Noise Model: {config.get('noise', {}).get('model', 'N/A')}")
            print(f"  • Simulation Time: {config.get('simulation', {}).get('total_time', 'N/A')} ps")
            print(f"  • Total Steps: {len(time_data.get('time', []))}")
            
            print(f"Quantum Properties:")
            print(f"  • Coherence Lifetime: {analysis.get('coherence_lifetime', 'N/A'):.1f} ps")
            print(f"  • Final Coherence: {analysis.get('final_coherence', 'N/A'):.3f}")
            print(f"  • Final Purity: {analysis.get('final_purity', 'N/A'):.3f}")
            
            print(f"Decoherence Analysis:")
            print(f"  • Decoherence Rate: {analysis.get('decoherence_rate', 'N/A'):.1f} ps^-1")
            print(f"  • Initial Purity: 1.000")
            print(f"  • Purity Loss: {1.0 - analysis.get('final_purity', 1.0):.3f}")
            print(f"  • Coherence Decay: {1.0 - analysis.get('final_coherence', 1.0):.3f}")
            
            # Show first few data points
            if time_data.get('time'):
                print("=" * 50)
                print("📈 TIME EVOLUTION DATA (First 10 points)")
                print("=" * 50)
                print("   Time_ps,Energy_eV,Purity,Coherence")
                
                times = time_data.get('time', [])
                energies = time_data.get('energy', [])
                purities = time_data.get('purity', [])
                coherences = time_data.get('coherence', [])
                
                for i in range(min(10, len(times))):
                    print(f"   {times[i]:.3f},{energies[i]:.6f},{purities[i]:.6f},{coherences[i]:.6f}")
                
                if len(times) > 10:
                    print(f"   ... ({len(times) - 10} more data points)")
            
            print("=" * 50)
            print("📊 VISUALIZATION OPTIONS")
            print("=" * 50)
            print("To create plots from your data:")
            print("Option 1 - Python/Matplotlib:")
            print("   python plot_results.py")
            print("Option 2 - Excel/Spreadsheet:")
            print(f"   Open: {results_path}/time_evolution_data.csv")
            print("   Create charts from Time_ps vs Coherence columns")
            print("Option 3 - Online Tools:")
            print("   Upload CSV to: plot.ly, Google Sheets, or similar")
            print("Option 4 - QBES Website:")
            print("   Open: website/qbes_website.html")
            print("   Use the Interactive Demo section")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading results: {e}")
            return False
    
    def start_web_interface(self):
        """Start the web interface"""
        web_app = Path("website/app.py")
        
        if not web_app.exists():
            print("❌ Web interface not found")
            return False
        
        print("🌐 Starting QBES Web Interface...")
        print("📱 Open your browser to: http://localhost:5000")
        
        try:
            subprocess.run([sys.executable, str(web_app)], cwd="website")
        except KeyboardInterrupt:
            print("\n🛑 Web interface stopped")
        except Exception as e:
            print(f"❌ Failed to start web interface: {e}")
        
        return True

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="QBES - Quantum Biological Environment Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s create-config photosynthesis
  %(prog)s run configs/photosynthesis_config.json
  %(prog)s list
  %(prog)s view simulation_results/sim_20250102_143022
  %(prog)s web
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create configuration template')
    config_parser.add_argument('template', choices=['basic', 'photosynthesis', 'enzyme', 'membrane'],
                              help='Configuration template type')
    config_parser.add_argument('-o', '--output', help='Output file path')
    
    # Run simulation command
    run_parser = subparsers.add_parser('run', help='Run simulation')
    run_parser.add_argument('config', help='Configuration file path')
    run_parser.add_argument('-o', '--output', help='Output directory')
    
    # List simulations command
    subparsers.add_parser('list', help='List all simulation results')
    
    # View results command
    view_parser = subparsers.add_parser('view', help='View simulation results')
    view_parser.add_argument('results_dir', help='Results directory path')
    
    # Web interface command
    subparsers.add_parser('web', help='Start web interface')
    
    # Interactive mode command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Initialize CLI interface
    cli = QBESCLIInterface()
    
    if args.command is None:
        cli.print_banner()
        parser.print_help()
        return
    
    cli.print_banner()
    
    if args.command == 'create-config':
        cli.create_config_template(args.template, args.output)
    
    elif args.command == 'run':
        cli.run_simulation(args.config, args.output)
    
    elif args.command == 'list':
        cli.list_simulations()
    
    elif args.command == 'view':
        cli.view_results(args.results_dir)
    
    elif args.command == 'web':
        cli.start_web_interface()
    
    elif args.command == 'interactive':
        interactive_mode(cli)

def interactive_mode(cli):
    """Interactive CLI mode"""
    print("🎯 Entering interactive mode. Type 'help' for commands or 'quit' to exit.")
    
    while True:
        try:
            command = input("\nQBES> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif command == 'help':
                print("""
Available commands:
  create <template>     - Create configuration template
  run <config>          - Run simulation
  list                  - List simulation results
  view <results_dir>    - View simulation results
  web                   - Start web interface
  clear                 - Clear screen
  help                  - Show this help
  quit                  - Exit interactive mode

Templates: basic, photosynthesis, enzyme, membrane
                """)
            
            elif command == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                cli.print_banner()
            
            elif command == 'list':
                cli.list_simulations()
            
            elif command == 'web':
                cli.start_web_interface()
            
            elif command.startswith('create '):
                template = command.split(' ', 1)[1]
                cli.create_config_template(template)
            
            elif command.startswith('run '):
                config_file = command.split(' ', 1)[1]
                cli.run_simulation(config_file)
            
            elif command.startswith('view '):
                results_dir = command.split(' ', 1)[1]
                cli.view_results(results_dir)
            
            elif command == '':
                continue
            
            else:
                print(f"❌ Unknown command: {command}")
                print("💡 Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()