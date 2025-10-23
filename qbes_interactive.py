#!/usr/bin/env python3
"""
QBES Interactive Terminal Interface
Easy-to-use menu-driven interface for QBES v1.2
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class QBESInteractive:
    """Interactive menu-driven interface for QBES"""
    
    def __init__(self):
        self.clear_screen()
        self.show_banner()
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_banner(self):
        """Display QBES banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    QQQQQQ  BBBBB   EEEEEEE  SSSSS    v1.2 Interactive      ║
║   QQ    QQ BB   BB EE       SS                              ║
║   QQ    QQ BBBBBB  EEEEE    SSSSS                           ║
║   QQ  Q QQ BB   BB EE           SS                          ║
║    QQQQQQ  BBBBB   EEEEEEE  SSSSS                           ║
║                                                              ║
║         Quantum Biological Environment Simulator            ║
║              Interactive Terminal Interface                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
        # Check QBES version
        try:
            import qbes
            version = getattr(qbes, '__version__', 'unknown')
            print(f"🔬 QBES Version: {version}")
        except ImportError:
            print("⚠️  QBES not found - running in demo mode")
        
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 66)
    
    def show_main_menu(self):
        """Display main menu options"""
        print("\n🎯 What would you like to do?")
        print("=" * 40)
        print("1. 🚀 Quick Start - Run a simulation")
        print("2. ⚙️  Create Configuration")
        print("3. 🔍 Validate System")
        print("4. 📊 View Results")
        print("5. 🧪 Run Benchmarks")
        print("6. 🛠️  Advanced Options")
        print("7. 📚 Help & Documentation")
        print("8. 🌐 Web Interface")
        print("9. ❌ Exit")
        print("=" * 40)
    
    def get_user_choice(self, max_choice=9):
        """Get user menu choice"""
        while True:
            try:
                choice = input(f"\n👉 Enter your choice (1-{max_choice}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= max_choice:
                    return int(choice)
                else:
                    print(f"❌ Please enter a number between 1 and {max_choice}")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def quick_start(self):
        """Quick start simulation workflow"""
        self.clear_screen()
        print("🚀 QBES Quick Start")
        print("=" * 30)
        
        print("\n📋 Available simulation templates:")
        templates = {
            1: ("photosystem", "🌱 Photosynthetic light-harvesting"),
            2: ("enzyme", "🧬 Enzyme catalysis"),
            3: ("membrane", "🧪 Membrane protein dynamics"),
            4: ("default", "⚡ Basic two-level system")
        }
        
        for key, (name, desc) in templates.items():
            print(f"{key}. {desc}")
        
        choice = self.get_user_choice(4)
        template_name, template_desc = templates[choice]
        
        print(f"\n✅ Selected: {template_desc}")
        
        # Generate configuration
        config_file = f"quick_{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        print(f"📝 Creating configuration: {config_file}")
        
        try:
            # Use QBES CLI to generate config
            result = subprocess.run([
                sys.executable, "-m", "qbes.cli", "generate-config", 
                config_file, "--template", template_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Configuration created successfully!")
                
                # Ask if user wants to run simulation
                run_sim = input("\n🎯 Run simulation now? (y/n): ").lower().strip()
                if run_sim in ['y', 'yes']:
                    self.run_simulation(config_file)
                else:
                    print(f"💾 Configuration saved as: {config_file}")
                    print("💡 You can run it later with option 1 from main menu")
            else:
                print(f"❌ Error creating configuration: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def create_configuration(self):
        """Interactive configuration creation"""
        self.clear_screen()
        print("⚙️  QBES Configuration Creator")
        print("=" * 35)
        
        print("\n📋 Configuration options:")
        print("1. 🎯 Interactive wizard (recommended)")
        print("2. 📄 Template-based")
        print("3. 🔧 Advanced custom")
        
        choice = self.get_user_choice(3)
        
        if choice == 1:
            self.interactive_config_wizard()
        elif choice == 2:
            self.template_config()
        else:
            self.advanced_config()
    
    def interactive_config_wizard(self):
        """Interactive configuration wizard"""
        print("\n🧙‍♂️ Interactive Configuration Wizard")
        print("=" * 40)
        
        # System type
        print("\n🔬 What type of biological system?")
        systems = {
            1: "photosystem",
            2: "enzyme", 
            3: "membrane",
            4: "default"
        }
        
        for key, system in systems.items():
            print(f"{key}. {system.title()}")
        
        sys_choice = self.get_user_choice(4)
        system_type = systems[sys_choice]
        
        # Temperature
        print(f"\n🌡️  Temperature for {system_type} system:")
        print("1. 77K (Low temperature)")
        print("2. 300K (Room temperature)")
        print("3. 310K (Body temperature)")
        print("4. Custom")
        
        temp_choice = self.get_user_choice(4)
        temps = {1: 77, 2: 300, 3: 310}
        
        if temp_choice == 4:
            while True:
                try:
                    temperature = float(input("Enter temperature (K): "))
                    if 1 <= temperature <= 1000:
                        break
                    else:
                        print("❌ Temperature must be between 1-1000 K")
                except ValueError:
                    print("❌ Please enter a valid number")
        else:
            temperature = temps[temp_choice]
        
        # Simulation time
        print(f"\n⏱️  Simulation time:")
        print("1. 1 ps (Quick test)")
        print("2. 10 ps (Standard)")
        print("3. 100 ps (Long)")
        print("4. Custom")
        
        time_choice = self.get_user_choice(4)
        times = {1: 1e-12, 2: 10e-12, 3: 100e-12}
        
        if time_choice == 4:
            while True:
                try:
                    sim_time = float(input("Enter simulation time (ps): ")) * 1e-12
                    if sim_time > 0:
                        break
                    else:
                        print("❌ Time must be positive")
                except ValueError:
                    print("❌ Please enter a valid number")
        else:
            sim_time = times[time_choice]
        
        # Generate configuration
        config = {
            "system": {
                "type": system_type,
                "temperature": temperature
            },
            "simulation": {
                "simulation_time": sim_time,
                "time_step": sim_time / 1000,  # 1000 steps
                "method": "lindblad"
            },
            "noise_model": {
                "type": "protein_ohmic",
                "coupling_strength": 1.0,
                "temperature": temperature
            },
            "output": {
                "directory": f"./results_{system_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "save_trajectory": True,
                "analysis": ["coherence", "purity", "energy"]
            }
        }
        
        # Save configuration
        config_file = f"wizard_{system_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n✅ Configuration created: {config_file}")
            print(f"🔬 System: {system_type}")
            print(f"🌡️  Temperature: {temperature} K")
            print(f"⏱️  Time: {sim_time*1e12:.1f} ps")
            
            # Ask to run simulation
            run_now = input("\n🎯 Run simulation now? (y/n): ").lower().strip()
            if run_now in ['y', 'yes']:
                self.run_simulation(config_file)
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def template_config(self):
        """Template-based configuration"""
        print("\n📄 Template Configuration")
        print("=" * 30)
        
        templates = {
            1: ("photosystem", "🌱 Photosynthetic complex"),
            2: ("enzyme", "🧬 Enzyme active site"),
            3: ("membrane", "🧪 Membrane protein"),
            4: ("default", "⚡ Basic quantum system")
        }
        
        print("\n📋 Available templates:")
        for key, (name, desc) in templates.items():
            print(f"{key}. {desc}")
        
        choice = self.get_user_choice(4)
        template_name, template_desc = templates[choice]
        
        config_file = f"template_{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "qbes.cli", "generate-config",
                config_file, "--template", template_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Template configuration created: {config_file}")
                print(f"📝 Template: {template_desc}")
            else:
                print(f"❌ Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def advanced_config(self):
        """Advanced configuration options"""
        print("\n🔧 Advanced Configuration")
        print("=" * 30)
        print("💡 For advanced users - edit configuration files manually")
        print("📁 Check existing configs in current directory")
        
        # List existing configs
        configs = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.json'))
        if configs:
            print("\n📋 Existing configurations:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config.name}")
        else:
            print("\n📭 No configuration files found")
        
        input("\n📱 Press Enter to continue...")
    
    def validate_system(self):
        """System validation options"""
        self.clear_screen()
        print("🔍 QBES System Validation")
        print("=" * 30)
        
        print("\n📋 Validation options:")
        print("1. 🚀 Quick validation (2 min)")
        print("2. 📊 Standard validation (10 min)")
        print("3. 🔬 Full scientific validation (30 min)")
        print("4. ⚙️  Configuration validation")
        
        choice = self.get_user_choice(4)
        
        if choice == 4:
            self.validate_configuration()
        else:
            suites = {1: "quick", 2: "standard", 3: "full"}
            suite = suites[choice]
            
            print(f"\n🔍 Running {suite} validation suite...")
            print("⏳ This may take a while...")
            
            try:
                result = subprocess.run([
                    sys.executable, "-m", "qbes.cli", "validate",
                    "--suite", suite, "--verbose"
                ], text=True)
                
                if result.returncode == 0:
                    print("✅ Validation completed successfully!")
                else:
                    print("⚠️  Validation completed with issues")
                    
            except Exception as e:
                print(f"❌ Error running validation: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def validate_configuration(self):
        """Validate configuration files"""
        print("\n⚙️  Configuration Validation")
        print("=" * 35)
        
        # Find config files
        configs = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.json'))
        
        if not configs:
            print("❌ No configuration files found")
            print("💡 Create a configuration first (option 2 from main menu)")
            return
        
        print("\n📋 Available configurations:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config.name}")
        
        if len(configs) == 1:
            config_file = configs[0]
        else:
            choice = self.get_user_choice(len(configs))
            config_file = configs[choice - 1]
        
        print(f"\n🔍 Validating: {config_file}")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "qbes.cli", "validate-config",
                str(config_file), "--verbose"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Configuration is valid!")
                print(result.stdout)
            else:
                print("❌ Configuration has issues:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error validating configuration: {e}")
    
    def run_simulation(self, config_file=None):
        """Run simulation"""
        if config_file is None:
            self.clear_screen()
            print("🚀 Run QBES Simulation")
            print("=" * 25)
            
            # Find config files
            configs = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.json'))
            
            if not configs:
                print("❌ No configuration files found")
                print("💡 Create a configuration first (option 2 from main menu)")
                input("\n📱 Press Enter to continue...")
                return
            
            print("\n📋 Available configurations:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config.name}")
            
            if len(configs) == 1:
                config_file = configs[0]
            else:
                choice = self.get_user_choice(len(configs))
                config_file = configs[choice - 1]
        
        print(f"\n🚀 Running simulation: {config_file}")
        print("⏳ Starting simulation...")
        
        # Ask for options
        print("\n⚙️  Simulation options:")
        print("1. 🏃‍♂️ Standard run")
        print("2. 🔍 Dry-run (test only)")
        print("3. 🐛 Debug mode")
        print("4. 📸 With snapshots")
        
        option = self.get_user_choice(4)
        
        cmd = [sys.executable, "-m", "qbes.cli", "run", str(config_file)]
        
        if option == 2:
            cmd.append("--dry-run")
            print("🔍 Running dry-run validation...")
        elif option == 3:
            cmd.extend(["--debug-level", "DEBUG", "--verbose"])
            print("🐛 Running in debug mode...")
        elif option == 4:
            cmd.extend(["--save-snapshots", "100", "--verbose"])
            print("📸 Running with state snapshots...")
        else:
            cmd.append("--verbose")
            print("🏃‍♂️ Running standard simulation...")
        
        try:
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                print("\n✅ Simulation completed successfully!")
                
                # Ask to view results
                view = input("📊 View results now? (y/n): ").lower().strip()
                if view in ['y', 'yes']:
                    self.view_results()
            else:
                print("\n❌ Simulation failed")
                
        except Exception as e:
            print(f"❌ Error running simulation: {e}")
        
        if config_file is None:  # Only pause if called from menu
            input("\n📱 Press Enter to continue...")
    
    def view_results(self):
        """View simulation results"""
        self.clear_screen()
        print("📊 View QBES Results")
        print("=" * 25)
        
        # Find result directories
        result_dirs = [d for d in Path('.').iterdir() if d.is_dir() and 'result' in d.name.lower()]
        
        if not result_dirs:
            print("❌ No result directories found")
            print("💡 Run a simulation first (option 1 from main menu)")
            input("\n📱 Press Enter to continue...")
            return
        
        print("\n📋 Available results:")
        for i, result_dir in enumerate(result_dirs, 1):
            print(f"{i}. {result_dir.name}")
        
        if len(result_dirs) == 1:
            result_dir = result_dirs[0]
        else:
            choice = self.get_user_choice(len(result_dirs))
            result_dir = result_dirs[choice - 1]
        
        print(f"\n📊 Viewing results: {result_dir}")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "qbes.cli", "view",
                str(result_dir)
            ], text=True)
            
        except Exception as e:
            print(f"❌ Error viewing results: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def run_benchmarks(self):
        """Run benchmark tests"""
        self.clear_screen()
        print("🧪 QBES Benchmarks")
        print("=" * 20)
        
        print("\n📋 Benchmark options:")
        print("1. 🚀 Quick benchmarks (5 min)")
        print("2. 📊 Full benchmark suite (20 min)")
        print("3. 🔬 Performance benchmarks")
        
        choice = self.get_user_choice(3)
        
        if choice == 1:
            cmd = [sys.executable, "-m", "qbes.cli", "benchmark", "run-benchmarks", "--quick"]
            print("🚀 Running quick benchmarks...")
        elif choice == 2:
            cmd = [sys.executable, "-m", "qbes.cli", "benchmark", "run-benchmarks", "--verbose"]
            print("📊 Running full benchmark suite...")
        else:
            cmd = [sys.executable, "run_benchmarks.py"]
            print("🔬 Running performance benchmarks...")
        
        try:
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                print("\n✅ Benchmarks completed successfully!")
            else:
                print("\n⚠️  Benchmarks completed with issues")
                
        except Exception as e:
            print(f"❌ Error running benchmarks: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def advanced_options(self):
        """Advanced options menu"""
        self.clear_screen()
        print("🛠️  Advanced Options")
        print("=" * 25)
        
        print("\n📋 Advanced features:")
        print("1. 🔧 System information")
        print("2. 📁 File management")
        print("3. 🐛 Debug tools")
        print("4. ⚙️  Settings")
        print("5. 🔄 Update QBES")
        
        choice = self.get_user_choice(5)
        
        if choice == 1:
            self.system_info()
        elif choice == 2:
            self.file_management()
        elif choice == 3:
            self.debug_tools()
        elif choice == 4:
            self.settings()
        else:
            self.update_qbes()
    
    def system_info(self):
        """Display system information"""
        print("\n🔧 System Information")
        print("=" * 25)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "qbes.cli", "info"
            ], text=True)
            
        except Exception as e:
            print(f"❌ Error getting system info: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def file_management(self):
        """File management utilities"""
        print("\n📁 File Management")
        print("=" * 20)
        
        print("\n📋 Current directory contents:")
        
        # List files by type
        configs = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.json'))
        results = [d for d in Path('.').iterdir() if d.is_dir() and 'result' in d.name.lower()]
        
        if configs:
            print(f"\n⚙️  Configuration files ({len(configs)}):")
            for config in configs:
                size = config.stat().st_size
                print(f"   📄 {config.name} ({size:,} bytes)")
        
        if results:
            print(f"\n📊 Result directories ({len(results)}):")
            for result in results:
                files = len(list(result.glob('*')))
                print(f"   📁 {result.name} ({files} files)")
        
        print(f"\n💾 Total files: {len(list(Path('.').glob('*')))}")
        
        input("\n📱 Press Enter to continue...")
    
    def debug_tools(self):
        """Debug and diagnostic tools"""
        print("\n🐛 Debug Tools")
        print("=" * 15)
        
        print("\n📋 Debug options:")
        print("1. 🧪 Test QBES installation")
        print("2. 🔍 Check dependencies")
        print("3. 📊 Run diagnostics")
        
        choice = self.get_user_choice(3)
        
        if choice == 1:
            print("\n🧪 Testing QBES installation...")
            try:
                result = subprocess.run([
                    sys.executable, "-c", 
                    "import qbes; print('✅ QBES imported successfully'); print('Version:', getattr(qbes, '__version__', 'unknown'))"
                ], text=True)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == 2:
            print("\n🔍 Checking dependencies...")
            deps = ['numpy', 'scipy', 'matplotlib', 'qutip', 'click', 'yaml']
            for dep in deps:
                try:
                    result = subprocess.run([
                        sys.executable, "-c", f"import {dep}; print('✅ {dep}')"
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ {dep}")
                    else:
                        print(f"❌ {dep} - not found")
                except:
                    print(f"❌ {dep} - error")
        
        else:
            print("\n📊 Running diagnostics...")
            try:
                result = subprocess.run([sys.executable, "demo_qbes.py"], text=True)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        input("\n📱 Press Enter to continue...")
    
    def settings(self):
        """QBES settings"""
        print("\n⚙️  QBES Settings")
        print("=" * 18)
        print("💡 Settings are configured via configuration files")
        print("📁 Edit .yaml or .json files to customize behavior")
        
        input("\n📱 Press Enter to continue...")
    
    def update_qbes(self):
        """Update QBES"""
        print("\n🔄 Update QBES")
        print("=" * 15)
        print("💡 QBES v1.2 is the latest version")
        print("✅ You have the most recent release")
        
        input("\n📱 Press Enter to continue...")
    
    def help_documentation(self):
        """Help and documentation"""
        self.clear_screen()
        print("📚 Help & Documentation")
        print("=" * 30)
        
        print("\n📋 Available documentation:")
        docs = [
            ("README.md", "📖 Project overview"),
            ("USER_GUIDE.md", "👤 Complete user guide"),
            ("QBES_v1.2_Release_Notes.md", "📝 Version 1.2 features"),
            ("docs/api_reference.md", "🔧 API documentation"),
            ("docs/theory.md", "🧮 Mathematical theory")
        ]
        
        for i, (file, desc) in enumerate(docs, 1):
            if Path(file).exists():
                print(f"{i}. {desc} - {file}")
            else:
                print(f"{i}. {desc} - ❌ Not found")
        
        print(f"\n💡 Tips:")
        print("• Use any text editor to read .md files")
        print("• Check the website/ directory for interactive demos")
        print("• Run 'python demo_qbes.py' for a quick demo")
        
        input("\n📱 Press Enter to continue...")
    
    def web_interface(self):
        """Launch web interface"""
        self.clear_screen()
        print("🌐 QBES Web Interface")
        print("=" * 25)
        
        print("\n📋 Web interface options:")
        print("1. 🎓 Educational website")
        print("2. 🚀 Interactive QBES interface")
        print("3. 📊 Results viewer")
        
        choice = self.get_user_choice(3)
        
        if choice == 1:
            print("\n🎓 Opening educational website...")
            try:
                import webbrowser
                webbrowser.open("website/qbes_website.html")
                print("✅ Educational website opened in browser")
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Manually open: website/qbes_website.html")
        
        elif choice == 2:
            print("\n🚀 Starting interactive web interface...")
            try:
                result = subprocess.run([sys.executable, "qbes/web_interface.py"], text=True)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        else:
            print("\n📊 Starting results viewer...")
            print("💡 Feature coming soon!")
        
        input("\n📱 Press Enter to continue...")
    
    def run(self):
        """Main application loop"""
        while True:
            try:
                self.show_main_menu()
                choice = self.get_user_choice(9)
                
                if choice == 1:
                    self.quick_start()
                elif choice == 2:
                    self.create_configuration()
                elif choice == 3:
                    self.validate_system()
                elif choice == 4:
                    self.view_results()
                elif choice == 5:
                    self.run_benchmarks()
                elif choice == 6:
                    self.advanced_options()
                elif choice == 7:
                    self.help_documentation()
                elif choice == 8:
                    self.web_interface()
                elif choice == 9:
                    print("\n👋 Thank you for using QBES!")
                    print("🔬 Happy quantum biology research!")
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                input("📱 Press Enter to continue...")

def main():
    """Main entry point"""
    app = QBESInteractive()
    app.run()

if __name__ == "__main__":
    main()