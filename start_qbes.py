#!/usr/bin/env python3
"""
QBES Startup Script
Easy launcher for all QBES interfaces
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import time

def print_banner():
    """Print startup banner"""
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
║                   Startup Launcher                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['flask', 'numpy', 'matplotlib', 'pandas']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(packages):
    """Install missing dependencies"""
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def start_web_interface():
    """Start the web interface"""
    print("🌐 Starting QBES Web Interface...")
    
    web_app = Path("website/app.py")
    if not web_app.exists():
        print("❌ Web interface not found")
        return False
    
    try:
        # Start Flask app in background
        process = subprocess.Popen([sys.executable, str(web_app)], 
                                 cwd="website",
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open('http://localhost:5000')
        
        print("🎉 Web interface started!")
        print("📱 Browser should open automatically")
        print("🌐 Manual URL: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping web interface...")
        process.terminate()
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        return False
    
    return True

def start_cli():
    """Start the CLI interface"""
    print("💻 Starting QBES CLI Interface...")
    
    cli_script = Path("qbes_cli.py")
    if not cli_script.exists():
        print("❌ CLI interface not found")
        return False
    
    try:
        subprocess.run([sys.executable, str(cli_script), 'interactive'])
    except Exception as e:
        print(f"❌ Failed to start CLI: {e}")
        return False
    
    return True

def run_quick_demo():
    """Run a quick demonstration"""
    print("🚀 Running Quick Demo...")
    
    demo_script = Path("demo_qbes.py")
    if demo_script.exists():
        try:
            subprocess.run([sys.executable, str(demo_script)])
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    else:
        # Create a simple demo
        print("📊 Creating sample simulation...")
        
        cli_script = Path("qbes_cli.py")
        if cli_script.exists():
            try:
                # Create config and run simulation
                subprocess.run([sys.executable, str(cli_script), 'create-config', 'basic'])
                subprocess.run([sys.executable, str(cli_script), 'run', 'configs/basic_config.json'])
                
                # Show results
                results_dirs = list(Path("simulation_results").glob("sim_*"))
                if results_dirs:
                    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
                    subprocess.run([sys.executable, str(cli_script), 'view', str(latest_dir)])
                
            except Exception as e:
                print(f"❌ Demo failed: {e}")

def show_menu():
    """Show main menu"""
    print("\n🎯 Choose an option:")
    print("1. 🌐 Start Web Interface (Recommended)")
    print("2. 💻 Start CLI Interface")
    print("3. 🚀 Run Quick Demo")
    print("4. 📚 Open Documentation")
    print("5. 🔧 System Check")
    print("6. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                return 'web'
            elif choice == '2':
                return 'cli'
            elif choice == '3':
                return 'demo'
            elif choice == '4':
                return 'docs'
            elif choice == '5':
                return 'check'
            elif choice == '6':
                return 'exit'
            else:
                print("❌ Invalid choice. Please enter 1-6.")
        
        except KeyboardInterrupt:
            return 'exit'

def open_documentation():
    """Open documentation"""
    docs = [
        Path("website/qbes_website.html"),
        Path("SIMPLE_GUIDE.md"),
        Path("HOW_TO_USE_QBES.md"),
        Path("README.md")
    ]
    
    for doc in docs:
        if doc.exists():
            if doc.suffix == '.html':
                webbrowser.open(f'file://{doc.absolute()}')
                print(f"📖 Opened documentation: {doc}")
                return True
            else:
                print(f"📄 Documentation available: {doc}")
    
    print("📚 Documentation files found in project directory")
    return True

def system_check():
    """Perform system check"""
    print("🔧 QBES System Check")
    print("=" * 40)
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version}")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        install = input("📦 Install missing packages? (y/n): ").lower().startswith('y')
        if install:
            if install_dependencies(missing):
                print("✅ All dependencies installed")
            else:
                print("❌ Some dependencies failed to install")
    else:
        print("✅ All required packages available")
    
    # Check QBES modules
    try:
        sys.path.insert(0, str(Path.cwd()))
        import qbes
        print("✅ QBES modules available")
    except ImportError:
        print("⚠️  QBES modules not available (demo mode only)")
    
    # Check file structure
    required_files = [
        "qbes_cli.py",
        "website/app.py",
        "website/templates/index.html",
        "website/templates/simulator.html"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
    
    print("\n🎯 System check complete!")

def main():
    """Main function"""
    print_banner()
    
    # Check if we're in the right directory
    if not Path("qbes_cli.py").exists() and not Path("website").exists():
        print("❌ Please run this script from the QBES project directory")
        sys.exit(1)
    
    while True:
        choice = show_menu()
        
        if choice == 'web':
            start_web_interface()
        
        elif choice == 'cli':
            start_cli()
        
        elif choice == 'demo':
            run_quick_demo()
        
        elif choice == 'docs':
            open_documentation()
        
        elif choice == 'check':
            system_check()
        
        elif choice == 'exit':
            print("👋 Thank you for using QBES!")
            break
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()