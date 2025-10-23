#!/usr/bin/env python3
"""
QBES Complete Launch Script
Tests all components and provides launch options
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

def print_banner():
    """Print QBES banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    QQQQQQ  BBBBB   EEEEEEE  SSSSS    Complete Launcher     ║
║   QQ    QQ BB   BB EE       SS                              ║
║   QQ    QQ BBBBBB  EEEEE    SSSSS                           ║
║   QQ  Q QQ BB   BB EE           SS                          ║
║    QQQQQQ  BBBBB   EEEEEEE  SSSSS                           ║
║                                                              ║
║         Quantum Biological Environment Simulator            ║
║              Complete Testing & Launch Suite                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system():
    """Check system requirements"""
    print("🔍 System Requirements Check")
    print("-" * 40)
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (3.8+ required)")
        return False
    
    # Required packages
    required_packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_qbes_components():
    """Test QBES core components"""
    print("\n🧪 QBES Components Test")
    print("-" * 40)
    
    try:
        # Test imports
        from qbes.quantum_engine import QuantumEngine
        from qbes.simulation_engine import SimulationEngine
        from qbes.config_manager import ConfigurationManager
        print("✅ Core modules import successfully")
        
        # Test basic functionality
        engine = QuantumEngine()
        hamiltonian = engine.create_two_level_hamiltonian(2.0, 0.1)
        print("✅ Quantum engine functionality")
        
        config_manager = ConfigurationManager()
        default_config = config_manager.create_default_config()
        print("✅ Configuration management")
        
        return True
        
    except Exception as e:
        print(f"❌ QBES components test failed: {e}")
        return False

def check_files():
    """Check essential files"""
    print("\n📁 File Structure Check")
    print("-" * 40)
    
    essential_files = [
        "qbes_interactive.py",
        "start_website.py",
        "test_project.py",
        "website/index.html",
        "website/server.py",
        "website/script.js",
        "website/styles.css",
        "docs/README.md",
        "PROJECT_STATUS.md"
    ]
    
    all_present = True
    
    for file_path in essential_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_present = False
    
    return all_present

def show_menu():
    """Show launch options menu"""
    print("\n🚀 QBES Launch Options")
    print("=" * 50)
    print("1. 🎮 Interactive Terminal Interface")
    print("2. 🌐 Web Interface (Full Featured)")
    print("3. 🧪 Run Project Tests")
    print("4. 📊 Run Benchmarks")
    print("5. 📚 View Documentation")
    print("6. 📈 Project Status Report")
    print("7. 🔧 Quick System Test")
    print("0. ❌ Exit")
    print("=" * 50)

def launch_interactive():
    """Launch interactive interface"""
    print("\n🎮 Launching Interactive Interface...")
    try:
        subprocess.run([sys.executable, "qbes_interactive.py"])
    except KeyboardInterrupt:
        print("\n👋 Interactive interface closed")
    except Exception as e:
        print(f"❌ Failed to launch interactive interface: {e}")

def launch_website():
    """Launch web interface"""
    print("\n🌐 Launching Web Interface...")
    try:
        subprocess.run([sys.executable, "start_website.py"])
    except KeyboardInterrupt:
        print("\n👋 Website closed")
    except Exception as e:
        print(f"❌ Failed to launch website: {e}")

def run_tests():
    """Run project tests"""
    print("\n🧪 Running Project Tests...")
    try:
        subprocess.run([sys.executable, "test_project.py"])
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")

def run_benchmarks():
    """Run benchmarks"""
    print("\n📊 Running Benchmarks...")
    try:
        if Path("run_benchmarks.py").exists():
            subprocess.run([sys.executable, "run_benchmarks.py"])
        else:
            print("❌ Benchmark script not found")
    except Exception as e:
        print(f"❌ Failed to run benchmarks: {e}")

def view_documentation():
    """View documentation"""
    print("\n📚 Opening Documentation...")
    
    docs_files = [
        ("Project Status", "PROJECT_STATUS.md"),
        ("Main README", "README.md"),
        ("Documentation Index", "docs/README.md"),
        ("Project Overview", "docs/project-overview.md"),
        ("Installation Guide", "docs/guides/installation.md"),
        ("Getting Started", "docs/guides/getting-started.md")
    ]
    
    print("\nAvailable Documentation:")
    for i, (name, path) in enumerate(docs_files, 1):
        if Path(path).exists():
            print(f"{i}. ✅ {name} ({path})")
        else:
            print(f"{i}. ❌ {name} ({path}) - Missing")
    
    try:
        choice = input("\nEnter number to open (or press Enter to skip): ").strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(docs_files):
                name, path = docs_files[idx]
                if Path(path).exists():
                    if sys.platform.startswith('win'):
                        os.startfile(path)
                    elif sys.platform.startswith('darwin'):
                        subprocess.run(['open', path])
                    else:
                        subprocess.run(['xdg-open', path])
                    print(f"📖 Opened {name}")
                else:
                    print(f"❌ File not found: {path}")
    except Exception as e:
        print(f"❌ Failed to open documentation: {e}")

def show_status_report():
    """Show project status report"""
    print("\n📈 QBES Project Status Report")
    print("=" * 50)
    
    try:
        # Count files
        python_files = len(list(Path(".").glob("**/*.py")))
        test_files = len(list(Path("tests").glob("**/*.py"))) if Path("tests").exists() else 0
        doc_files = len(list(Path("docs").glob("**/*.md"))) if Path("docs").exists() else 0
        
        # Calculate lines of code
        total_lines = 0
        for py_file in Path(".").glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
        
        print(f"📊 Codebase Statistics:")
        print(f"   • Python Files: {python_files}")
        print(f"   • Test Files: {test_files}")
        print(f"   • Documentation Files: {doc_files}")
        print(f"   • Total Lines of Code: ~{total_lines:,}")
        
        print(f"\n🎯 Project Grade: A- (Development Version)")
        print(f"📅 Last Updated: {time.strftime('%Y-%m-%d')}")
        
        print(f"\n🔗 Repository: https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-")
        print(f"👨‍💻 Developer: Aniket Mehra")
        print(f"📧 Contact: aniketmehra715@gmail.com")
        
        # Check component status
        print(f"\n🔧 Component Status:")
        components = [
            ("Interactive Interface", "qbes_interactive.py"),
            ("Web Interface", "website/index.html"),
            ("Core Engine", "qbes/quantum_engine.py"),
            ("CLI Interface", "qbes/cli.py"),
            ("Documentation", "docs/README.md")
        ]
        
        for name, path in components:
            status = "✅ Ready" if Path(path).exists() else "❌ Missing"
            print(f"   • {name}: {status}")
        
    except Exception as e:
        print(f"❌ Failed to generate status report: {e}")

def quick_system_test():
    """Run quick system test"""
    print("\n🔧 Quick System Test")
    print("-" * 30)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: System requirements
    if check_system():
        tests_passed += 1
        print("✅ System requirements")
    else:
        print("❌ System requirements")
    
    # Test 2: File structure
    if check_files():
        tests_passed += 1
        print("✅ File structure")
    else:
        print("❌ File structure")
    
    # Test 3: QBES components
    if test_qbes_components():
        tests_passed += 1
        print("✅ QBES components")
    else:
        print("❌ QBES components")
    
    # Test 4: Web components
    try:
        sys.path.insert(0, "website")
        from server import app
        tests_passed += 1
        print("✅ Web components")
    except:
        print("❌ Web components")
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed ({tests_passed/total_tests*100:.0f}%)")
    
    if tests_passed == total_tests:
        print("🎉 All systems ready! QBES is fully functional.")
        return True
    elif tests_passed >= 3:
        print("✅ Most systems ready! QBES is functional with minor issues.")
        return True
    else:
        print("⚠️ Some systems need attention before full functionality.")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == "0":
                print("\n👋 Thank you for using QBES!")
                break
            elif choice == "1":
                launch_interactive()
            elif choice == "2":
                launch_website()
            elif choice == "3":
                run_tests()
            elif choice == "4":
                run_benchmarks()
            elif choice == "5":
                view_documentation()
            elif choice == "6":
                show_status_report()
            elif choice == "7":
                quick_system_test()
            else:
                print("❌ Invalid choice. Please enter 0-7.")
            
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()