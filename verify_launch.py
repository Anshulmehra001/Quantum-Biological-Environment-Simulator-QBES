#!/usr/bin/env python3
"""
QBES Launch Verification Script
Comprehensive test of all components and documentation
"""

import os
import sys
import json
import time
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)

def check_documentation():
    """Check all documentation files for accuracy"""
    print_header("Documentation Verification")
    
    docs_to_check = [
        ("README.md", "Main project README"),
        ("PROJECT_STATUS.md", "Project status report"),
        ("docs/README.md", "Documentation index"),
        ("docs/project-overview.md", "Project overview"),
        ("docs/guides/installation.md", "Installation guide"),
        ("docs/guides/getting-started.md", "Getting started guide"),
        ("docs/guides/user-guide.md", "User guide"),
        ("docs/technical/complete-user-guide.md", "Complete technical guide"),
        ("docs/technical/mathematical-foundations.md", "Mathematical foundations"),
        ("docs/business/market-analysis.md", "Market analysis"),
        ("docs/examples/photosynthesis-example.md", "Photosynthesis example"),
        ("docs/examples/enzyme-example.md", "Enzyme example")
    ]
    
    issues = []
    
    for doc_path, description in docs_to_check:
        if Path(doc_path).exists():
            print(f"✅ {description} ({doc_path})")
            
            # Check for common issues
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for placeholder URLs
                    if "[REPOSITORY_URL]" in content:
                        issues.append(f"❌ {doc_path}: Contains placeholder URL")
                    
                    # Check for incorrect repository references
                    if "your-org/qbes" in content:
                        issues.append(f"❌ {doc_path}: Contains incorrect repository reference")
                    
                    # Check for 3.8M lines claim
                    if "3.8M" in content or "3,800,000" in content:
                        issues.append(f"❌ {doc_path}: Contains incorrect 3.8M lines claim")
                    
                    # Check for correct repository URL
                    correct_repo = "https://github.com/Anshulmehra001/Quantum-Biological-Environment-Simulator-QBES-"
                    if "github.com" in content and correct_repo not in content:
                        # Check if it has any github reference
                        if "github.com" in content:
                            issues.append(f"⚠️ {doc_path}: May have incorrect GitHub URL")
                    
            except Exception as e:
                issues.append(f"❌ {doc_path}: Error reading file - {e}")
                
        else:
            issues.append(f"❌ {description}: File missing ({doc_path})")
    
    if issues:
        print("\n🚨 Documentation Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All documentation files verified successfully!")
        return True

def check_web_interface():
    """Check web interface components"""
    print_header("Web Interface Verification")
    
    web_files = [
        ("website/index.html", "Main HTML file"),
        ("website/script.js", "JavaScript functionality"),
        ("website/styles.css", "CSS styling"),
        ("website/server.py", "Flask backend server")
    ]
    
    all_present = True
    
    for file_path, description in web_files:
        if Path(file_path).exists():
            print(f"✅ {description} ({file_path})")
            
            # Check file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    if file_path == "website/index.html":
                        # Check for updated stats
                        if "18,000+" in content:
                            print("  ✅ Correct line count in HTML")
                        else:
                            print("  ⚠️ Line count may need updating in HTML")
                    
                    elif file_path == "website/server.py":
                        # Check for Flask app
                        if "Flask" in content and "app = Flask" in content:
                            print("  ✅ Flask server properly configured")
                        else:
                            print("  ❌ Flask server configuration issue")
                            all_present = False
                            
            except Exception as e:
                print(f"  ❌ Error reading {file_path}: {e}")
                all_present = False
        else:
            print(f"❌ {description}: File missing ({file_path})")
            all_present = False
    
    return all_present

def check_launch_scripts():
    """Check launch scripts"""
    print_header("Launch Scripts Verification")
    
    scripts = [
        ("launch_qbes.py", "Main launcher"),
        ("qbes_interactive.py", "Interactive interface"),
        ("start_website.py", "Website launcher"),
        ("test_project.py", "Project tester"),
        ("verify_launch.py", "This verification script")
    ]
    
    all_present = True
    
    for script_path, description in scripts:
        if Path(script_path).exists():
            print(f"✅ {description} ({script_path})")
            
            # Check if script is executable
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.startswith('#!/usr/bin/env python3'):
                        print("  ✅ Proper shebang line")
                    elif "def main(" in content or "if __name__ == '__main__'" in content:
                        print("  ✅ Main function present")
                    else:
                        print("  ⚠️ May not be properly structured as executable script")
                        
            except Exception as e:
                print(f"  ❌ Error reading {script_path}: {e}")
                all_present = False
        else:
            print(f"❌ {description}: File missing ({script_path})")
            all_present = False
    
    return all_present

def check_core_components():
    """Check QBES core components"""
    print_header("Core Components Verification")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path.cwd()))
        
        print("Testing core imports...")
        
        # Basic imports
        import numpy as np
        print("✅ NumPy")
        
        import scipy
        print("✅ SciPy")
        
        # QBES imports
        from qbes.core.data_models import DensityMatrix, Hamiltonian
        print("✅ QBES Data Models")
        
        from qbes.quantum_engine import QuantumEngine
        print("✅ QBES Quantum Engine")
        
        from qbes.simulation_engine import SimulationEngine
        print("✅ QBES Simulation Engine")
        
        from qbes.config_manager import ConfigurationManager
        print("✅ QBES Configuration Manager")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        
        engine = QuantumEngine()
        hamiltonian = engine.create_two_level_hamiltonian(2.0, 0.1)
        print("✅ Quantum engine functionality")
        
        config_manager = ConfigurationManager()
        default_config = config_manager.create_default_config()
        print("✅ Configuration management")
        
        return True
        
    except Exception as e:
        print(f"❌ Core components test failed: {e}")
        return False

def generate_launch_report():
    """Generate comprehensive launch report"""
    print_header("Launch Verification Report")
    
    # Run all checks
    checks = [
        ("Documentation", check_documentation),
        ("Web Interface", check_web_interface),
        ("Launch Scripts", check_launch_scripts),
        ("Core Components", check_core_components)
    ]
    
    results = {}
    total_score = 0
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if result:
                total_score += 1
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Generate summary
    print_header("Final Verification Summary")
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    score_percentage = (total_score / len(checks)) * 100
    print(f"\nOverall Score: {total_score}/{len(checks)} ({score_percentage:.0f}%)")
    
    if score_percentage == 100:
        grade = "A+"
        status = "🎉 EXCELLENT - All systems ready!"
    elif score_percentage >= 90:
        grade = "A"
        status = "✅ VERY GOOD - Minor issues only"
    elif score_percentage >= 80:
        grade = "B+"
        status = "✅ GOOD - Some issues to address"
    elif score_percentage >= 70:
        grade = "B"
        status = "⚠️ FAIR - Several issues need attention"
    else:
        grade = "C"
        status = "❌ NEEDS WORK - Major issues present"
    
    print(f"Grade: {grade}")
    print(f"Status: {status}")
    
    # Recommendations
    print(f"\n💡 Launch Recommendations:")
    if score_percentage == 100:
        print("1. 🚀 Ready to launch! All systems operational")
        print("2. 🌐 Start with: python start_website.py")
        print("3. 🎮 Or try: python qbes_interactive.py")
        print("4. 📊 Run tests: python test_project.py")
    elif score_percentage >= 80:
        print("1. ✅ Most systems ready - can launch with minor issues")
        print("2. 🔧 Address failed checks before full deployment")
        print("3. 🧪 Run tests to verify functionality")
    else:
        print("1. ⚠️ Address major issues before launching")
        print("2. 🔧 Fix failed components first")
        print("3. 🧪 Re-run verification after fixes")
    
    return score_percentage >= 80

def main():
    """Main verification function"""
    print("🧬 QBES Launch Verification System")
    print("=" * 60)
    print("Comprehensive verification of all QBES components")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run verification
    success = generate_launch_report()
    
    print(f"\n📋 Verification completed at {time.strftime('%H:%M:%S')}")
    
    if success:
        print("\n🎯 QBES is ready for launch!")
        print("\nQuick Launch Commands:")
        print("  python launch_qbes.py      # Complete launcher")
        print("  python start_website.py    # Web interface")
        print("  python qbes_interactive.py # Terminal interface")
        print("  python test_project.py     # Run tests")
    else:
        print("\n⚠️ Please address issues before launching")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)