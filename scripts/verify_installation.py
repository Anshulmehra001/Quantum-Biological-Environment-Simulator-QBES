#!/usr/bin/env python3
"""
QBES Installation Verification Script

This script performs comprehensive verification of QBES installation,
checking dependencies, system requirements, and core functionality.
"""

import sys
import os
import subprocess
import importlib
import platform
import psutil
from pathlib import Path
import tempfile
import shutil

def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(message, status):
    """Print status message with colored indicator."""
    if status:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
    return status

def check_python_version():
    """Check Python version requirements."""
    print_header("Python Version Check")
    
    version = sys.version_info
    min_version = (3, 8)
    recommended_version = (3, 9)
    
    current_version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Current Python version: {current_version_str}")
    
    if version >= recommended_version:
        return print_status(f"Python {current_version_str} (recommended)", True)
    elif version >= min_version:
        print_status(f"Python {current_version_str} (minimum met)", True)
        print("⚠️  Consider upgrading to Python 3.9+ for better performance")
        return True
    else:
        return print_status(f"Python {current_version_str} (too old, need 3.8+)", False)

def check_system_resources():
    """Check system resource requirements."""
    print_header("System Resources Check")
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_ok = ram_gb >= 8
    print_status(f"RAM: {ram_gb:.1f} GB (minimum: 8 GB)", ram_ok)
    
    if ram_gb >= 32:
        print("💡 Excellent RAM for large simulations")
    elif ram_gb >= 16:
        print("💡 Good RAM for medium simulations")
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count()
    cpu_ok = cpu_cores >= 2
    print_status(f"CPU cores: {cpu_cores} (minimum: 2)", cpu_ok)
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    disk_ok = free_gb >= 2
    print_status(f"Free disk space: {free_gb:.1f} GB (minimum: 2 GB)", disk_ok)
    
    # Check platform
    system = platform.system()
    supported_platforms = ['Linux', 'Darwin', 'Windows']
    platform_ok = system in supported_platforms
    print_status(f"Platform: {system} (supported: {', '.join(supported_platforms)})", platform_ok)
    
    return ram_ok and cpu_ok and disk_ok and platform_ok

def check_core_dependencies():
    """Check core Python dependencies."""
    print_header("Core Dependencies Check")
    
    core_deps = {
        'numpy': '1.20.0',
        'scipy': '1.7.0',
        'matplotlib': '3.3.0',
        'yaml': '5.4.0',  # PyYAML
        'click': '8.0.0'
    }
    
    all_ok = True
    
    for module_name, min_version in core_deps.items():
        try:
            if module_name == 'yaml':
                import yaml
                module = yaml
                module_name = 'PyYAML'
            else:
                module = importlib.import_module(module_name)
            
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{module_name}: {version}", True)
            
        except ImportError:
            print_status(f"{module_name}: NOT FOUND", False)
            all_ok = False
    
    return all_ok

def check_optional_dependencies():
    """Check optional dependencies for enhanced functionality."""
    print_header("Optional Dependencies Check")
    
    optional_deps = {
        'qutip': 'Quantum calculations',
        'openmm': 'Molecular dynamics',
        'mdtraj': 'Trajectory analysis',
        'numba': 'JIT compilation',
        'cupy': 'GPU acceleration',
        'mpi4py': 'MPI parallelization'
    }
    
    available = []
    
    for module_name, description in optional_deps.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{module_name}: {version} ({description})", True)
            available.append(module_name)
        except ImportError:
            print(f"⚪ {module_name}: Not installed ({description})")
    
    if 'qutip' not in available:
        print("⚠️  QuTiP is highly recommended for quantum calculations")
    
    return available

def check_qbes_installation():
    """Check QBES package installation."""
    print_header("QBES Installation Check")
    
    try:
        import qbes
        version = getattr(qbes, '__version__', 'unknown')
        print_status(f"QBES package: {version}", True)
        
        # Check core modules
        modules_to_check = [
            'qbes.config_manager',
            'qbes.simulation_engine',
            'qbes.analysis',
            'qbes.cli'
        ]
        
        all_modules_ok = True
        for module_name in modules_to_check:
            try:
                importlib.import_module(module_name)
                print_status(f"Module {module_name}: OK", True)
            except ImportError as e:
                print_status(f"Module {module_name}: FAILED ({e})", False)
                all_modules_ok = False
        
        return all_modules_ok
        
    except ImportError:
        print_status("QBES package: NOT FOUND", False)
        print("💡 Install QBES with: pip install -e .")
        return False

def check_cli_functionality():
    """Check command-line interface functionality."""
    print_header("CLI Functionality Check")
    
    try:
        # Test qbes command availability
        result = subprocess.run(['qbes', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version_output = result.stdout.strip()
            print_status(f"CLI command: {version_output}", True)
            
            # Test help command
            help_result = subprocess.run(['qbes', '--help'], 
                                       capture_output=True, text=True, timeout=10)
            help_ok = help_result.returncode == 0 and 'QBES' in help_result.stdout
            print_status("Help system: Working", help_ok)
            
            return help_ok
        else:
            print_status("CLI command: FAILED", False)
            print(f"Error: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("CLI command: NOT FOUND", False)
        print("💡 Make sure QBES is installed and in your PATH")
        return False

def test_basic_functionality():
    """Test basic QBES functionality."""
    print_header("Basic Functionality Test")
    
    try:
        from qbes import ConfigurationManager, SimulationEngine
        from qbes.core.data_models import SimulationConfig
        
        # Test configuration manager
        config_manager = ConfigurationManager()
        print_status("ConfigurationManager: OK", True)
        
        # Test creating a basic configuration
        config = SimulationConfig(
            system_pdb="test.pdb",
            temperature=300.0,
            simulation_time=1e-12,
            time_step=1e-15,
            quantum_subsystem_selection="chromophores",
            noise_model_type="protein_ohmic",
            output_directory="./test_output"
        )
        print_status("Configuration creation: OK", True)
        
        # Test validation
        validation = config_manager.validate_parameters(config)
        print_status("Parameter validation: OK", True)
        
        # Test simulation engine initialization
        engine = SimulationEngine()
        print_status("SimulationEngine: OK", True)
        
        return True
        
    except Exception as e:
        print_status(f"Basic functionality: FAILED ({e})", False)
        return False

def run_example_config_test():
    """Test configuration generation and validation."""
    print_header("Configuration System Test")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.yaml")
            
            # Test config generation
            result = subprocess.run(['qbes', 'generate-config', config_file, '--template', 'default'],
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print_status("Config generation: OK", True)
                
                # Test config validation
                val_result = subprocess.run(['qbes', 'validate', config_file],
                                          capture_output=True, text=True, timeout=30)
                
                if val_result.returncode == 0:
                    print_status("Config validation: OK", True)
                    return True
                else:
                    print_status("Config validation: FAILED", False)
                    print(f"Validation error: {val_result.stderr}")
                    return False
            else:
                print_status("Config generation: FAILED", False)
                print(f"Generation error: {result.stderr}")
                return False
                
    except Exception as e:
        print_status(f"Configuration test: FAILED ({e})", False)
        return False

def generate_installation_report():
    """Generate comprehensive installation report."""
    print_header("Installation Report Summary")
    
    checks = [
        ("Python Version", check_python_version()),
        ("System Resources", check_system_resources()),
        ("Core Dependencies", check_core_dependencies()),
        ("QBES Installation", check_qbes_installation()),
        ("CLI Functionality", check_cli_functionality()),
        ("Basic Functionality", test_basic_functionality()),
        ("Configuration System", run_example_config_test())
    ]
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    print(f"\nInstallation Verification Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 QBES installation is complete and fully functional!")
        print("You can now run quantum biological simulations.")
        print("\nNext steps:")
        print("1. Read the tutorial: TUTORIAL.md")
        print("2. Check the user guide: USER_GUIDE.md")
        print("3. Run your first simulation:")
        print("   qbes generate-config my_sim.yaml --interactive")
        print("   qbes run my_sim.yaml")
        return True
    else:
        print(f"\n⚠️  Installation incomplete: {total - passed} issues found")
        print("\nFailed checks:")
        for name, status in checks:
            if not status:
                print(f"  ❌ {name}")
        
        print("\nTroubleshooting:")
        print("1. Check system requirements in README.md")
        print("2. Reinstall with: pip install -e .")
        print("3. Install missing dependencies")
        print("4. Check the troubleshooting guide")
        return False

def main():
    """Main verification function."""
    print("QBES Installation Verification Script")
    print("====================================")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    
    # Check optional dependencies first (informational)
    optional_deps = check_optional_dependencies()
    
    # Run all verification checks
    success = generate_installation_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()