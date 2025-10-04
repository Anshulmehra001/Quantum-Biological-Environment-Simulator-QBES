#!/usr/bin/env python3
"""
QBES Installation and Setup Script

This script automates the installation of QBES and its dependencies,
including system compatibility checking and virtual environment setup.
"""

import os
import sys
import subprocess
import platform
import shutil
import venv
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class QBESInstaller:
    """Automated installer for QBES."""
    
    def __init__(self):
        """Initialize the installer."""
        self.system_info = self._get_system_info()
        self.python_version = sys.version_info
        self.install_log = []
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for compatibility checking."""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
    
    def _log(self, message: str, level: str = 'INFO'):
        """Log installation messages."""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.install_log.append(log_entry)
    
    def check_system_compatibility(self) -> bool:
        """Check if the system meets QBES requirements."""
        self._log("Checking system compatibility...")
        
        # Check Python version
        if self.python_version < (3, 8):
            self._log(f"Python {self.python_version.major}.{self.python_version.minor} detected. "
                     f"QBES requires Python 3.8 or higher.", 'ERROR')
            return False
        
        self._log(f"✓ Python {self.python_version.major}.{self.python_version.minor} is compatible")
        
        # Check platform support
        supported_platforms = ['Windows', 'Linux', 'Darwin']  # Darwin = macOS
        if self.system_info['platform'] not in supported_platforms:
            self._log(f"Platform {self.system_info['platform']} is not officially supported.", 'WARNING')
        else:
            self._log(f"✓ Platform {self.system_info['platform']} is supported")
        
        # Check for required system tools
        required_tools = ['git']
        missing_tools = []
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
        
        if missing_tools:
            self._log(f"Missing required tools: {', '.join(missing_tools)}", 'WARNING')
            self._log("Please install missing tools before proceeding.", 'WARNING')
        else:
            self._log("✓ All required system tools are available")
        
        return True
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are already installed."""
        self._log("Checking existing dependencies...")
        
        dependencies = {
            'numpy': False,
            'scipy': False,
            'matplotlib': False,
            'qutip': False,
            'openmm': False,
            'mdtraj': False,
            'biopython': False,
            'pandas': False,
            'h5py': False,
            'pyyaml': False,
            'click': False,
            'tqdm': False
        }
        
        for package in dependencies:
            try:
                __import__(package)
                dependencies[package] = True
                self._log(f"✓ {package} is already installed")
            except ImportError:
                self._log(f"✗ {package} needs to be installed")
        
        return dependencies
    
    def create_virtual_environment(self, venv_path: str) -> bool:
        """Create a virtual environment for QBES."""
        self._log(f"Creating virtual environment at: {venv_path}")
        
        try:
            # Remove existing venv if it exists
            if os.path.exists(venv_path):
                self._log("Removing existing virtual environment...")
                shutil.rmtree(venv_path)
            
            # Create new virtual environment
            venv.create(venv_path, with_pip=True)
            self._log("✓ Virtual environment created successfully")
            
            # Get the path to the virtual environment's Python executable
            if platform.system() == 'Windows':
                venv_python = os.path.join(venv_path, 'Scripts', 'python.exe')
                venv_pip = os.path.join(venv_path, 'Scripts', 'pip.exe')
            else:
                venv_python = os.path.join(venv_path, 'bin', 'python')
                venv_pip = os.path.join(venv_path, 'bin', 'pip')
            
            # Upgrade pip in the virtual environment
            self._log("Upgrading pip in virtual environment...")
            result = subprocess.run([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("✓ Pip upgraded successfully")
                return True
            else:
                self._log(f"Failed to upgrade pip: {result.stderr}", 'ERROR')
                return False
                
        except Exception as e:
            self._log(f"Failed to create virtual environment: {str(e)}", 'ERROR')
            return False
    
    def install_dependencies(self, venv_path: Optional[str] = None) -> bool:
        """Install QBES dependencies."""
        self._log("Installing QBES dependencies...")
        
        # Determine which Python executable to use
        if venv_path:
            if platform.system() == 'Windows':
                python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
            else:
                python_exe = os.path.join(venv_path, 'bin', 'python')
        else:
            python_exe = sys.executable
        
        # Install dependencies in order of importance
        dependency_groups = [
            # Core scientific computing (required)
            ['numpy>=1.20.0', 'scipy>=1.7.0', 'matplotlib>=3.3.0'],
            
            # Data handling (required)
            ['pandas>=1.3.0', 'h5py>=3.1.0', 'pyyaml>=5.4.0'],
            
            # User interface (required)
            ['click>=8.0.0', 'tqdm>=4.60.0'],
            
            # Quantum mechanics (required)
            ['qutip>=4.6.0'],
            
            # Molecular dynamics (optional but recommended)
            ['openmm>=7.6.0', 'mdtraj>=1.9.0', 'biopython>=1.78'],
            
            # Additional scientific packages
            ['numba>=0.54.0', 'seaborn>=0.11.0', 'plotly>=5.0.0', 'psutil>=5.8.0'],
            
            # Development tools
            ['pytest>=6.0.0', 'pytest-cov>=2.0.0']
        ]
        
        for i, group in enumerate(dependency_groups):
            self._log(f"Installing dependency group {i+1}/{len(dependency_groups)}: {group}")
            
            for package in group:
                try:
                    result = subprocess.run([python_exe, '-m', 'pip', 'install', package],
                                          capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        self._log(f"✓ Installed {package}")
                    else:
                        self._log(f"✗ Failed to install {package}: {result.stderr}", 'WARNING')
                        
                        # For optional packages, continue; for required ones, suggest alternatives
                        if i < 4:  # Required packages
                            self._log(f"Package {package} is required. Please install manually.", 'ERROR')
                        
                except subprocess.TimeoutExpired:
                    self._log(f"✗ Installation of {package} timed out", 'WARNING')
                except Exception as e:
                    self._log(f"✗ Error installing {package}: {str(e)}", 'WARNING')
        
        return True
    
    def install_qbes(self, venv_path: Optional[str] = None, development_mode: bool = False) -> bool:
        """Install QBES package itself."""
        self._log("Installing QBES package...")
        
        # Determine which Python executable to use
        if venv_path:
            if platform.system() == 'Windows':
                python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
            else:
                python_exe = os.path.join(venv_path, 'bin', 'python')
        else:
            python_exe = sys.executable
        
        try:
            if development_mode:
                # Install in development mode (editable)
                result = subprocess.run([python_exe, '-m', 'pip', 'install', '-e', '.'],
                                      capture_output=True, text=True)
            else:
                # Install normally
                result = subprocess.run([python_exe, 'setup.py', 'install'],
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("✓ QBES installed successfully")
                return True
            else:
                self._log(f"Failed to install QBES: {result.stderr}", 'ERROR')
                return False
                
        except Exception as e:
            self._log(f"Error installing QBES: {str(e)}", 'ERROR')
            return False
    
    def verify_installation(self, venv_path: Optional[str] = None) -> bool:
        """Verify that QBES was installed correctly."""
        self._log("Verifying QBES installation...")
        
        # Determine which Python executable to use
        if venv_path:
            if platform.system() == 'Windows':
                python_exe = os.path.join(venv_path, 'Scripts', 'python.exe')
            else:
                python_exe = os.path.join(venv_path, 'bin', 'python')
        else:
            python_exe = sys.executable
        
        # Test basic import
        try:
            result = subprocess.run([python_exe, '-c', 'import qbes; print("QBES import successful")'],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("✓ QBES imports successfully")
            else:
                self._log(f"✗ QBES import failed: {result.stderr}", 'ERROR')
                return False
        except Exception as e:
            self._log(f"Error testing QBES import: {str(e)}", 'ERROR')
            return False
        
        # Test CLI functionality
        try:
            result = subprocess.run([python_exe, '-m', 'qbes.cli', '--help'],
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and 'Quantum Biological Environment Simulator' in result.stdout:
                self._log("✓ QBES CLI is functional")
            else:
                self._log("✗ QBES CLI test failed", 'WARNING')
        except Exception as e:
            self._log(f"Error testing QBES CLI: {str(e)}", 'WARNING')
        
        # Test core functionality
        test_script = '''
import qbes
from qbes.core.data_models import SimulationConfig
from qbes.config_manager import ConfigurationManager

# Test configuration manager
config_manager = ConfigurationManager()
print("✓ Configuration manager created")

# Test data models
config = SimulationConfig(
    system_pdb="test.pdb",
    temperature=300.0,
    simulation_time=1e-12,
    time_step=1e-15,
    quantum_subsystem_selection="chromophores",
    noise_model_type="protein_ohmic",
    output_directory="./test_output"
)
print("✓ Data models functional")

print("QBES core functionality verified")
'''
        
        try:
            result = subprocess.run([python_exe, '-c', test_script],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log("✓ QBES core functionality verified")
                return True
            else:
                self._log(f"✗ QBES functionality test failed: {result.stderr}", 'ERROR')
                return False
        except Exception as e:
            self._log(f"Error testing QBES functionality: {str(e)}", 'ERROR')
            return False
    
    def generate_activation_script(self, venv_path: str) -> bool:
        """Generate scripts to activate the QBES environment."""
        self._log("Generating environment activation scripts...")
        
        try:
            # Create activation script for different platforms
            if platform.system() == 'Windows':
                # Windows batch script
                batch_script = f'''@echo off
echo Activating QBES environment...
call "{venv_path}\\Scripts\\activate.bat"
echo QBES environment activated. Type 'python -m qbes.cli --help' to get started.
cmd /k
'''
                with open('activate_qbes.bat', 'w') as f:
                    f.write(batch_script)
                
                self._log("✓ Created activate_qbes.bat")
                
                # PowerShell script
                ps_script = f'''Write-Host "Activating QBES environment..." -ForegroundColor Green
& "{venv_path}\\Scripts\\Activate.ps1"
Write-Host "QBES environment activated. Type 'python -m qbes.cli --help' to get started." -ForegroundColor Green
'''
                with open('activate_qbes.ps1', 'w') as f:
                    f.write(ps_script)
                
                self._log("✓ Created activate_qbes.ps1")
                
            else:
                # Unix/Linux/macOS bash script
                bash_script = f'''#!/bin/bash
echo "Activating QBES environment..."
source "{venv_path}/bin/activate"
echo "QBES environment activated. Type 'python -m qbes.cli --help' to get started."
exec "$SHELL"
'''
                with open('activate_qbes.sh', 'w') as f:
                    f.write(bash_script)
                
                # Make executable
                os.chmod('activate_qbes.sh', 0o755)
                self._log("✓ Created activate_qbes.sh")
            
            return True
            
        except Exception as e:
            self._log(f"Error generating activation scripts: {str(e)}", 'ERROR')
            return False
    
    def save_installation_log(self) -> bool:
        """Save the installation log to a file."""
        try:
            with open('qbes_installation.log', 'w') as f:
                f.write("QBES Installation Log\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"System Information:\n")
                for key, value in self.system_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("Installation Log:\n")
                for entry in self.install_log:
                    f.write(entry + "\n")
            
            self._log("✓ Installation log saved to qbes_installation.log")
            return True
            
        except Exception as e:
            self._log(f"Error saving installation log: {str(e)}", 'ERROR')
            return False
    
    def run_full_installation(self, venv_path: str = './qbes_env', 
                            development_mode: bool = False) -> bool:
        """Run the complete QBES installation process."""
        self._log("Starting QBES installation...")
        self._log(f"Target virtual environment: {venv_path}")
        
        # Step 1: Check system compatibility
        if not self.check_system_compatibility():
            self._log("System compatibility check failed. Installation aborted.", 'ERROR')
            return False
        
        # Step 2: Check existing dependencies
        existing_deps = self.check_dependencies()
        
        # Step 3: Create virtual environment
        if not self.create_virtual_environment(venv_path):
            self._log("Virtual environment creation failed. Installation aborted.", 'ERROR')
            return False
        
        # Step 4: Install dependencies
        if not self.install_dependencies(venv_path):
            self._log("Dependency installation failed. Installation aborted.", 'ERROR')
            return False
        
        # Step 5: Install QBES
        if not self.install_qbes(venv_path, development_mode):
            self._log("QBES installation failed. Installation aborted.", 'ERROR')
            return False
        
        # Step 6: Verify installation
        if not self.verify_installation(venv_path):
            self._log("Installation verification failed.", 'WARNING')
        
        # Step 7: Generate activation scripts
        self.generate_activation_script(venv_path)
        
        # Step 8: Save installation log
        self.save_installation_log()
        
        self._log("QBES installation completed successfully!", 'SUCCESS')
        self._log(f"To activate QBES environment:", 'SUCCESS')
        
        if platform.system() == 'Windows':
            self._log(f"  Windows: activate_qbes.bat", 'SUCCESS')
            self._log(f"  PowerShell: .\\activate_qbes.ps1", 'SUCCESS')
        else:
            self._log(f"  Unix/Linux/macOS: source activate_qbes.sh", 'SUCCESS')
        
        return True


def main():
    """Main installation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QBES Installation Script')
    parser.add_argument('--venv-path', default='./qbes_env',
                       help='Path for virtual environment (default: ./qbes_env)')
    parser.add_argument('--dev', action='store_true',
                       help='Install in development mode')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check system compatibility')
    
    args = parser.parse_args()
    
    installer = QBESInstaller()
    
    if args.check_only:
        installer.check_system_compatibility()
        installer.check_dependencies()
        return
    
    success = installer.run_full_installation(args.venv_path, args.dev)
    
    if not success:
        print("\nInstallation failed. Check qbes_installation.log for details.")
        sys.exit(1)
    else:
        print("\nInstallation completed successfully!")


if __name__ == '__main__':
    main()