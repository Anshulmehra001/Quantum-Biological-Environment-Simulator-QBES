#!/usr/bin/env python3
"""
QBES Automated Setup Script

This script provides a complete automated setup experience for QBES,
including system checking, environment creation, installation, and verification.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the scripts directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from check_system import SystemChecker
from install_qbes import QBESInstaller
from manage_env import EnvironmentManager
from verify_installation import InstallationVerifier


class QBESSetup:
    """Complete QBES setup orchestrator."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the setup orchestrator."""
        self.verbose = verbose
        self.system_checker = SystemChecker()
        self.installer = QBESInstaller()
        self.env_manager = EnvironmentManager()
        self.verifier = InstallationVerifier(verbose=verbose)
    
    def _log(self, message: str, level: str = 'INFO'):
        """Log setup messages."""
        if self.verbose or level in ['ERROR', 'WARNING', 'SUCCESS']:
            prefix = {
                'INFO': '  ',
                'SUCCESS': '✓ ',
                'WARNING': '⚠ ',
                'ERROR': '✗ '
            }.get(level, '  ')
            print(f"{prefix}{message}")
    
    def run_system_check(self) -> bool:
        """Run comprehensive system compatibility check."""
        print("Step 1: System Compatibility Check")
        print("-" * 40)
        
        success = self.system_checker.run_all_checks()
        compatible = self.system_checker.print_summary()
        
        if not compatible:
            self._log("System compatibility check failed. Please resolve issues before continuing.", 'ERROR')
            return False
        
        self._log("System compatibility check passed", 'SUCCESS')
        return True
    
    def setup_environment(self, env_name: str, env_path: str = None) -> bool:
        """Set up virtual environment."""
        print(f"\nStep 2: Environment Setup")
        print("-" * 40)
        
        # Check if environment already exists
        if env_name in self.env_manager.environments:
            response = input(f"Environment '{env_name}' already exists. Recreate it? (y/N): ")
            if response.lower() == 'y':
                self._log(f"Removing existing environment '{env_name}'", 'INFO')
                if not self.env_manager.remove_environment(env_name, force=True):
                    self._log("Failed to remove existing environment", 'ERROR')
                    return False
            else:
                self._log(f"Using existing environment '{env_name}'", 'INFO')
                return True
        
        # Create new environment
        self._log(f"Creating virtual environment '{env_name}'", 'INFO')
        success = self.env_manager.create_environment(env_name, env_path)
        
        if not success:
            self._log("Environment creation failed", 'ERROR')
            return False
        
        self._log("Environment created successfully", 'SUCCESS')
        return True
    
    def install_qbes(self, env_name: str, development_mode: bool = False, 
                    extras: list = None) -> bool:
        """Install QBES in the environment."""
        print(f"\nStep 3: QBES Installation")
        print("-" * 40)
        
        # Install dependencies first using the installer
        env_info = self.env_manager.environments[env_name]
        env_path = env_info['path']
        
        self._log("Installing dependencies...", 'INFO')
        if not self.installer.install_dependencies(env_path):
            self._log("Dependency installation failed", 'ERROR')
            return False
        
        # Install QBES package
        self._log("Installing QBES package...", 'INFO')
        success = self.env_manager.install_qbes(env_name, development_mode, extras)
        
        if not success:
            self._log("QBES installation failed", 'ERROR')
            return False
        
        self._log("QBES installed successfully", 'SUCCESS')
        return True
    
    def verify_installation(self, env_name: str) -> bool:
        """Verify the installation."""
        print(f"\nStep 4: Installation Verification")
        print("-" * 40)
        
        # Switch to the environment for verification
        env_info = self.env_manager.environments[env_name]
        env_path = Path(env_info['path'])
        
        # Temporarily modify sys.path to use the environment
        if sys.platform == 'win32':
            env_python = env_path / 'Scripts' / 'python.exe'
        else:
            env_python = env_path / 'bin' / 'python'
        
        # Run verification in the environment
        import subprocess
        try:
            result = subprocess.run([
                str(env_python), str(Path(__file__).parent / 'verify_installation.py'),
                '--verbose' if self.verbose else ''
            ], capture_output=False, text=True)
            
            success = result.returncode == 0
            
            if success:
                self._log("Installation verification passed", 'SUCCESS')
            else:
                self._log("Installation verification failed", 'ERROR')
            
            return success
            
        except Exception as e:
            self._log(f"Error running verification: {e}", 'ERROR')
            return False
    
    def generate_activation_instructions(self, env_name: str):
        """Generate activation instructions for the user."""
        print(f"\nStep 5: Setup Complete")
        print("-" * 40)
        
        env_info = self.env_manager.environments[env_name]
        env_path = Path(env_info['path'])
        
        print("QBES has been successfully installed!")
        print("\nTo activate your QBES environment:")
        
        if sys.platform == 'win32':
            print(f"  Command Prompt: {env_path / 'Scripts' / 'activate.bat'}")
            print(f"  PowerShell: {env_path / 'Scripts' / 'Activate.ps1'}")
        else:
            print(f"  source {env_path / 'bin' / 'activate'}")
        
        print("\nAlternatively, use the environment manager:")
        print(f"  python scripts/manage_env.py activate {env_name}")
        
        print("\nTo get started with QBES:")
        print("  qbes --help")
        print("  qbes-config generate")
        print("  qbes-benchmark list")
        
        print("\nFor more information, see the documentation in docs/README.md")
    
    def run_complete_setup(self, env_name: str = 'qbes', env_path: str = None,
                          development_mode: bool = False, extras: list = None,
                          skip_system_check: bool = False) -> bool:
        """Run the complete QBES setup process."""
        print("QBES Automated Setup")
        print("=" * 50)
        
        try:
            # Step 1: System check
            if not skip_system_check:
                if not self.run_system_check():
                    return False
            else:
                print("Skipping system compatibility check...")
            
            # Step 2: Environment setup
            if not self.setup_environment(env_name, env_path):
                return False
            
            # Step 3: Installation
            if not self.install_qbes(env_name, development_mode, extras):
                return False
            
            # Step 4: Verification
            if not self.verify_installation(env_name):
                self._log("Installation verification failed, but QBES may still be functional", 'WARNING')
            
            # Step 5: Instructions
            self.generate_activation_instructions(env_name)
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nSetup interrupted by user.")
            return False
        except Exception as e:
            self._log(f"Setup failed with error: {e}", 'ERROR')
            return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description='QBES Automated Setup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup with default environment
  python setup_qbes.py
  
  # Development setup with custom environment name
  python setup_qbes.py --env-name qbes-dev --dev
  
  # Setup with GPU support
  python setup_qbes.py --extras gpu visualization
  
  # Quick setup (skip system check)
  python setup_qbes.py --skip-system-check
        """
    )
    
    parser.add_argument('--env-name', default='qbes',
                       help='Name for the virtual environment (default: qbes)')
    parser.add_argument('--env-path',
                       help='Custom path for the virtual environment')
    parser.add_argument('--dev', action='store_true',
                       help='Install in development mode (editable install)')
    parser.add_argument('--extras', nargs='*', 
                       choices=['dev', 'gpu', 'visualization'],
                       help='Extra dependencies to install')
    parser.add_argument('--skip-system-check', action='store_true',
                       help='Skip system compatibility check')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    # Add subcommands for individual operations
    subparsers = parser.add_subparsers(dest='command', help='Individual setup commands')
    
    # System check only
    check_parser = subparsers.add_parser('check', help='Run system compatibility check only')
    check_parser.add_argument('--save-report', action='store_true',
                             help='Save compatibility report to file')
    
    # Environment management
    env_parser = subparsers.add_parser('env', help='Environment management commands')
    env_subparsers = env_parser.add_subparsers(dest='env_command')
    
    env_create = env_subparsers.add_parser('create', help='Create environment only')
    env_create.add_argument('name', help='Environment name')
    env_create.add_argument('--path', help='Custom environment path')
    
    env_list = env_subparsers.add_parser('list', help='List environments')
    
    # Installation only
    install_parser = subparsers.add_parser('install', help='Install QBES in existing environment')
    install_parser.add_argument('env_name', help='Environment name')
    install_parser.add_argument('--dev', action='store_true', help='Development mode')
    install_parser.add_argument('--extras', nargs='*', help='Extra dependencies')
    
    # Verification only
    verify_parser = subparsers.add_parser('verify', help='Verify existing installation')
    verify_parser.add_argument('--env-name', help='Environment to verify')
    
    args = parser.parse_args()
    
    setup = QBESSetup(verbose=args.verbose)
    
    # Handle subcommands
    if args.command == 'check':
        checker = SystemChecker()
        success = checker.run_all_checks()
        compatible = checker.print_summary()
        
        if args.save_report:
            checker.save_report()
        
        sys.exit(0 if compatible else 1)
    
    elif args.command == 'env':
        if args.env_command == 'create':
            success = setup.env_manager.create_environment(args.name, args.path)
            sys.exit(0 if success else 1)
        elif args.env_command == 'list':
            setup.env_manager.list_environments()
            sys.exit(0)
        else:
            env_parser.print_help()
            sys.exit(1)
    
    elif args.command == 'install':
        success = setup.install_qbes(args.env_name, args.dev, args.extras)
        sys.exit(0 if success else 1)
    
    elif args.command == 'verify':
        if args.env_name:
            success = setup.verify_installation(args.env_name)
        else:
            # Run verification in current environment
            verifier = InstallationVerifier(verbose=args.verbose)
            success = verifier.run_full_verification()
            verifier.print_summary()
        
        sys.exit(0 if success else 1)
    
    else:
        # Run complete setup
        success = setup.run_complete_setup(
            env_name=args.env_name,
            env_path=args.env_path,
            development_mode=args.dev,
            extras=args.extras,
            skip_system_check=args.skip_system_check
        )
        
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()