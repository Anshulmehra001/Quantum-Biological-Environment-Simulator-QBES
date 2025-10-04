#!/usr/bin/env python3
"""
QBES Environment Management Script

This script provides utilities for managing QBES virtual environments,
including creation, activation, deactivation, and cleanup.
"""

import os
import sys
import subprocess
import platform
import shutil
import venv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EnvironmentManager:
    """Manager for QBES virtual environments."""
    
    def __init__(self):
        """Initialize the environment manager."""
        self.platform = platform.system()
        self.config_file = Path.home() / '.qbes' / 'environments.json'
        self.environments = self._load_environments()
    
    def _load_environments(self) -> Dict[str, Dict]:
        """Load environment configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_environments(self):
        """Save environment configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.environments, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save environment config: {e}")
    
    def _get_venv_paths(self, venv_path: str) -> Dict[str, str]:
        """Get platform-specific virtual environment paths."""
        venv_path = Path(venv_path).resolve()
        
        if self.platform == 'Windows':
            return {
                'python': str(venv_path / 'Scripts' / 'python.exe'),
                'pip': str(venv_path / 'Scripts' / 'pip.exe'),
                'activate': str(venv_path / 'Scripts' / 'activate.bat'),
                'activate_ps': str(venv_path / 'Scripts' / 'Activate.ps1'),
                'deactivate': str(venv_path / 'Scripts' / 'deactivate.bat'),
            }
        else:
            return {
                'python': str(venv_path / 'bin' / 'python'),
                'pip': str(venv_path / 'bin' / 'pip'),
                'activate': str(venv_path / 'bin' / 'activate'),
                'deactivate': 'deactivate',
            }
    
    def create_environment(self, name: str, path: Optional[str] = None, 
                         python_executable: Optional[str] = None) -> bool:
        """Create a new virtual environment."""
        if name in self.environments:
            print(f"Environment '{name}' already exists.")
            return False
        
        if path is None:
            path = Path.home() / '.qbes' / 'envs' / name
        else:
            path = Path(path).resolve()
        
        print(f"Creating environment '{name}' at {path}")
        
        try:
            # Remove existing directory if it exists
            if path.exists():
                print(f"Removing existing directory at {path}")
                shutil.rmtree(path)
            
            # Create virtual environment
            if python_executable:
                # Use specific Python executable
                result = subprocess.run([
                    python_executable, '-m', 'venv', str(path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Failed to create environment: {result.stderr}")
                    return False
            else:
                # Use current Python
                venv.create(path, with_pip=True)
            
            # Get environment paths
            paths = self._get_venv_paths(str(path))
            
            # Upgrade pip
            print("Upgrading pip...")
            result = subprocess.run([
                paths['python'], '-m', 'pip', 'install', '--upgrade', 'pip'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Warning: Failed to upgrade pip: {result.stderr}")
            
            # Store environment info
            self.environments[name] = {
                'path': str(path),
                'python_executable': python_executable or sys.executable,
                'created_with_python': platform.python_version(),
                'platform': self.platform,
                'paths': paths
            }
            
            self._save_environments()
            print(f"✓ Environment '{name}' created successfully")
            return True
            
        except Exception as e:
            print(f"Failed to create environment: {e}")
            return False
    
    def remove_environment(self, name: str, force: bool = False) -> bool:
        """Remove a virtual environment."""
        if name not in self.environments:
            print(f"Environment '{name}' not found.")
            return False
        
        env_path = Path(self.environments[name]['path'])
        
        if not force:
            response = input(f"Are you sure you want to remove environment '{name}' at {env_path}? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return False
        
        try:
            if env_path.exists():
                print(f"Removing environment directory: {env_path}")
                shutil.rmtree(env_path)
            
            del self.environments[name]
            self._save_environments()
            print(f"✓ Environment '{name}' removed successfully")
            return True
            
        except Exception as e:
            print(f"Failed to remove environment: {e}")
            return False
    
    def list_environments(self) -> None:
        """List all managed environments."""
        if not self.environments:
            print("No environments found.")
            return
        
        print("QBES Environments:")
        print("-" * 50)
        
        for name, info in self.environments.items():
            path = Path(info['path'])
            exists = "✓" if path.exists() else "✗"
            python_version = info.get('created_with_python', 'unknown')
            
            print(f"{exists} {name}")
            print(f"    Path: {path}")
            print(f"    Python: {python_version}")
            print(f"    Platform: {info.get('platform', 'unknown')}")
            print()
    
    def activate_environment(self, name: str) -> bool:
        """Generate activation command for an environment."""
        if name not in self.environments:
            print(f"Environment '{name}' not found.")
            return False
        
        env_info = self.environments[name]
        env_path = Path(env_info['path'])
        
        if not env_path.exists():
            print(f"Environment directory does not exist: {env_path}")
            return False
        
        paths = self._get_venv_paths(str(env_path))
        
        print(f"To activate environment '{name}', run:")
        
        if self.platform == 'Windows':
            print(f"  Command Prompt: {paths['activate']}")
            print(f"  PowerShell: {paths['activate_ps']}")
        else:
            print(f"  source {paths['activate']}")
        
        return True
    
    def install_qbes(self, name: str, development_mode: bool = False, 
                    extras: Optional[List[str]] = None) -> bool:
        """Install QBES in the specified environment."""
        if name not in self.environments:
            print(f"Environment '{name}' not found.")
            return False
        
        env_info = self.environments[name]
        paths = self._get_venv_paths(env_info['path'])
        
        if not Path(paths['python']).exists():
            print(f"Python executable not found: {paths['python']}")
            return False
        
        print(f"Installing QBES in environment '{name}'...")
        
        try:
            # Install QBES
            if development_mode:
                cmd = [paths['python'], '-m', 'pip', 'install', '-e', '.']
            else:
                cmd = [paths['python'], '-m', 'pip', 'install', '.']
            
            # Add extras if specified
            if extras:
                extras_str = '[' + ','.join(extras) + ']'
                if development_mode:
                    cmd[-1] = f'.{extras_str}'
                else:
                    cmd[-1] = f'.{extras_str}'
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ QBES installed successfully")
                
                # Update environment info
                self.environments[name]['qbes_installed'] = True
                self.environments[name]['development_mode'] = development_mode
                self.environments[name]['extras'] = extras or []
                self._save_environments()
                
                return True
            else:
                print(f"Failed to install QBES: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error installing QBES: {e}")
            return False
    
    def update_qbes(self, name: str) -> bool:
        """Update QBES installation in the specified environment."""
        if name not in self.environments:
            print(f"Environment '{name}' not found.")
            return False
        
        env_info = self.environments[name]
        
        if not env_info.get('qbes_installed', False):
            print(f"QBES is not installed in environment '{name}'")
            return False
        
        development_mode = env_info.get('development_mode', False)
        extras = env_info.get('extras', [])
        
        print(f"Updating QBES in environment '{name}'...")
        
        if development_mode:
            print("Development mode detected - no update needed (using editable install)")
            return True
        else:
            return self.install_qbes(name, development_mode, extras)
    
    def run_tests(self, name: str, test_args: Optional[List[str]] = None) -> bool:
        """Run tests in the specified environment."""
        if name not in self.environments:
            print(f"Environment '{name}' not found.")
            return False
        
        env_info = self.environments[name]
        paths = self._get_venv_paths(env_info['path'])
        
        print(f"Running tests in environment '{name}'...")
        
        try:
            cmd = [paths['python'], '-m', 'pytest']
            if test_args:
                cmd.extend(test_args)
            
            result = subprocess.run(cmd)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error running tests: {e}")
            return False
    
    def check_environment(self, name: str) -> bool:
        """Check the health of an environment."""
        if name not in self.environments:
            print(f"Environment '{name}' not found.")
            return False
        
        env_info = self.environments[name]
        env_path = Path(env_info['path'])
        paths = self._get_venv_paths(str(env_path))
        
        print(f"Checking environment '{name}'...")
        
        # Check if directory exists
        if not env_path.exists():
            print(f"✗ Environment directory does not exist: {env_path}")
            return False
        
        print(f"✓ Environment directory exists: {env_path}")
        
        # Check if Python executable exists
        if not Path(paths['python']).exists():
            print(f"✗ Python executable not found: {paths['python']}")
            return False
        
        print(f"✓ Python executable found: {paths['python']}")
        
        # Check Python version
        try:
            result = subprocess.run([
                paths['python'], '--version'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✓ Python version: {version}")
            else:
                print(f"✗ Could not get Python version")
                return False
        except Exception as e:
            print(f"✗ Error checking Python version: {e}")
            return False
        
        # Check if QBES is installed
        try:
            result = subprocess.run([
                paths['python'], '-c', 'import qbes; print("QBES version:", qbes.__version__ if hasattr(qbes, "__version__") else "unknown")'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ QBES is installed: {result.stdout.strip()}")
            else:
                print(f"✗ QBES is not installed or has issues")
                return False
        except Exception as e:
            print(f"✗ Error checking QBES installation: {e}")
            return False
        
        print(f"✓ Environment '{name}' is healthy")
        return True
    
    def cleanup_environments(self) -> None:
        """Clean up environments that no longer exist on disk."""
        to_remove = []
        
        for name, info in self.environments.items():
            env_path = Path(info['path'])
            if not env_path.exists():
                to_remove.append(name)
        
        if to_remove:
            print(f"Found {len(to_remove)} environments with missing directories:")
            for name in to_remove:
                print(f"  - {name}: {self.environments[name]['path']}")
            
            response = input("Remove these entries from the configuration? (y/N): ")
            if response.lower() == 'y':
                for name in to_remove:
                    del self.environments[name]
                self._save_environments()
                print(f"✓ Removed {len(to_remove)} environment entries")
            else:
                print("Cleanup cancelled")
        else:
            print("No cleanup needed - all environments exist on disk")


def main():
    """Main function for environment management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QBES Environment Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create environment
    create_parser = subparsers.add_parser('create', help='Create a new environment')
    create_parser.add_argument('name', help='Environment name')
    create_parser.add_argument('--path', help='Custom path for environment')
    create_parser.add_argument('--python', help='Python executable to use')
    
    # Remove environment
    remove_parser = subparsers.add_parser('remove', help='Remove an environment')
    remove_parser.add_argument('name', help='Environment name')
    remove_parser.add_argument('--force', action='store_true', help='Force removal without confirmation')
    
    # List environments
    list_parser = subparsers.add_parser('list', help='List all environments')
    
    # Activate environment
    activate_parser = subparsers.add_parser('activate', help='Show activation command for environment')
    activate_parser.add_argument('name', help='Environment name')
    
    # Install QBES
    install_parser = subparsers.add_parser('install', help='Install QBES in environment')
    install_parser.add_argument('name', help='Environment name')
    install_parser.add_argument('--dev', action='store_true', help='Install in development mode')
    install_parser.add_argument('--extras', nargs='*', help='Extra dependencies to install')
    
    # Update QBES
    update_parser = subparsers.add_parser('update', help='Update QBES in environment')
    update_parser.add_argument('name', help='Environment name')
    
    # Run tests
    test_parser = subparsers.add_parser('test', help='Run tests in environment')
    test_parser.add_argument('name', help='Environment name')
    test_parser.add_argument('test_args', nargs='*', help='Additional test arguments')
    
    # Check environment
    check_parser = subparsers.add_parser('check', help='Check environment health')
    check_parser.add_argument('name', help='Environment name')
    
    # Cleanup environments
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up missing environments')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = EnvironmentManager()
    
    if args.command == 'create':
        success = manager.create_environment(args.name, args.path, args.python)
        sys.exit(0 if success else 1)
    
    elif args.command == 'remove':
        success = manager.remove_environment(args.name, args.force)
        sys.exit(0 if success else 1)
    
    elif args.command == 'list':
        manager.list_environments()
    
    elif args.command == 'activate':
        success = manager.activate_environment(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == 'install':
        success = manager.install_qbes(args.name, args.dev, args.extras)
        sys.exit(0 if success else 1)
    
    elif args.command == 'update':
        success = manager.update_qbes(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == 'test':
        success = manager.run_tests(args.name, args.test_args)
        sys.exit(0 if success else 1)
    
    elif args.command == 'check':
        success = manager.check_environment(args.name)
        sys.exit(0 if success else 1)
    
    elif args.command == 'cleanup':
        manager.cleanup_environments()


if __name__ == '__main__':
    main()