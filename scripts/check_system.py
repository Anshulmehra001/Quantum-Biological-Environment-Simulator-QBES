#!/usr/bin/env python3
"""
QBES System Compatibility Checker

This script checks if the current system meets the requirements for running QBES.
"""

import sys
import platform
import subprocess
import shutil
import importlib.util
from typing import Dict, List, Tuple


class SystemChecker:
    """System compatibility checker for QBES."""
    
    def __init__(self):
        """Initialize the system checker."""
        self.results = {}
        self.warnings = []
        self.errors = []
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            self.results['python_version'] = {
                'status': 'PASS',
                'current': f"{version.major}.{version.minor}.{version.micro}",
                'required': f"{min_version[0]}.{min_version[1]}+",
                'message': f"Python {version.major}.{version.minor} is compatible"
            }
            return True
        else:
            self.results['python_version'] = {
                'status': 'FAIL',
                'current': f"{version.major}.{version.minor}.{version.micro}",
                'required': f"{min_version[0]}.{min_version[1]}+",
                'message': f"Python {version.major}.{version.minor} is too old. Please upgrade to Python {min_version[0]}.{min_version[1]} or higher."
            }
            self.errors.append("Python version too old")
            return False
    
    def check_platform_support(self) -> bool:
        """Check if the platform is supported."""
        system = platform.system()
        architecture = platform.machine()
        
        supported_platforms = {
            'Windows': ['AMD64', 'x86_64'],
            'Linux': ['x86_64', 'aarch64'],
            'Darwin': ['x86_64', 'arm64']  # macOS
        }
        
        if system in supported_platforms:
            if architecture in supported_platforms[system]:
                self.results['platform'] = {
                    'status': 'PASS',
                    'system': system,
                    'architecture': architecture,
                    'message': f"{system} {architecture} is fully supported"
                }
                return True
            else:
                self.results['platform'] = {
                    'status': 'WARNING',
                    'system': system,
                    'architecture': architecture,
                    'message': f"{system} {architecture} may have limited support"
                }
                self.warnings.append(f"Architecture {architecture} may not be fully supported")
                return True
        else:
            self.results['platform'] = {
                'status': 'WARNING',
                'system': system,
                'architecture': architecture,
                'message': f"{system} is not officially supported"
            }
            self.warnings.append(f"Platform {system} is not officially supported")
            return True
    
    def check_memory(self) -> bool:
        """Check available system memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            min_memory_gb = 4.0
            recommended_memory_gb = 8.0
            
            if total_gb >= recommended_memory_gb:
                status = 'PASS'
                message = f"{total_gb:.1f} GB RAM available (recommended: {recommended_memory_gb} GB)"
            elif total_gb >= min_memory_gb:
                status = 'WARNING'
                message = f"{total_gb:.1f} GB RAM available (minimum: {min_memory_gb} GB, recommended: {recommended_memory_gb} GB)"
                self.warnings.append("Low memory may affect performance")
            else:
                status = 'FAIL'
                message = f"{total_gb:.1f} GB RAM available (minimum required: {min_memory_gb} GB)"
                self.errors.append("Insufficient memory")
            
            self.results['memory'] = {
                'status': status,
                'total_gb': total_gb,
                'available_gb': available_gb,
                'message': message
            }
            
            return status != 'FAIL'
            
        except ImportError:
            self.results['memory'] = {
                'status': 'UNKNOWN',
                'message': "Cannot check memory (psutil not available)"
            }
            return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage('.')
            free_gb = free / (1024**3)
            
            min_space_gb = 2.0
            recommended_space_gb = 10.0
            
            if free_gb >= recommended_space_gb:
                status = 'PASS'
                message = f"{free_gb:.1f} GB free space (recommended: {recommended_space_gb} GB)"
            elif free_gb >= min_space_gb:
                status = 'WARNING'
                message = f"{free_gb:.1f} GB free space (minimum: {min_space_gb} GB, recommended: {recommended_space_gb} GB)"
                self.warnings.append("Low disk space may affect simulation output")
            else:
                status = 'FAIL'
                message = f"{free_gb:.1f} GB free space (minimum required: {min_space_gb} GB)"
                self.errors.append("Insufficient disk space")
            
            self.results['disk_space'] = {
                'status': status,
                'free_gb': free_gb,
                'message': message
            }
            
            return status != 'FAIL'
            
        except Exception as e:
            self.results['disk_space'] = {
                'status': 'UNKNOWN',
                'message': f"Cannot check disk space: {str(e)}"
            }
            return True
    
    def check_system_tools(self) -> bool:
        """Check for required system tools."""
        required_tools = ['git']
        optional_tools = ['gcc', 'g++', 'make']
        
        tool_results = {}
        all_required_present = True
        
        # Check required tools
        for tool in required_tools:
            if shutil.which(tool):
                tool_results[tool] = {'status': 'FOUND', 'required': True}
            else:
                tool_results[tool] = {'status': 'MISSING', 'required': True}
                all_required_present = False
                self.errors.append(f"Required tool '{tool}' not found")
        
        # Check optional tools
        for tool in optional_tools:
            if shutil.which(tool):
                tool_results[tool] = {'status': 'FOUND', 'required': False}
            else:
                tool_results[tool] = {'status': 'MISSING', 'required': False}
                self.warnings.append(f"Optional tool '{tool}' not found (may limit some functionality)")
        
        self.results['system_tools'] = {
            'status': 'PASS' if all_required_present else 'FAIL',
            'tools': tool_results,
            'message': 'All required tools found' if all_required_present else 'Some required tools missing'
        }
        
        return all_required_present
    
    def check_python_packages(self) -> bool:
        """Check for critical Python packages."""
        critical_packages = ['pip', 'setuptools', 'wheel']
        important_packages = ['numpy', 'scipy', 'matplotlib']
        
        package_results = {}
        all_critical_present = True
        
        for package in critical_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    package_results[package] = {'status': 'FOUND', 'critical': True}
                else:
                    package_results[package] = {'status': 'MISSING', 'critical': True}
                    all_critical_present = False
                    self.errors.append(f"Critical package '{package}' not found")
            except ImportError:
                package_results[package] = {'status': 'MISSING', 'critical': True}
                all_critical_present = False
                self.errors.append(f"Critical package '{package}' not found")
        
        for package in important_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    package_results[package] = {'status': 'FOUND', 'critical': False}
                else:
                    package_results[package] = {'status': 'MISSING', 'critical': False}
                    self.warnings.append(f"Important package '{package}' not found")
            except ImportError:
                package_results[package] = {'status': 'MISSING', 'critical': False}
                self.warnings.append(f"Important package '{package}' not found")
        
        self.results['python_packages'] = {
            'status': 'PASS' if all_critical_present else 'FAIL',
            'packages': package_results,
            'message': 'All critical packages found' if all_critical_present else 'Some critical packages missing'
        }
        
        return all_critical_present
    
    def check_network_connectivity(self) -> bool:
        """Check network connectivity for package downloads."""
        try:
            import urllib.request
            
            test_urls = [
                'https://pypi.org',
                'https://github.com'
            ]
            
            connectivity_results = {}
            any_connection = False
            
            for url in test_urls:
                try:
                    response = urllib.request.urlopen(url, timeout=10)
                    if response.getcode() == 200:
                        connectivity_results[url] = 'ACCESSIBLE'
                        any_connection = True
                    else:
                        connectivity_results[url] = 'ERROR'
                except Exception as e:
                    connectivity_results[url] = f'FAILED: {str(e)}'
            
            self.results['network'] = {
                'status': 'PASS' if any_connection else 'WARNING',
                'urls': connectivity_results,
                'message': 'Network connectivity OK' if any_connection else 'Network connectivity issues detected'
            }
            
            if not any_connection:
                self.warnings.append("Network connectivity issues may prevent package installation")
            
            return True
            
        except Exception as e:
            self.results['network'] = {
                'status': 'UNKNOWN',
                'message': f"Cannot check network connectivity: {str(e)}"
            }
            return True
    
    def run_all_checks(self) -> bool:
        """Run all system compatibility checks."""
        print("QBES System Compatibility Check")
        print("=" * 50)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Platform Support", self.check_platform_support),
            ("System Memory", self.check_memory),
            ("Disk Space", self.check_disk_space),
            ("System Tools", self.check_system_tools),
            ("Python Packages", self.check_python_packages),
            ("Network Connectivity", self.check_network_connectivity)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            print(f"\nChecking {check_name}...")
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def print_summary(self):
        """Print a summary of all check results."""
        print("\n" + "=" * 50)
        print("SYSTEM COMPATIBILITY SUMMARY")
        print("=" * 50)
        
        # Print results by category
        for category, result in self.results.items():
            status = result['status']
            message = result['message']
            
            if status == 'PASS':
                print(f"✓ {category.replace('_', ' ').title()}: {message}")
            elif status == 'WARNING':
                print(f"⚠ {category.replace('_', ' ').title()}: {message}")
            elif status == 'FAIL':
                print(f"✗ {category.replace('_', ' ').title()}: {message}")
            else:
                print(f"? {category.replace('_', ' ').title()}: {message}")
        
        # Print warnings and errors
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")
        
        # Overall assessment
        print("\n" + "-" * 50)
        if not self.errors:
            if not self.warnings:
                print("✓ System is fully compatible with QBES")
                print("  You can proceed with installation.")
            else:
                print("⚠ System is compatible with QBES but has some warnings")
                print("  Installation should work but some features may be limited.")
        else:
            print("✗ System has compatibility issues")
            print("  Please resolve the errors before installing QBES.")
        
        return len(self.errors) == 0
    
    def save_report(self, filename: str = 'qbes_system_check.txt'):
        """Save the compatibility report to a file."""
        try:
            with open(filename, 'w') as f:
                f.write("QBES System Compatibility Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("System Information:\n")
                f.write(f"  Platform: {platform.system()} {platform.release()}\n")
                f.write(f"  Architecture: {platform.machine()}\n")
                f.write(f"  Python: {platform.python_version()}\n")
                f.write(f"  Python Executable: {sys.executable}\n\n")
                
                f.write("Check Results:\n")
                for category, result in self.results.items():
                    f.write(f"\n{category.replace('_', ' ').title()}:\n")
                    f.write(f"  Status: {result['status']}\n")
                    f.write(f"  Message: {result['message']}\n")
                    
                    # Add detailed information if available
                    for key, value in result.items():
                        if key not in ['status', 'message']:
                            f.write(f"  {key}: {value}\n")
                
                if self.warnings:
                    f.write(f"\nWarnings:\n")
                    for warning in self.warnings:
                        f.write(f"  - {warning}\n")
                
                if self.errors:
                    f.write(f"\nErrors:\n")
                    for error in self.errors:
                        f.write(f"  - {error}\n")
                
                f.write(f"\nOverall Status: {'COMPATIBLE' if not self.errors else 'INCOMPATIBLE'}\n")
            
            print(f"\nCompatibility report saved to: {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False


def main():
    """Main function for system compatibility checking."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QBES System Compatibility Checker')
    parser.add_argument('--save-report', action='store_true',
                       help='Save compatibility report to file')
    parser.add_argument('--report-file', default='qbes_system_check.txt',
                       help='Report filename (default: qbes_system_check.txt)')
    
    args = parser.parse_args()
    
    checker = SystemChecker()
    
    # Run all checks
    all_passed = checker.run_all_checks()
    
    # Print summary
    compatible = checker.print_summary()
    
    # Save report if requested
    if args.save_report:
        checker.save_report(args.report_file)
    
    # Exit with appropriate code
    sys.exit(0 if compatible else 1)


if __name__ == '__main__':
    main()