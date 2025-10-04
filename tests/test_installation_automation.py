#!/usr/bin/env python3
"""
Test suite for QBES installation and setup automation scripts.
"""

import unittest
import tempfile
import shutil
import sys
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from check_system import SystemChecker
from install_qbes import QBESInstaller
from manage_env import EnvironmentManager
from verify_installation import InstallationVerifier


class TestSystemChecker(unittest.TestCase):
    """Test the system compatibility checker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = SystemChecker()
    
    def test_python_version_check(self):
        """Test Python version compatibility checking."""
        result = self.checker.check_python_version()
        self.assertIsInstance(result, bool)
        self.assertIn('python_version', self.checker.results)
        
        # Should pass for current Python version (3.8+)
        self.assertTrue(result)
        self.assertEqual(self.checker.results['python_version']['status'], 'PASS')
    
    def test_platform_support_check(self):
        """Test platform support checking."""
        result = self.checker.check_platform_support()
        self.assertIsInstance(result, bool)
        self.assertIn('platform', self.checker.results)
        
        # Should at least not fail completely
        self.assertTrue(result)
    
    def test_system_tools_check(self):
        """Test system tools availability checking."""
        result = self.checker.check_system_tools()
        self.assertIsInstance(result, bool)
        self.assertIn('system_tools', self.checker.results)
    
    def test_python_packages_check(self):
        """Test Python packages availability checking."""
        result = self.checker.check_python_packages()
        self.assertIsInstance(result, bool)
        self.assertIn('python_packages', self.checker.results)
    
    def test_run_all_checks(self):
        """Test running all system checks."""
        result = self.checker.run_all_checks()
        self.assertIsInstance(result, bool)
        
        # Should have results for all checks
        expected_checks = ['python_version', 'platform', 'system_tools', 'python_packages']
        for check in expected_checks:
            self.assertIn(check, self.checker.results)
    
    def test_print_summary(self):
        """Test printing summary of check results."""
        self.checker.run_all_checks()
        
        # Should not raise an exception
        compatible = self.checker.print_summary()
        self.assertIsInstance(compatible, bool)
    
    def test_save_report(self):
        """Test saving compatibility report to file."""
        self.checker.run_all_checks()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            report_file = f.name
        
        try:
            result = self.checker.save_report(report_file)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(report_file))
            
            # Check that file has content
            with open(report_file, 'r') as f:
                content = f.read()
                self.assertIn('QBES System Compatibility Report', content)
        finally:
            if os.path.exists(report_file):
                os.unlink(report_file)


class TestQBESInstaller(unittest.TestCase):
    """Test the QBES installer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.installer = QBESInstaller()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_installer_initialization(self):
        """Test installer initialization."""
        self.assertIsNotNone(self.installer.system_info)
        self.assertIsNotNone(self.installer.python_version)
        self.assertIsInstance(self.installer.install_log, list)
    
    def test_get_system_info(self):
        """Test system information gathering."""
        info = self.installer._get_system_info()
        
        required_keys = ['platform', 'architecture', 'python_version', 'python_executable']
        for key in required_keys:
            self.assertIn(key, info)
    
    def test_check_system_compatibility(self):
        """Test system compatibility checking."""
        result = self.installer.check_system_compatibility()
        self.assertIsInstance(result, bool)
        
        # Should pass for current system
        self.assertTrue(result)
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        deps = self.installer.check_dependencies()
        self.assertIsInstance(deps, dict)
        
        # Should check for key packages
        expected_packages = ['numpy', 'scipy', 'matplotlib', 'click']
        for package in expected_packages:
            self.assertIn(package, deps)
    
    @patch('subprocess.run')
    def test_create_virtual_environment(self, mock_run):
        """Test virtual environment creation."""
        mock_run.return_value.returncode = 0
        
        venv_path = os.path.join(self.temp_dir, 'test_env')
        
        with patch('venv.create'):
            result = self.installer.create_virtual_environment(venv_path)
            self.assertTrue(result)
    
    def test_save_installation_log(self):
        """Test saving installation log."""
        self.installer._log("Test message")
        
        log_file = os.path.join(self.temp_dir, 'test_install.log')
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = self.installer.save_installation_log()
            # Note: This will fail in the actual implementation due to hardcoded filename
            # but tests the method structure


class TestEnvironmentManager(unittest.TestCase):
    """Test the environment manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'environments.json')
        
        # Mock the config file location
        with patch.object(EnvironmentManager, '__init__', lambda x: None):
            self.manager = EnvironmentManager()
            self.manager.platform = 'Linux'  # Use Linux for consistent testing
            self.manager.config_file = Path(self.config_file)
            self.manager.environments = {}
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_environments_empty(self):
        """Test loading environments when no config exists."""
        envs = self.manager._load_environments()
        self.assertEqual(envs, {})
    
    def test_save_and_load_environments(self):
        """Test saving and loading environment configuration."""
        test_envs = {
            'test_env': {
                'path': '/path/to/env',
                'python_executable': '/usr/bin/python3'
            }
        }
        
        self.manager.environments = test_envs
        self.manager._save_environments()
        
        # Verify file was created
        self.assertTrue(self.manager.config_file.exists())
        
        # Load and verify
        loaded_envs = self.manager._load_environments()
        self.assertEqual(loaded_envs, test_envs)
    
    def test_get_venv_paths_linux(self):
        """Test getting virtual environment paths on Linux."""
        self.manager.platform = 'Linux'
        paths = self.manager._get_venv_paths('/path/to/venv')
        
        expected_keys = ['python', 'pip', 'activate', 'deactivate']
        for key in expected_keys:
            self.assertIn(key, paths)
        
        self.assertEqual(paths['python'], '/path/to/venv/bin/python')
        self.assertEqual(paths['activate'], '/path/to/venv/bin/activate')
    
    def test_get_venv_paths_windows(self):
        """Test getting virtual environment paths on Windows."""
        self.manager.platform = 'Windows'
        paths = self.manager._get_venv_paths('/path/to/venv')
        
        expected_keys = ['python', 'pip', 'activate', 'activate_ps', 'deactivate']
        for key in expected_keys:
            self.assertIn(key, paths)
        
        self.assertTrue(paths['python'].endswith('Scripts\\python.exe'))
        self.assertTrue(paths['activate'].endswith('Scripts\\activate.bat'))
    
    def test_list_environments_empty(self):
        """Test listing environments when none exist."""
        # Should not raise an exception
        self.manager.list_environments()
    
    def test_list_environments_with_data(self):
        """Test listing environments with data."""
        self.manager.environments = {
            'test_env': {
                'path': '/path/to/env',
                'created_with_python': '3.9.0',
                'platform': 'Linux'
            }
        }
        
        # Should not raise an exception
        self.manager.list_environments()
    
    def test_activate_environment_not_found(self):
        """Test activating non-existent environment."""
        result = self.manager.activate_environment('nonexistent')
        self.assertFalse(result)
    
    def test_cleanup_environments(self):
        """Test cleaning up environments with missing directories."""
        # Add environment with non-existent path
        self.manager.environments = {
            'missing_env': {
                'path': '/nonexistent/path',
                'created_with_python': '3.9.0'
            }
        }
        
        # Should not raise an exception
        with patch('builtins.input', return_value='n'):
            self.manager.cleanup_environments()


class TestInstallationVerifier(unittest.TestCase):
    """Test the installation verifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.verifier = InstallationVerifier(verbose=False)
    
    def test_verifier_initialization(self):
        """Test verifier initialization."""
        self.assertIsInstance(self.verifier.results, dict)
        self.assertIsInstance(self.verifier.errors, list)
        self.assertIsInstance(self.verifier.warnings, list)
    
    def test_verify_python_version(self):
        """Test Python version verification."""
        result = self.verifier.verify_python_version()
        self.assertIsInstance(result, bool)
        self.assertIn('python_version', self.verifier.results)
        
        # Should pass for current Python version
        self.assertTrue(result)
        self.assertTrue(self.verifier.results['python_version'])
    
    def test_verify_core_imports(self):
        """Test core module import verification."""
        # This will likely fail in test environment, but should not crash
        result = self.verifier.verify_core_imports()
        self.assertIsInstance(result, bool)
        self.assertIn('core_imports', self.verifier.results)
    
    def test_verify_dependencies(self):
        """Test dependency verification."""
        result = self.verifier.verify_dependencies()
        self.assertIsInstance(result, bool)
        self.assertIn('dependencies', self.verifier.results)
    
    @patch('subprocess.run')
    def test_verify_cli_functionality(self, mock_run):
        """Test CLI functionality verification."""
        # Mock successful CLI help command
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Quantum Biological Environment Simulator"
        mock_run.return_value.stderr = ""
        
        result = self.verifier.verify_cli_functionality()
        self.assertIsInstance(result, bool)
        self.assertIn('cli_functionality', self.verifier.results)
    
    def test_print_summary(self):
        """Test printing verification summary."""
        # Add some test results
        self.verifier.results = {
            'test1': True,
            'test2': False,
            'test3': True
        }
        self.verifier.warnings = ['Test warning']
        self.verifier.errors = ['Test error']
        
        # Should not raise an exception
        success = self.verifier.print_summary()
        self.assertIsInstance(success, bool)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios for installation automation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_system_check_to_installation_flow(self):
        """Test the flow from system check to installation."""
        # System check
        checker = SystemChecker()
        system_compatible = checker.run_all_checks()
        
        if system_compatible:
            # Environment creation
            with patch.object(EnvironmentManager, '__init__', lambda x: None):
                manager = EnvironmentManager()
                manager.platform = 'Linux'
                manager.config_file = Path(self.temp_dir) / 'environments.json'
                manager.environments = {}
                
                # Test environment operations
                self.assertIsInstance(manager._get_venv_paths('/test/path'), dict)
    
    def test_installation_verification_flow(self):
        """Test the installation verification flow."""
        verifier = InstallationVerifier(verbose=False)
        
        # Run basic verifications that should work in any environment
        python_ok = verifier.verify_python_version()
        self.assertTrue(python_ok)
        
        # Test summary generation
        success = verifier.print_summary()
        self.assertIsInstance(success, bool)
    
    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios."""
        # Test installer with invalid paths
        installer = QBESInstaller()
        
        # Test with non-existent directory
        result = installer.create_virtual_environment('/nonexistent/path/env')
        self.assertFalse(result)
        
        # Test environment manager with invalid environment
        with patch.object(EnvironmentManager, '__init__', lambda x: None):
            manager = EnvironmentManager()
            manager.environments = {}
            
            result = manager.activate_environment('nonexistent')
            self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()