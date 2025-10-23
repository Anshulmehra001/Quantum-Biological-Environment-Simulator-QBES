#!/usr/bin/env python3
"""
QBES Project Functionality Test
Quick test to verify core components are working
"""

import sys
import os
import traceback
from pathlib import Path

def test_imports():
    """Test core module imports"""
    print("🔍 Testing core imports...")
    
    try:
        # Test basic imports
        import numpy as np
        print("  ✅ NumPy")
        
        import scipy
        print("  ✅ SciPy")
        
        # Test QBES imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        from qbes.core.data_models import DensityMatrix, Hamiltonian
        print("  ✅ QBES Data Models")
        
        from qbes.quantum_engine import QuantumEngine
        print("  ✅ QBES Quantum Engine")
        
        from qbes.simulation_engine import SimulationEngine
        print("  ✅ QBES Simulation Engine")
        
        from qbes.config_manager import ConfigurationManager
        print("  ✅ QBES Configuration Manager")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_quantum_engine():
    """Test quantum engine functionality"""
    print("\n🧪 Testing Quantum Engine...")
    
    try:
        from qbes.quantum_engine import QuantumEngine
        import numpy as np
        
        engine = QuantumEngine()
        
        # Test Hamiltonian creation
        hamiltonian = engine.create_two_level_hamiltonian(
            energy_gap=2.0,
            coupling=0.1
        )
        print("  ✅ Two-level Hamiltonian creation")
        
        # Test state initialization
        from qbes.core.data_models import QuantumSubsystem
        
        # Create mock quantum subsystem
        basis_states = ["ground", "excited"]
        coupling_matrix = np.array([[0.0, 0.1], [0.1, 2.0]])
        
        subsystem = QuantumSubsystem(
            atoms=[],
            basis_states=basis_states,
            coupling_matrix=coupling_matrix
        )
        
        initial_state = engine.initialize_state(subsystem, "ground")
        print("  ✅ Quantum state initialization")
        
        # Test purity calculation
        purity = engine.calculate_purity(initial_state)
        print(f"  ✅ Purity calculation: {purity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantum engine test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration management"""
    print("\n⚙️ Testing Configuration Management...")
    
    try:
        from qbes.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        print("  ✅ Configuration manager creation")
        
        # Test default configuration
        default_config = config_manager.create_default_config()
        print("  ✅ Default configuration creation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_cli():
    """Test CLI functionality"""
    print("\n💻 Testing CLI Interface...")
    
    try:
        from qbes.cli import main
        print("  ✅ CLI module import")
        
        return True
        
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False

def test_interactive():
    """Test interactive interface"""
    print("\n🎮 Testing Interactive Interface...")
    
    try:
        # Check if interactive file exists
        interactive_path = Path(__file__).parent / "qbes_interactive.py"
        if interactive_path.exists():
            print("  ✅ Interactive interface file exists")
            
            # Try to import the class
            sys.path.insert(0, str(Path(__file__).parent))
            from qbes_interactive import QBESInteractive
            print("  ✅ Interactive interface class import")
            
            return True
        else:
            print("  ❌ Interactive interface file not found")
            return False
        
    except Exception as e:
        print(f"  ❌ Interactive interface test failed: {e}")
        return False

def test_website():
    """Test website functionality"""
    print("\n🌐 Testing Website Components...")
    
    try:
        website_dir = Path(__file__).parent / "website"
        
        # Check essential files
        essential_files = [
            "index.html",
            "script.js",
            "styles.css",
            "server.py"
        ]
        
        for file in essential_files:
            file_path = website_dir / file
            if file_path.exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} missing")
                return False
        
        # Test server import
        sys.path.insert(0, str(website_dir))
        from server import app
        print("  ✅ Flask server import")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Website test failed: {e}")
        return False

def test_documentation():
    """Test documentation completeness"""
    print("\n📚 Testing Documentation...")
    
    try:
        docs_dir = Path(__file__).parent / "docs"
        
        # Check key documentation files
        key_docs = [
            "README.md",
            "project-overview.md",
            "technical/mathematical-foundations.md",
            "guides/installation.md",
            "guides/getting-started.md"
        ]
        
        for doc in key_docs:
            doc_path = docs_dir / doc
            if doc_path.exists():
                print(f"  ✅ {doc}")
            else:
                print(f"  ❌ {doc} missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Documentation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧬 QBES Project Functionality Test")
    print("=" * 60)
    print("Testing core components and functionality...")
    print()
    
    tests = [
        ("Core Imports", test_imports),
        ("Quantum Engine", test_quantum_engine),
        ("Configuration", test_configuration),
        ("CLI Interface", test_cli),
        ("Interactive Interface", test_interactive),
        ("Website Components", test_website),
        ("Documentation", test_documentation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! QBES is ready to use.")
        grade = "A"
    elif passed >= total * 0.8:
        print("\n✅ Most tests passed! QBES is functional with minor issues.")
        grade = "B+"
    elif passed >= total * 0.6:
        print("\n⚠️  Some tests failed. QBES has functionality but needs attention.")
        grade = "C+"
    else:
        print("\n❌ Many tests failed. QBES needs significant work.")
        grade = "D"
    
    print(f"Project Grade: {grade}")
    
    print("\n💡 Next Steps:")
    print("1. Run interactive demo: python qbes_interactive.py")
    print("2. Start website: python start_website.py")
    print("3. Run benchmarks: python run_benchmarks.py")
    print("4. Check documentation: docs/README.md")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)