#!/usr/bin/env python3
"""
Test script to verify hydrogen addition fix
"""
import sys
from qbes.config_manager import ConfigurationManager
from qbes.simulation_engine import SimulationEngine

def test_photosystem_simulation():
    """Test the photosystem simulation with hydrogen addition"""
    
    print("=" * 70)
    print("🧪 Testing Hydrogen Addition Fix")
    print("=" * 70)
    
    config_file = "quick_photosystem_20251023_193742.yaml"
    
    try:
        # Load configuration
        print(f"\n📂 Loading configuration: {config_file}")
        config_manager = ConfigurationManager()
        config = config_manager.load_config(config_file)
        print("✅ Configuration loaded successfully")
        
        # Initialize simulation engine
        print("\n🔧 Initializing simulation engine...")
        engine = SimulationEngine()
        
        # Initialize with config - this will test hydrogen addition
        print("\n🧬 Initializing molecular system (adding hydrogens)...")
        validation = engine.initialize_simulation(config)
        
        if validation.is_valid:
            print("✅ System initialized successfully!")
            print(f"   - Total atoms with hydrogens: {len(list(engine.md_engine.topology.atoms()))}")
            print(f"   - Quantum subsystem size: {len(engine.quantum_subsystem.atoms) if engine.quantum_subsystem else 'N/A'}")
            print("\n✨ Hydrogen addition fix is working correctly!")
            return True
        else:
            print("❌ Validation failed:")
            for error in validation.errors:
                print(f"   - {error}")
            return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_photosystem_simulation()
    sys.exit(0 if success else 1)
