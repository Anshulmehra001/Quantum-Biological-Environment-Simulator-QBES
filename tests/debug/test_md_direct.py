#!/usr/bin/env python3
"""Direct test of MD engine"""
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from qbes.md_engine import MDEngine

print("="*70)
print("Testing MD Engine Directly")
print("="*70)

# Initialize engine
engine = MDEngine()

# Initialize system
print("\n1. Initializing system...")
system = engine.initialize_system('photosystem.pdb', 'amber14')
print(f"✅ System initialized with {len(list(engine.topology.atoms()))} atoms")
print(f"   Force field name stored: {engine.force_field_name}")

# Setup environment
print("\n2. Setting up environment...")
try:
    engine.setup_environment(system, 'tip3p', 0.15)
    print(f"✅ Environment setup successful!")
    print(f"   Total atoms: {len(list(engine.topology.atoms()))}")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
