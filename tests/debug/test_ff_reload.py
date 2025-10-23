#!/usr/bin/env python3
import openmm.app as app
import openmm.unit as unit

# Test 1: Load force field with water first
print("Test 1: Loading force field with water from the start")
try:
    ff1 = app.ForceField('amber14-all.xml', 'tip3p.xml')
    print("✅ Force field loaded successfully")
    
    pdb = app.PDBFile('photosystem.pdb')
    print(f"✅ PDB loaded with {len(list(pdb.topology.atoms()))} atoms")
    
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff1)
    print(f"✅ Hydrogens added: {len(list(modeller.topology.atoms()))} atoms")
    
    # Now try solvation
    modeller.addSolvent(ff1, model='tip3p', padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)
    print(f"✅ Solvation successful: {len(list(modeller.topology.atoms()))} atoms")
    
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*70)
print("Test 2: Load force field, then reload with water")
try:
    # Step 1: Load without water
    ff2 = app.ForceField('amber14-all.xml')
    print("✅ Force field loaded (no water)")
    
    pdb2 = app.PDBFile('photosystem.pdb')
    modeller2 = app.Modeller(pdb2.topology, pdb2.positions)
    modeller2.addHydrogens(ff2)
    print(f"✅ Hydrogens added: {len(list(modeller2.topology.atoms()))} atoms")
    
    # Step 2: Reload with water
    ff2 = app.ForceField('amber14-all.xml', 'tip3p.xml')
    print("✅ Force field reloaded with water")
    
    # Now try solvation
    modeller2.addSolvent(ff2, model='tip3p', padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)
    print(f"✅ Solvation successful: {len(list(modeller2.topology.atoms()))} atoms")
    
except Exception as e:
    print(f"❌ Failed: {e}")
