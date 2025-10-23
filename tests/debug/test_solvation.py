#!/usr/bin/env python3
import openmm.app as app
import openmm.unit as unit

# Load PDB
pdb = app.PDBFile('photosystem.pdb')

# Create force field with water
ff = app.ForceField('amber14-all.xml', 'tip3p.xml')

# Add hydrogens
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(ff)

print(f"Atoms after adding hydrogens: {len(list(modeller.topology.atoms()))}")

# Try to add solvent with ions
try:
    modeller.addSolvent(ff, model='tip3p', padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)
    print(f"✅ Solvation successful! Total atoms: {len(list(modeller.topology.atoms()))}")
    
    # Now try createSystem like the MD engine does
    print("\nTrying createSystem with PME...")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005
    )
    print(f"✅ createSystem successful!")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

