import openmm.app as app

ff = app.ForceField('amber14-all.xml', 'tip3p.xml')
print("Checking for ion templates...")
print(f"Total templates: {len(ff._templates)}")

# Check for ion-related templates
for key, template in ff._templates.items():
    if 'NA' in str(key) or 'CL' in str(key) or 'na' in str(key).lower() or 'cl' in str(key).lower():
        print(f"  - {key}: {template}")

# Also check residue templates
print("\nAll template keys (first 30):")
for i, key in enumerate(list(ff._templates.keys())[:30]):
    print(f"  {i+1}. {key}")
