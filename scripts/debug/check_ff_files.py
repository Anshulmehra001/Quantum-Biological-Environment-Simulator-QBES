import openmm.app as app
import os
import glob

ff_dir = os.path.dirname(app.__file__)
print(f'Force field directory: {ff_dir}')

data_dir = os.path.join(ff_dir, 'data')
files = glob.glob(os.path.join(data_dir, '*.xml'))

print('\nAvailable XML files:')
for f in sorted(files):
    basename = os.path.basename(f)
    if 'amber' in basename.lower() or 'tip' in basename.lower() or 'ion' in basename.lower():
        print(f'  - {basename}')
