'''
This is a settings file which will store variables that are reused in many files.

Makes it easy to modify things across the whole project
'''

from os.path import dirname

######## Directories
BASE_DIR = dirname(__file__)




#### list of the elements
elements = ['Pr', 'Ni', 'Ru', 'Ne', 'Rb', 'Pt', 'La', 'Na', 'Nb', 'Nd',
            'Mg', 'Li', 'Pb', 'Re', 'Tl', 'Lu', 'Pd', 'Ti', 'Te', 'Rh',
            'Tc', 'Sr', 'Ta', 'Be', 'Ba', 'Tb', 'Yb', 'Si', 'Bi', 'W',
            'Gd', 'Fe', 'Br', 'Dy', 'Hf', 'Hg', 'Y', 'He', 'C', 'B', 'P',
            'F', 'I', 'H', 'K', 'Mn', 'O', 'N', 'Kr', 'S', 'U', 'Sn', 'Sm',
            'V', 'Sc', 'Sb', 'Mo', 'Os', 'Se', 'Th', 'Zn', 'Co', 'Ge',
            'Ag', 'Cl', 'Ca', 'Ir', 'Al', 'Ce', 'Cd', 'Ho', 'As', 'Ar',
            'Au', 'Zr', 'Ga', 'In', 'Cs', 'Cr', 'Tm', 'Cu', 'Er']


### neural net config
batch_size = 32
epochs = 200

material_prop_list = ['ael_bulk_modulus_vrh', 'ael_debye_temperature', 'ael_shear_modulus_vrh', 'agl_log10_thermal_expansion_300K',
                      'agl_thermal_conductivity_300K', 'Egap', 'energy_atom']

cbfv_list = ['atom2vec', 'jarvis', 'jarvis_shuffled', 'magpie', 'mat2vec', 'mat_shuffled', 'oliynyk', 'random_400']

material_prop = 'energy_atom'
cbfv = 'atom2vec'


### seed
seed_setting = 5