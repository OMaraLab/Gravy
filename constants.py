import enum


ir_intervals = {
    # bond stretching IR absorption intervals in cm^-1
    # https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Spectroscopic_Reference_Tables/Infrared_Spectroscopy_Absorption_Table
    # supplemented with https://orgchemboulder.com/Spectroscopy/specttutor/irchart.shtml
    "ALKANE_C_H": (2840, 3000),
    "ALCOHOL_O_H": (3584, 3700),
    "CARBOXYLIC_C_O": (1210, 1320),
    "CARBOXYLIC_C_DOUBLE_O": (1750, 1770),
    "PRIMARY_AMINE_N_H": (3400, 3500),
    "AMMONIUM_N_H": (2800, 3000),
    "ALIPHATIC_C_F": (1000, 1400),
    "THIOL_S_H": (2550, 2600),
}


fg_map = {
    # functional group and neighbourhood mappings
    "ALKANE_C_H": [("C4", "C4H1H1H1"), ("C4", "C4C4H1H1")],
    "ALCOHOL_O_H": [("C4H1", "O2")],
    "CARBOXYLIC_SALT_C_DOUBLE_O": [("C3", "C4O1O1")],
    "CARBOXYLIC_C_O": [("C3H1", "C4O1O2")],
    "CARBOXYLIC_C_DOUBLE_O": [("C3", "C4O1O2")],
    "PRIMARY_AMINE_N_H": [("C4H1H1", "N3")],
    "AMMONIUM_N_H": [("C4H1H1H1", "N4")],
    "ALIPHATIC_C_F": [("C4", "C4F1F1F1"), ("C4", "C4C4F1F1")],
    "THIOL_S_H": [("C4H1", "S2")],
}


COVALENT_BOND_ORDER_THRESHOLD = 0.5


# The node feature vector created by get_bond_features() for each pair of atoms in a bond is formatted as follows (verified by looking at get_bond_features()):
class nodeFeatures(enum.IntEnum):
    ATOM1_ID = 0
    ATOM1_ELEMENT = 1
    ATOM1_ATOMIC_NUMBER = 2
    ATOM1_RADIUS = 3
    ATOM1_MASS = 4
    ATOM1_ELECTRONEGATIVITY = 5
    ATOM1_HYBRIDISATION = 6
    ATOM2_ID = 7
    ATOM2_ELEMENT = 8
    ATOM2_ATOMIC_NUMBER = 9
    ATOM2_RADIUS = 10
    ATOM2_MASS = 11
    ATOM2_ELECTRONEGATIVITY = 12
    ATOM2_HYBRIDISATION = 13
    BOND_LENGTH = 14
    BOND_ORDER = 15
    FIRST_DEGREE_NEIGHBOURS = 16
