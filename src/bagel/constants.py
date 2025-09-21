"""
Key amino acid constants

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

# {1-letter: 3-letter} aminoacid names
aa_dict = {
    'A': 'ALA',  # alanine
    'R': 'ARG',  # arginine
    'N': 'ASN',  # asparagine
    'D': 'ASP',  # aspartic acid
    'C': 'CYS',  # cysteine (not mutatable by default)
    'Q': 'GLN',  # glutamine
    'E': 'GLU',  # glutamic acid
    'G': 'GLY',  # glycine
    'H': 'HIS',  # histidine
    'I': 'ILE',  # isoleucine
    'L': 'LEU',  # leucine
    'K': 'LYS',  # lysine
    'M': 'MET',  # methionine
    'F': 'PHE',  # phenylalanine
    'P': 'PRO',  # proline
    'S': 'SER',  # serine
    'T': 'THR',  # threonine
    'W': 'TRP',  # tryptophan
    'Y': 'TYR',  # tyrosine
    'V': 'VAL',  # valine
}

aminoacids_letters = list(aa_dict.keys())
mutation_bias = {aa: 1.0 / len(aa_dict) for aa in aa_dict.keys()}

mutation_bias_no_cystein = {aa: 1.0 / (len(aa_dict) - 1) if aa != 'C' else 0.0 for aa in aa_dict.keys()}

hydrophobic_residues = ('VAL', 'ILE', 'LEU', 'PHE', 'MET', 'TRP')

backbone_atoms = ('CA', 'N', 'C')

angstrom = 1.0  # Units of measure for distances
nm = 10.0  # nm value in units of measure for distances

# These are the maximum values calculated for different atom types using a probe radius of 1.4 Å,
# GPT suggests Lee&Richards 1971 and Connolly as references.
max_sasa_values = {
    'H': 14.0 * angstrom**2,
    'C': 20.0 * angstrom**2,
    'N': 16.0 * angstrom**2,
    'O': 17.0 * angstrom**2,
    'S': 22.0 * angstrom**2,
    'P': 24.0 * angstrom**2,
}

probe_radius_water = 1.4 * angstrom

# From OpenFold, used in ESMFold
# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    'N',
    'CA',
    'C',
    'CB',
    'O',
    'CG',
    'CG1',
    'CG2',
    'OG',
    'OG1',
    'SG',
    'CD',
    'CD1',
    'CD2',
    'ND1',
    'ND2',
    'OD1',
    'OD2',
    'SD',
    'CE',
    'CE1',
    'CE2',
    'CE3',
    'NE',
    'NE1',
    'NE2',
    'OE1',
    'OE2',
    'CH2',
    'NH1',
    'NH2',
    'OH',
    'CZ',
    'CZ2',
    'CZ3',
    'NZ',
    'OXT',
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
