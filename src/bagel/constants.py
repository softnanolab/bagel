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

# This dictionary contains hydrophobicity values for each amino acid, based on the GRAVY (Grand Average of Hydropathy) score.
# The values are taken from the Kyte-Doolittle scale, which is commonly used to assess the hydrophobicity of amino acids.
hydropathy_index = {
    'ILE': 4.5,  # Isoleucine
    'VAL': 4.2,  # Valine
    'LEU': 3.8,  # Leucine
    'PHE': 2.8,  # Phenylalanine
    'CYS': 2.5,  # Cysteine
    'MET': 1.9,  # Methionine
    'ALA': 1.8,  # Alanine
    'GLY': -0.4,  # Glycine
    'THR': -0.7,  # Threonine
    'TRP': -0.9,  # Tryptophan
    'SER': -0.8,  # Serine
    'TYR': -1.3,  # Tyrosine
    'PRO': -1.6,  # Proline
    'HIS': -3.2,  # Histidine
    'GLU': -3.5,  # Glutamic acid
    'GLN': -3.5,  # Glutamine
    'ASP': -3.5,  # Aspartic acid
    'ASN': -3.5,  # Asparagine
    'LYS': -3.9,  # Lysine
    'ARG': -4.5,  # Arginine
}

# This maximum value possible for a residue calculated theoretically, could be used for normalization
# The value is based on the largest amino acid (Tryptophan) and assumes all its atoms are fully exposed to solvent,
# which is a theoretical upper bound for SASA of a residue.
# Ref: Tein et al, 2013 https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/ for max SASA values of amino acids
max_residue_sasa = 285.0 * angstrom**2

# This dictionary contains the maximum theoretical SASA values for each amino acid, used for normalization.
max_theoretical_sasa_for_residues = {
    'TRP': 285.0 * angstrom**2,  # Tryptophan
    'ARG': 274.0 * angstrom**2,  # Arginine
    'TYR': 263.0 * angstrom**2,  # Tyrosine
    'PHE': 240.0 * angstrom**2,  # Phenylalanine
    'LYS': 236.0 * angstrom**2,  # Lysine
    'GLN': 225.0 * angstrom**2,  # Glutamine
    'HIS': 224.0 * angstrom**2,  # Histidine
    'MET': 224.0 * angstrom**2,  # Methionine
    'GLU': 223.0 * angstrom**2,  # Glutamate
    'LEU': 201.0 * angstrom**2,  # Leucine
    'ILE': 197.0 * angstrom**2,  # Isoleucine
    'ASN': 195.0 * angstrom**2,  # Asparagine
    'ASP': 193.0 * angstrom**2,  # Aspartate
    'VAL': 174.0 * angstrom**2,  # Valine
    'THR': 172.0 * angstrom**2,  # Threonine
    'CYS': 167.0 * angstrom**2,  # Cysteine
    'PRO': 159.0 * angstrom**2,  # Proline
    'SER': 155.0 * angstrom**2,  # Serine
    'ALA': 129.0 * angstrom**2,  # Alanine
    'GLY': 104.0 * angstrom**2,  # Glycine
}
