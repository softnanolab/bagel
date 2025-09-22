import numpy as np
from biotite.structure.io import load_structure

from bagel.utils import get_atomarray_in_residue_range, sequence_from_atomarray
from bagel.constants import aa_dict


def test_sequence_from_atomarray(pdb_path):
    structure = load_structure(pdb_path)

    expected_sequence = 'PYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGRTLGLYGKDQQEAALVDMVNDGVEDLRCKYISLIYTNYEAGKDDYVKALPGQLKPFETLLSQNQGGKTFIVGDQISFADYNLLDLLLIHEVLAPGCLDAFPLLSAYVGRLSARPKLKAFLASPEYVNLPINGNGKQ'

    for chain_identifier in np.unique(structure.chain_id):
        chain_atoms = structure[structure.chain_id == chain_identifier]
        observed_sequence = sequence_from_atomarray(chain_atoms)
        assert observed_sequence == expected_sequence


def test_get_atomarray_in_residue_range(pdb_path):
    structure = load_structure(pdb_path)

    chain_identifier = 'A'
    start_residue_id = 200
    end_residue_id = 209

    subset_atoms = get_atomarray_in_residue_range(
        structure,
        start=start_residue_id,
        end=end_residue_id,
        chain=chain_identifier,
    )

    # Unique residue ids in ascending order and their first-name occurrence
    unique_residue_ids, first_indices = np.unique(subset_atoms.res_id, return_index=True)
    unique_residue_names = subset_atoms.res_name[first_indices]
    unique_chain_ids = subset_atoms.chain_id[first_indices]

    # Expected from the provided PDB excerpt for chain A 200-209
    expected_residue_ids = np.arange(start_residue_id, end_residue_id + 1)
    expected_residue_names = np.array(['ASN', 'LEU', 'PRO', 'ILE', 'ASN', 'GLY', 'ASN', 'GLY', 'LYS', 'GLN'])

    assert np.array_equal(unique_chain_ids, np.array([chain_identifier] * len(expected_residue_ids)))
    assert np.array_equal(unique_residue_ids, expected_residue_ids)
    assert np.array_equal(unique_residue_names, expected_residue_names)
