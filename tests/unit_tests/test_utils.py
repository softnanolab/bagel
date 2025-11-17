import numpy as np
from biotite.structure.io import load_structure
from biotite.structure import AtomArray, array, Atom

from bagel.utils import get_atomarray_in_residue_range, sequence_from_atomarray, get_reconciled_sequence


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

def test_get_reconciled_sequence():
    """Test the get_reconciled_sequence function with various scenarios."""

    def create_fake_atomarray(res_ids, res_names, chain_id='A'):
        atoms_list = []
        for res_id, res_name in zip(res_ids, res_names):
            atoms_list.append(
                Atom(
                    coord=[0.0, 0.0, 0.0],
                    res_id=res_id,
                    res_name=res_name,
                    chain_id=chain_id,
                    atom_name='CA',
                    element='C',
                )
            )
        return array(atoms_list)

    # Test 1: Perfect match - no missing residues, sequences match
    res_ids = [1, 2, 3, 4, 5]
    res_names = ['ALA', 'GLY', 'MET', 'PHE', 'TRP']
    atoms = create_fake_atomarray(res_ids, res_names)
    fasta_seq = 'AGMFW'  # ALA, GLY, MET, PHE, TRP
    result, added = get_reconciled_sequence(atoms, fasta_seq)
    assert result == 'AGMFW', f"Expected 'AGMFW', got '{result}'"
    assert added == False, 'No residues should be added'
    
    # Test 2: Missing residues in AtomArray - should use FASTA to fill gaps
    res_ids = [1, 2, 4, 5]  # Missing residue 3
    res_names = ['ALA', 'GLY', 'PHE', 'TRP']
    atoms = create_fake_atomarray(res_ids, res_names)
    fasta_seq = 'AGMFW'  # Full sequence
    result, added = get_reconciled_sequence(atoms, fasta_seq)
    assert result == 'AGMFW', f"Expected 'AGMFW', got '{result}'"
    assert added == True, 'Residues should be added'
    
    # Test 3: Mismatch between PDB and FASTA - PDB should be preferred
    res_ids = [1, 2, 3]
    res_names = ['ALA', 'GLY', 'MET']  # PDB has MET
    atoms = create_fake_atomarray(res_ids, res_names)
    fasta_seq = 'AGP'  # FASTA has PRO instead of MET
    result, added = get_reconciled_sequence(atoms, fasta_seq)
    assert result == 'AGM', f"Expected 'AGM' (PDB preferred), got '{result}'"
    assert added == False, 'No residues should be added'
    
    # Test 4: Multiple missing residues
    res_ids = [1, 3, 5]  # Missing 2 and 4
    res_names = ['ALA', 'MET', 'TRP']
    atoms = create_fake_atomarray(res_ids, res_names)
    fasta_seq = 'AGMFW'
    result, added = get_reconciled_sequence(atoms, fasta_seq)
    assert result == 'AGMFW', f"Expected 'AGMFW', got '{result}'"
    assert added == True, 'Residues should be added'
    
    # Test 5: Missing residues in the middle and at the end
    res_ids = [1, 2, 5]  # Missing 3 and 4, but has 5 so max_res_id is 5
    res_names = ['ALA', 'GLY', 'TRP']
    atoms = create_fake_atomarray(res_ids, res_names)
    fasta_seq = 'AGMFW'  # Full sequence
    result, added = get_reconciled_sequence(atoms, fasta_seq)
    assert result == 'AGMFW', f"Expected 'AGMFW', got '{result}'"
    assert added == True, 'Residues should be added'
    
    # Test 6: No FASTA sequence provided - should use random for missing residues
    res_ids = [1, 3, 5]
    res_names = ['ALA', 'MET', 'TRP']
    atoms = create_fake_atomarray(res_ids, res_names)
    result, added = get_reconciled_sequence(atoms, None)
    assert len(result) == 5, f'Expected length 5, got {len(result)}'
    assert result[0] == 'A', 'First residue should be A'
    assert result[2] == 'M', 'Third residue should be M'
    assert result[4] == 'W', 'Fifth residue should be W'
    assert added == True, 'Residues should be added'

    # Test 7: Non-standard residue numbering (start not equal to 1)
    res_ids = [10, 12, 13]  # Missing residue 11, numbering starts at 10
    res_names = ['ALA', 'MET', 'PRO']
    atoms = create_fake_atomarray(res_ids, res_names)
    fasta_seq = 'AGMP'  # Covers residues 10-13 inclusive
    result, added = get_reconciled_sequence(atoms, fasta_seq)
    assert result == 'AGMP', f"Expected 'AGMP', got '{result}'"
    assert added == True, 'Residues should be added when numbering is non-contiguous'
