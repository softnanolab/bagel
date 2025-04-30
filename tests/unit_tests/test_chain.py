import pytest
import bagel as bg


def test_remove_residue_from_start_of_chain(short_chain: bg.Chain) -> None:
    starting_length = len(short_chain.residues)
    short_chain.remove_residue(index=0)
    print(short_chain.residues)
    assert len(short_chain.residues) == starting_length - 1, 'residue not removed from chain'
    assert short_chain.residues[-1].index == starting_length - 2, 'residue indices not updated properly'


def test_remove_residue_from_end_of_chain(short_chain: bg.Chain) -> None:
    starting_length = len(short_chain.residues)
    short_chain.remove_residue(index=-2)
    print(short_chain.residues)
    assert len(short_chain.residues) == starting_length - 1, 'residue not removed from chain'
    assert short_chain.residues[-1].index == starting_length - 2, 'residue indices not updated properly'


def test_add_residue_at_start_of_chain(short_chain: bg.Chain) -> None:
    starting_length = len(short_chain.residues)
    short_chain.add_residue(amino_acid='A', index=0)
    print(short_chain.residues)
    assert len(short_chain.residues) == starting_length + 1, 'residue not added to chain'
    assert short_chain.residues[-1].index == starting_length, 'residue indices not updated properly'


def test_add_residue_at_end_of_chain(short_chain: bg.Chain) -> None:
    starting_length = len(short_chain.residues)
    short_chain.add_residue(amino_acid='V', index=-2)
    print(short_chain.residues)
    assert len(short_chain.residues) == starting_length + 1, 'residue not added to chain'
    assert short_chain.residues[-1].index == starting_length, 'residue indices not updated properly'


def test_mutate_residue_at_end_of_chain(short_chain: bg.Chain) -> None:
    starting_length = len(short_chain.residues)
    short_chain.mutate_residue(index=-2, amino_acid='N')
    print(short_chain.residues)
    assert len(short_chain.residues) == starting_length, "chain changed length when it shouldn't have"
    assert short_chain.residues[-2].name == 'N', 'residue not mutated to desired amino acid'


def test_create_chain_from_pdb(pdb_path: str) -> None:
    chain = bg.Chain.from_pdb(pdb_path, chain_id='A')
    assert len(chain.residues) == 303, 'not all residues detected'
    first_50_residues = 'PTTVVTPPVAGACAALAMLLAAGGGSTLGGVVTVGTTGGGSLLASCLTGG'
    assert chain.sequence[:50] == first_50_residues, 'amino acid identity of residues read in incorrectly'


def test_custom_chain_ids(base_sequence: str) -> None:
    _ = [bg.Residue(name=aa, chain_ID='1234', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    with pytest.raises(AssertionError):
        _ = [bg.Residue(name=aa, chain_ID='12345', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    with pytest.raises(AssertionError):
        _ = [
            bg.Residue(name=aa, chain_ID='TOO_LONG_CHAIN_ID', index=i, mutable=True)
            for i, aa in enumerate(base_sequence)
        ]
