"""Unit tests for the ESM3 oracle and its track decoders."""

import numpy as np
import pytest

import bagel as bg
from bagel.oracles.embedding.esm3 import (
    ESM3,
    ESM3Result,
    _SASA_BIN_MIDPOINTS,
    _SS8_VOCAB,
    decode_sasa,
    decode_secondary_structure,
)


class TestDecodeSasa:
    def test_onehot_bins_return_bin_midpoints(self):
        n_bins = _SASA_BIN_MIDPOINTS.shape[0]
        logits = np.zeros((2, n_bins))
        logits[0, 3] = 1e3
        logits[1, 0] = 1e3
        sasa = decode_sasa(logits)
        assert sasa.shape == (2,)
        assert np.isclose(sasa[0], _SASA_BIN_MIDPOINTS[3])
        assert np.isclose(sasa[1], _SASA_BIN_MIDPOINTS[0])

    def test_uniform_logits_give_mean_of_midpoints(self):
        logits = np.zeros((1, _SASA_BIN_MIDPOINTS.shape[0]))
        assert np.isclose(decode_sasa(logits)[0], _SASA_BIN_MIDPOINTS.mean())

    def test_leading_special_tokens_are_ignored(self):
        n_bins = _SASA_BIN_MIDPOINTS.shape[0]
        bins = np.zeros((1, n_bins))
        bins[0, 5] = 1e3
        with_specials = np.concatenate([np.full((1, 3), 1e6), bins], axis=1)  # 3 specials + 16 bins
        assert np.isclose(decode_sasa(with_specials)[0], _SASA_BIN_MIDPOINTS[5])


class TestDecodeSecondaryStructure:
    def test_argmax_returns_ss8_string(self):
        n = len(_SS8_VOCAB)
        logits = np.zeros((2, n))
        logits[0, _SS8_VOCAB.index('H')] = 10
        logits[1, _SS8_VOCAB.index('C')] = 10
        assert decode_secondary_structure(logits) == 'HC'

    def test_leading_special_tokens_are_ignored(self):
        n = len(_SS8_VOCAB)
        logits = np.zeros((1, n))
        logits[0, _SS8_VOCAB.index('E')] = 10
        with_specials = np.concatenate([np.full((1, 3), 99), logits], axis=1)
        assert decode_secondary_structure(with_specials) == 'E'


class TestESM3Result:
    def test_stores_decoded_and_raw_tracks(self):
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=0)])]
        result = ESM3Result(
            input_chains=chains,
            embeddings=np.zeros((3, 8)),
            sasa=np.array([1.0, 2.0, 3.0]),
            secondary_structure='HEC',
        )
        assert result.sasa.tolist() == [1.0, 2.0, 3.0]
        assert result.secondary_structure == 'HEC'
        assert result.function_logits is None
        assert result.residue_annotation_logits is None


class TestESM3Oracle:
    def test_result_class(self):
        assert ESM3.result_class is ESM3Result

    def test_pre_process_monomer(self, monkeypatch):
        monkeypatch.setattr(ESM3, '_load', lambda self, config=None: None)
        oracle = ESM3(tracks=['sasa'])
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        assert oracle._pre_process(chains) == ['AAA']

    def test_pre_process_multimer(self, monkeypatch):
        monkeypatch.setattr(ESM3, '_load', lambda self, config=None: None)
        oracle = ESM3(tracks=['sasa'])
        chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
        chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
        assert oracle._pre_process([chain_a, chain_b]) == ['AAA:GG']

    def test_unknown_track_raises(self, monkeypatch):
        monkeypatch.setattr(ESM3, '_load', lambda self, config=None: None)
        with pytest.raises(ValueError, match='Unknown ESM3 tracks'):
            ESM3(tracks=['nope'])

    def test_post_process_decodes_requested_tracks(self, monkeypatch):
        monkeypatch.setattr(ESM3, '_load', lambda self, config=None: None)
        oracle = ESM3(tracks=['sasa', 'secondary_structure'])
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(2)])]

        n_sasa = _SASA_BIN_MIDPOINTS.shape[0]
        sasa_logits = np.zeros((1, 2, n_sasa))
        sasa_logits[0, 0, 2] = 1e3  # residue 0 -> bin 2
        sasa_logits[0, 1, 7] = 1e3  # residue 1 -> bin 7
        ss_logits = np.zeros((1, 2, len(_SS8_VOCAB)))
        ss_logits[0, 0, _SS8_VOCAB.index('H')] = 10
        ss_logits[0, 1, _SS8_VOCAB.index('E')] = 10

        class _Output:
            embeddings = np.zeros((1, 2, 8))

        output = _Output()
        output.sasa_logits = sasa_logits
        output.secondary_structure_logits = ss_logits

        result = oracle._post_process(output, chains)
        assert result.embeddings.shape == (2, 8)
        assert np.allclose(result.sasa, [_SASA_BIN_MIDPOINTS[2], _SASA_BIN_MIDPOINTS[7]])
        assert result.secondary_structure == 'HE'
        assert result.function_logits is None
        assert result.residue_annotation_logits is None

    @pytest.mark.parametrize(
        ('logits', 'message'),
        [
            (np.asarray(1.0), r'sasa_logits must have shape \(1, residues, classes\)'),
            (np.zeros((2, 4)), r'sasa_logits must have shape \(1, residues, classes\)'),
            (np.zeros((2, 4, 16)), r'sasa_logits must have shape \(1, residues, classes\)'),
            (np.full((1, 4, 16), 'invalid'), 'sasa_logits must have a numeric dtype'),
        ],
    )
    def test_post_process_rejects_malformed_track_logits(self, monkeypatch, logits, message):
        monkeypatch.setattr(ESM3, '_load', lambda self, config=None: None)
        oracle = ESM3(tracks=['sasa'])
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(4)])]

        class _Output:
            embeddings = np.zeros((1, 4, 8))
            sasa_logits = logits

        with pytest.raises(ValueError, match=message):
            oracle._post_process(_Output(), chains)


def _backbone_structure(chains, offset_per_chain=100.0):
    """Structure with N, CA, C atoms per residue; coord = (chain_no*100 + residue, atom_no, 0)."""
    from biotite.structure import Atom, array

    atoms = []
    for c, chain in enumerate(chains):
        for residue in chain.residues:
            for a, name in enumerate(['N', 'CA', 'C', 'O']):
                atoms.append(
                    Atom(
                        coord=[c * offset_per_chain + residue.index, a, 0.0],
                        chain_id=chain.chain_ID,
                        res_id=residue.index,
                        res_name='ALA',
                        atom_name=name,
                        element='C',
                    )
                )
    return array(atoms)


def _two_chains():
    chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
    chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
    return [chain_a, chain_b]


class TestBackboneCoordinates:
    def test_extracts_n_ca_c_in_chain_order(self):
        from bagel.oracles.embedding.esm3 import backbone_coordinates

        chains = _two_chains()
        coords = backbone_coordinates(chains, _backbone_structure(chains))
        assert coords.shape == (5, 3, 3)
        assert coords[:, 1, 0].tolist() == [0, 1, 2, 100, 101]  # CA x = chain*100 + residue
        assert coords[0, :, 1].tolist() == [0, 1, 2]  # N, CA, C (no O)

    def test_missing_backbone_atom_is_nan(self):
        from bagel.oracles.embedding.esm3 import backbone_coordinates

        chains = _two_chains()
        structure = _backbone_structure(chains)
        structure = structure[~((structure.chain_id == 'A') & (structure.res_id == 1) & (structure.atom_name == 'C'))]
        coords = backbone_coordinates(chains, structure)
        assert np.isnan(coords[1, 2]).all()
        assert not np.isnan(coords[1, :2]).any()

    def test_residue_count_mismatch_raises(self):
        from bagel.oracles.embedding.esm3 import backbone_coordinates

        chains = _two_chains()
        structure = _backbone_structure(chains)
        structure = structure[~((structure.chain_id == 'B') & (structure.res_id == 1))]
        with pytest.raises(ValueError, match="chain 'B'"):
            backbone_coordinates(chains, structure)


class TestInverseFold:
    @staticmethod
    def _oracle(monkeypatch, logits, calls):
        from types import SimpleNamespace

        monkeypatch.setattr(ESM3, '_load', lambda self, config=None: None)
        oracle = ESM3()

        def inverse_fold(sequence, coordinates, positions):
            calls.append((sequence, coordinates, positions))
            return SimpleNamespace(logits=logits, amino_acids='ACDEFGHIKLMNPQRSTVWY')

        oracle.model = SimpleNamespace(inverse_fold=inverse_fold)
        return oracle

    def test_returns_softmax_probabilities_and_passes_global_position(self, monkeypatch):
        calls = []
        logits = np.zeros((1, 20))
        logits[0, 3] = np.log(19.0)  # 'E' gets 19/(19+19)=0.5 after softmax over 19 ones + 19
        oracle = self._oracle(monkeypatch, logits, calls)
        chains = _two_chains()

        probabilities = oracle.inverse_fold(chains, _backbone_structure(chains), chain_id='B', residue_index=1)

        sequence, coordinates, positions = calls[0]
        assert sequence == 'AAA:GG'
        assert coordinates.shape == (5, 3, 3)
        assert positions == [4]  # chain A has 3 residues, so B[1] -> global 4
        assert list(probabilities) == list('ACDEFGHIKLMNPQRSTVWY')
        assert np.isclose(sum(probabilities.values()), 1.0)
        assert np.isclose(probabilities['E'], 0.5)

    def test_unknown_chain_and_bad_index_raise(self, monkeypatch):
        oracle = self._oracle(monkeypatch, np.zeros((1, 20)), [])
        chains = _two_chains()
        structure = _backbone_structure(chains)
        with pytest.raises(ValueError, match='not found'):
            oracle.inverse_fold(chains, structure, chain_id='Z', residue_index=0)
        with pytest.raises(ValueError, match='out of range'):
            oracle.inverse_fold(chains, structure, chain_id='B', residue_index=2)

    def test_unexpected_logits_shape_raises(self, monkeypatch):
        oracle = self._oracle(monkeypatch, np.zeros((2, 20)), [])
        chains = _two_chains()
        with pytest.raises(ValueError, match='shape'):
            oracle.inverse_fold(chains, _backbone_structure(chains), chain_id='A', residue_index=0)
