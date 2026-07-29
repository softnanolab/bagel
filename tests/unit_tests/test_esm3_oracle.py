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
