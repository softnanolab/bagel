"""Unit tests for the SAE feature oracle and its result object."""

from dataclasses import dataclass

import numpy as np
import pytest

import bagel as bg
from bagel.oracles.base import OraclesResultDict
from bagel.oracles.embedding.sae import SAE, SAEResult


@dataclass
class _FakeSAEOutput:
    """Mimics boileroom's SAEFeaturesOutput for the fields the oracle reads."""

    pooled_features: np.ndarray
    features: np.ndarray | None = None
    layer: int = 27
    sae_model: str = 'test/sae'
    chain_index: np.ndarray | None = None
    residue_index: np.ndarray | None = None


class _FakeBoilerSAE:
    def __init__(self, output: _FakeSAEOutput) -> None:
        self._output = output
        self.calls: list = []

    def embed(self, sequences, options=None):  # noqa: ANN001
        self.calls.append(sequences)
        return self._output


def _make_oracle(output: _FakeSAEOutput) -> SAE:
    oracle = SAE.__new__(SAE)
    oracle.backend = 'modal'
    oracle.device = None
    oracle.result_class = SAEResult
    oracle.model = _FakeBoilerSAE(output)
    return oracle


class TestSAEResult:
    def test_stores_feature_vector(self):
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=0)])]
        features = np.arange(16, dtype=np.float64)
        result = SAEResult(input_chains=chains, features=features)
        assert result.features.shape == (16,)
        assert result.embeddings is None
        assert result.input_chains == chains

    def test_result_in_oracles_dict(self):
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=0)])]
        oracle = SAE.__new__(SAE)
        oracle.result_class = SAEResult
        result = SAEResult(input_chains=chains, features=np.ones(8, dtype=np.float64))
        oracles_result = OraclesResultDict()
        oracles_result[oracle] = result
        assert oracles_result[oracle].features.shape == (8,)


class TestSAEOracle:
    def test_result_class(self):
        assert SAE.result_class is SAEResult

    def test_pre_process_monomer(self):
        oracle = SAE.__new__(SAE)
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        assert oracle._pre_process(chains) == ['AAA']

    def test_pre_process_multimer(self):
        oracle = SAE.__new__(SAE)
        chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
        chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
        assert oracle._pre_process([chain_a, chain_b]) == ['AAA:GG']

    def test_embed_returns_pooled_feature_vector(self):
        pooled = np.arange(16, dtype=np.float64)[None, :]  # (1, 16)
        oracle = _make_oracle(_FakeSAEOutput(pooled_features=pooled))
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        result = oracle.embed(chains)
        assert isinstance(result, SAEResult)
        assert result.features.shape == (16,)
        np.testing.assert_allclose(result.features, np.arange(16))
        assert result.layer == 27
        assert result.sae_model == 'test/sae'
        # The joined sequence must have been passed to the model.
        assert oracle.model.calls == [['AAA']]

    def test_embed_keeps_per_residue_features_when_present(self):
        pooled = np.ones((1, 4), dtype=np.float64)
        per_residue = np.arange(3 * 4, dtype=np.float64).reshape(1, 3, 4)  # (batch, residues, features)
        oracle = _make_oracle(_FakeSAEOutput(pooled_features=pooled, features=per_residue))
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        result = oracle.embed(chains)
        assert result.embeddings is not None
        assert result.embeddings.shape == (3, 4)

    def test_embed_rejects_batched_pooled_features(self):
        pooled = np.ones((2, 4), dtype=np.float64)  # batch > 1
        oracle = _make_oracle(_FakeSAEOutput(pooled_features=pooled))
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=0)])]
        with pytest.raises(ValueError, match='num_features'):
            oracle.embed(chains)

    def test_embed_persists_unpadded_chain_and_residue_index(self):
        pooled = np.ones((1, 4), dtype=np.float64)
        # boileroom returns (1, residues) padded with -1; two chains of 3 and 2.
        chain_index = np.array([[0, 0, 0, 1, 1, -1]], dtype=np.int64)
        residue_index = np.array([[0, 1, 2, 0, 1, -1]], dtype=np.int64)
        oracle = _make_oracle(
            _FakeSAEOutput(pooled_features=pooled, chain_index=chain_index, residue_index=residue_index)
        )
        chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
        chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
        result = oracle.embed([chain_a, chain_b])
        np.testing.assert_array_equal(result.chain_index, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result.residue_index, [0, 1, 2, 0, 1])

    def test_embed_indices_none_when_absent(self):
        pooled = np.ones((1, 4), dtype=np.float64)
        oracle = _make_oracle(_FakeSAEOutput(pooled_features=pooled))
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        result = oracle.embed(chains)
        assert result.chain_index is None
        assert result.residue_index is None
