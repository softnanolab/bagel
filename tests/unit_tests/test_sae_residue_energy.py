"""Unit tests for the per-residue SAEnergy term (ResidueSAEnergy)."""

import numpy as np
import pytest

import bagel as bg
from bagel.oracles.base import OraclesResultDict
from bagel.oracles.embedding.sae import SAE, SAEResult


def _multichain_chains():
    # Chain A: residues 0,1,2 ; Chain B: residues 0,1
    chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
    chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
    return chain_a, chain_b


def _oracle_with_embeddings(embeddings, chains, chain_index=None, residue_index=None):
    """Return a fake SAE oracle + result holding per-residue ``embeddings``."""
    oracle = SAE.__new__(SAE)
    oracle.result_class = SAEResult
    result = SAEResult(
        input_chains=list(chains),
        features=np.asarray(embeddings, dtype=np.float64).max(axis=0),
        embeddings=np.asarray(embeddings, dtype=np.float64),
        chain_index=None if chain_index is None else np.asarray(chain_index, dtype=int),
        residue_index=None if residue_index is None else np.asarray(residue_index, dtype=int),
    )
    oracles_result = OraclesResultDict()
    oracles_result[oracle] = result
    return oracle, oracles_result


# 5 residues x 4 features; distinctive per-row values.
_EMB = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [7.0, 0.0, 0.0, 0.0],
        [9.0, 0.0, 0.0, 0.0],
    ]
)
_CI = [0, 0, 0, 1, 1]
_RI = [0, 1, 2, 0, 1]


def test_all_residues_mean_pool_default():
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, _RI)
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0])  # pooling='mean'
    unweighted, _ = energy.compute(results)
    # mean of feature 0 over all residues = (1+3+5+7+9)/5 = 5; negated -> -5
    assert unweighted == pytest.approx(-5.0)


def test_max_pool_over_all_residues_matches_whole_protein():
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, _RI)
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], pooling='max')
    assert energy.compute(results)[0] == pytest.approx(-9.0)  # max = 9


def test_sum_pool():
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, _RI)
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], pooling='sum')
    assert energy.compute(results)[0] == pytest.approx(-(1 + 3 + 5 + 7 + 9))


def test_residue_selection_by_residue_objects_multichain():
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, _RI)
    # Select chain B residue 1 -> that's the 5th row (value 9), and chain A residue 0 (value 1).
    selected = [chain_b.residues[1], chain_a.residues[0]]
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], residues=selected, pooling='mean')
    # mean of {1, 9} = 5 -> -5
    assert energy.compute(results)[0] == pytest.approx(-5.0)
    # max of {1, 9} = 9
    energy_max = bg.energies.ResidueSAEnergy(
        oracle=oracle, feature_indices=[0], residues=selected, pooling='max'
    )
    assert energy_max.compute(results)[0] == pytest.approx(-9.0)


def test_single_chain_b_residue_targeting():
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, _RI)
    # Only chain B residue 0 -> 4th row, value 7.
    energy = bg.energies.ResidueSAEnergy(
        oracle=oracle, feature_indices=[0], residues=[chain_b.residues[0]], pooling='mean'
    )
    assert energy.compute(results)[0] == pytest.approx(-7.0)


def test_inheritable_is_false():
    chain_a, _ = _multichain_chains()
    oracle, _ = _oracle_with_embeddings(_EMB, (chain_a,), _CI[:3], _RI[:3])
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], residues=[chain_a.residues[0]])
    assert energy.inheritable is False


def test_missing_embeddings_raises():
    chains = _multichain_chains()
    oracle = SAE.__new__(SAE)
    oracle.result_class = SAEResult
    result = SAEResult(input_chains=list(chains), features=np.ones(4, dtype=np.float64))  # no embeddings
    results = OraclesResultDict()
    results[oracle] = result
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0])
    with pytest.raises(ValueError, match='include_per_residue'):
        energy.compute(results)


def test_crosscheck_detects_index_disagreement():
    chain_a, chain_b = _multichain_chains()
    # Corrupt residue_index so it disagrees with input_chains reconstruction.
    bad_ri = [0, 1, 2, 1, 0]  # chain B order swapped
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, bad_ri)
    energy = bg.energies.ResidueSAEnergy(
        oracle=oracle, feature_indices=[0], residues=[chain_b.residues[0]]
    )
    with pytest.raises(ValueError, match='disagree'):
        energy.compute(results)


def test_crosscheck_skipped_when_indices_absent():
    # No chain_index/residue_index -> mapping falls back to input_chains only.
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b))
    energy = bg.energies.ResidueSAEnergy(
        oracle=oracle, feature_indices=[0], residues=[chain_b.residues[1]], pooling='mean'
    )
    assert energy.compute(results)[0] == pytest.approx(-9.0)


def test_weight_and_sign_conventions():
    chain_a, chain_b = _multichain_chains()
    oracle, results = _oracle_with_embeddings(_EMB, (chain_a, chain_b), _CI, _RI)
    energy = bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], weight=2.0, pooling='mean')
    unweighted, weighted = energy.compute(results)
    assert unweighted == pytest.approx(-5.0)
    assert weighted == pytest.approx(-10.0)
    # maximize=False flips the sign.
    energy_min = bg.energies.ResidueSAEnergy(
        oracle=oracle, feature_indices=[0], pooling='mean', maximize=False
    )
    assert energy_min.compute(results)[0] == pytest.approx(5.0)


def test_invalid_pooling_raises():
    chain_a, _ = _multichain_chains()
    oracle, _ = _oracle_with_embeddings(_EMB[:3], (chain_a,), _CI[:3], _RI[:3])
    with pytest.raises(ValueError, match='pooling'):
        bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], pooling='median')


def test_name_prefix():
    chain_a, _ = _multichain_chains()
    oracle, _ = _oracle_with_embeddings(_EMB[:3], (chain_a,), _CI[:3], _RI[:3])
    assert bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0]).name == 'residue_sae'
    assert (
        bg.energies.ResidueSAEnergy(oracle=oracle, feature_indices=[0], name='motif').name
        == 'residue_sae_motif'
    )
