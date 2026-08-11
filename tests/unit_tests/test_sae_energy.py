"""Unit tests for the SAEnergy energy term."""

import numpy as np
import pytest

import bagel as bg
from bagel.oracles.base import OraclesResultDict
from bagel.oracles.embedding.sae import SAE, SAEResult


def _oracle_with_features(features: np.ndarray):
    """Return a (fake) SAE oracle and an OraclesResultDict holding ``features``."""
    oracle = SAE.__new__(SAE)
    oracle.result_class = SAEResult
    chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=0)])]
    result = SAEResult(input_chains=chains, features=np.asarray(features, dtype=np.float64))
    oracles_result = OraclesResultDict()
    oracles_result[oracle] = result
    return oracle, oracles_result


def test_default_is_negated_and_l1_normalized():
    # Defaults: maximize=True (negate) and normalize_coefficients=True (sum|c|=1).
    # coeffs default to ones -> [1/3, 1/3, 1/3]; energy = -mean of selected features.
    features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    oracle, results = _oracle_with_features(features)
    energy = bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 2, 4])
    unweighted, weighted = energy.compute(results)
    assert unweighted == pytest.approx(-(1.0 + 3.0 + 5.0) / 3.0)
    assert weighted == pytest.approx(-(1.0 + 3.0 + 5.0) / 3.0)  # weight defaults to 1


def test_positive_coefficient_promotes_feature_by_default():
    # With the default sign, a larger activation gives a *lower* (more negative)
    # energy for a positive coefficient -- i.e. minimisation promotes the feature.
    oracle_low, results_low = _oracle_with_features(np.array([0.1, 0.0]))
    oracle_high, results_high = _oracle_with_features(np.array([5.0, 0.0]))
    e_low = bg.energies.SAEnergy(oracle=oracle_low, feature_indices=[0]).compute(results_low)[0]
    e_high = bg.energies.SAEnergy(oracle=oracle_high, feature_indices=[0]).compute(results_high)[0]
    assert e_high < e_low


def test_coefficients_are_l1_normalized():
    # Scaling all coefficients must not change the energy once normalized.
    features = np.array([1.0, 2.0, 3.0, 4.0])
    oracle, results = _oracle_with_features(features)
    e1 = bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 1], coefficients=[1.0, 1.0]).compute(results)[0]
    e2 = bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 1], coefficients=[5.0, 5.0]).compute(results)[0]
    assert e1 == pytest.approx(e2)
    # [1,1] -> [0.5, 0.5]; energy = -(0.5*1 + 0.5*2) = -1.5
    assert e1 == pytest.approx(-1.5)


def test_custom_coefficients_normalized_and_negated():
    features = np.array([1.0, 10.0, 3.0, 4.0])
    oracle, results = _oracle_with_features(features)
    energy = bg.energies.SAEnergy(oracle=oracle, feature_indices=[1, 3], coefficients=[2.0, -1.0])
    unweighted, _ = energy.compute(results)
    # sum|c| = 3 -> [2/3, -1/3]; features[1]=10, features[3]=4
    expected = -((2.0 / 3.0) * 10.0 + (-1.0 / 3.0) * 4.0)
    assert unweighted == pytest.approx(expected)


def test_maximize_false_minimizes_features():
    # maximize=False and normalization off -> raw linear combination (old behaviour).
    features = np.array([1.0, 10.0, 3.0, 4.0])
    oracle, results = _oracle_with_features(features)
    energy = bg.energies.SAEnergy(
        oracle=oracle,
        feature_indices=[1, 3],
        coefficients=[2.0, -1.0],
        maximize=False,
        normalize_coefficients=False,
    )
    unweighted, _ = energy.compute(results)
    assert unweighted == pytest.approx(2.0 * 10.0 + (-1.0) * 4.0)


def test_normalization_can_be_disabled():
    features = np.array([1.0, 1.0, 1.0])
    oracle, results = _oracle_with_features(features)
    energy = bg.energies.SAEnergy(
        oracle=oracle, feature_indices=[0, 1], coefficients=[2.0, 3.0], normalize_coefficients=False
    )
    # maximize default True -> negate; no normalization -> -(2*1 + 3*1) = -5
    assert energy.compute(results)[0] == pytest.approx(-5.0)


def test_zero_coefficients_with_normalization_raises():
    oracle, _ = _oracle_with_features(np.ones(3))
    with pytest.raises(ValueError, match='non-zero'):
        bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 1], coefficients=[0.0, 0.0])


def test_weight_scales_energy():
    features = np.array([1.0, 1.0, 1.0])
    oracle, results = _oracle_with_features(features)
    energy = bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 1], weight=3.0)
    unweighted, weighted = energy.compute(results)
    # default coeffs [1,1] -> [0.5,0.5]; energy = -(0.5+0.5) = -1.0
    assert unweighted == pytest.approx(-1.0)
    assert weighted == pytest.approx(-3.0)


def test_name_suffix():
    features = np.ones(3)
    oracle, _ = _oracle_with_features(features)
    assert bg.energies.SAEnergy(oracle=oracle, feature_indices=[0]).name == 'sae'
    assert bg.energies.SAEnergy(oracle=oracle, feature_indices=[0], name='motif').name == 'sae_motif'


def test_empty_feature_indices_raises():
    oracle, _ = _oracle_with_features(np.ones(3))
    with pytest.raises(ValueError, match='non-empty'):
        bg.energies.SAEnergy(oracle=oracle, feature_indices=[])


def test_duplicate_feature_indices_raises():
    oracle, _ = _oracle_with_features(np.ones(3))
    with pytest.raises(ValueError, match='unique'):
        bg.energies.SAEnergy(oracle=oracle, feature_indices=[1, 1])


def test_negative_feature_indices_raises():
    oracle, _ = _oracle_with_features(np.ones(3))
    with pytest.raises(ValueError, match='non-negative'):
        bg.energies.SAEnergy(oracle=oracle, feature_indices=[-1])


def test_coefficients_length_mismatch_raises():
    oracle, _ = _oracle_with_features(np.ones(3))
    with pytest.raises(ValueError, match='must match'):
        bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 1], coefficients=[1.0])


def test_feature_index_out_of_range_raises():
    features = np.array([1.0, 2.0, 3.0])
    oracle, results = _oracle_with_features(features)
    energy = bg.energies.SAEnergy(oracle=oracle, feature_indices=[0, 5])
    with pytest.raises(IndexError):
        energy.compute(results)
