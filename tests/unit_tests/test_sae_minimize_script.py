"""Smoke test for scripts/sae_features/minimize_sae.py.

Runs a short canonical Monte Carlo minimisation end-to-end with a fake,
sequence-dependent SAE oracle so no Modal backend, GPU, or network is needed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import bagel as bg
from bagel.oracles.embedding.sae import SAE, SAEResult

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'sae_features' / 'minimize_sae.py'

_NUM_FEATURES = 8


def _load_script():
    spec = importlib.util.spec_from_file_location('minimize_sae', _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SeqFeatureModel:
    """Fake boileroom SAE model: features depend on amino-acid composition."""

    def embed(self, sequences, options=None):  # noqa: ANN001
        seq = sequences[0].replace(':', '')
        pooled = np.zeros((1, _NUM_FEATURES), dtype=np.float64)
        for aa in seq:
            pooled[0, ord(aa) % _NUM_FEATURES] += 1.0

        class _Out:
            pooled_features = pooled
            features = None
            layer = 27
            sae_model = 'fake/sae'

        return _Out()


def _fake_oracle() -> SAE:
    oracle = SAE.__new__(SAE)
    oracle.backend = 'modal'
    oracle.device = None
    oracle.result_class = SAEResult
    oracle.model = _SeqFeatureModel()
    return oracle


def test_script_exposes_expected_helpers():
    module = _load_script()
    for attr in ('build_sae_oracle', 'build_system', 'build_minimizer', 'run', 'main'):
        assert hasattr(module, attr)


def test_build_system_has_single_sae_energy_term():
    module = _load_script()
    system = module.build_system(oracle=_fake_oracle(), sequence='ACDEFGHIK', feature_indices=[0, 1, 2])
    state = system.states[0]
    assert len(state.energy_terms) == 1
    assert state.energy_terms[0].name == 'sae'


def test_short_monte_carlo_runs_end_to_end(tmp_path, monkeypatch):
    # Run in a temp dir so minimizer output does not litter the repo.
    monkeypatch.chdir(tmp_path)
    module = _load_script()
    oracle = _fake_oracle()
    system = module.build_system(
        oracle=oracle,
        sequence='ACDEFGHIKLMN',
        feature_indices=[0, 1, 2, 3],
    )
    minimizer = module.build_minimizer(temperature=0.01, n_steps=5, log_interval=100)
    final_system = minimizer.minimize_system(system=system)
    assert final_system is not None
    # The final system should carry a finite total energy.
    assert np.isfinite(final_system.get_total_energy())
