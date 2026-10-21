"""
MosaicScreening against the real services on Modal. Runs only with ``--oracles modal`` and needs the mosaic app deployed
(``modal deploy src/bagel/mosaic_screening/modal_service.py``); the tests are skipped if it cannot be reached.
"""

import pathlib as pl

import numpy as np
import pytest

import bagel as bg
from bagel.mosaic_screening import ModalMosaicBackend, MosaicBackendError, MosaicScreening
from bagel.mosaic_screening.plan import build_plans


@pytest.fixture(scope='module')
def mosaic_backend() -> ModalMosaicBackend:
    backend = ModalMosaicBackend()
    try:
        backend.versions()
    except MosaicBackendError as exc:
        pytest.skip(f'mosaic service on Modal not available: {exc}')
    return backend


def _system(oracle: bg.oracles.folding.ESMFold2) -> bg.System:
    binder = bg.Chain([bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate('ACDEFGHIKLMNPQRSTVWY')])
    target = bg.Chain([bg.Residue(a, 'T', i, mutable=False) for i, a in enumerate('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQ')])
    terms = [
        bg.energies.PTMEnergy(oracle=oracle),
        bg.energies.OverallPLDDTEnergy(oracle=oracle),
        bg.energies.PAEEnergy(oracle=oracle, residues=[binder.residues, target.residues], weight=2.0),
    ]
    return bg.System(states=[bg.State(name='binder', chains=[binder, target], energy_terms=terms)])


def test_mosaic_pseudo_energy_runs_in_a_minimizer_with_a_real_oracle(
    esmfold2: bg.oracles.folding.ESMFold2, mosaic_backend: ModalMosaicBackend, test_output_path: pl.Path
) -> None:
    np.random.seed(0)
    protocol = MosaicScreening(
        backend=mosaic_backend, n_candidates=4, calibration_steps=1, calibration_mutants=4, calibration_threshold=-1.0
    )
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=protocol,
        temperature=0.1,
        n_steps=3,
        log_path=test_output_path,
        experiment_name='mosaic_screening',
        callbacks=[],
    )
    minimizer.minimize_system(_system(esmfold2))
    assert protocol.stats['screened_steps'] == 3 and protocol.stats['mosaic_calls'] >= 1
    assert (minimizer.log_path / 'Mosaic_pseudo_energy.txt').exists()
    assert protocol.calibration['n'] == 4


def test_mosaic_and_bagel_agree_on_the_sign_of_a_clearly_good_vs_bad_sequence(
    esmfold2: bg.oracles.folding.ESMFold2, mosaic_backend: ModalMosaicBackend
) -> None:
    """Weak sanity check, not a calibration: a poly-glycine binder must score worse than a folded-looking one in both."""
    from bagel.mosaic_screening.adapter import pseudo_energy

    system = _system(esmfold2)
    plan = build_plans(system)[0]
    assert plan is not None
    group = plan.groups[0]
    sequences = ['ACDEFGHIKLMNPQRSTVWY', 'G' * 20]
    predictions = mosaic_backend.predict(group.spec, plan.target_sequences, sequences + [sequences[-1]] * 7)[:2]
    pseudo, real = [], []
    for sequence, prediction in zip(sequences, predictions):
        trial = system.__copy__()
        trial.states[0].chains[0].residues[:] = [bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate(sequence)]
        value, _ = pseudo_energy(
            plan.terms_of(trial.states[0], group), plan.ordered_chains(trial.states[0]), prediction
        )
        pseudo.append(value)
        real.append(trial.get_total_energy())
    assert np.sign(pseudo[1] - pseudo[0]) == np.sign(real[1] - real[0]), (pseudo, real)
