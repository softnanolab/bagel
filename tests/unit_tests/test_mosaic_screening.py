"""
Tests of the mosaic screening protocol. They need neither jax, mosaic, Modal nor a GPU: a fake backend stands in for
mosaic and a fake oracle for bagel's folding oracle.
"""

import math
from pathlib import Path

import numpy as np
import pytest

import bagel as bg
from bagel.mosaic_screening import (
    MosaicBackendError,
    MosaicIncompatibleError,
    MosaicModelSpec,
    MosaicScreening,
    SubprocessMosaicBackend,
    build_plans,
    default_spec_for,
)
from bagel.mosaic_screening.adapter import MosaicPrediction, build_result, pseudo_energy
from bagel.mosaic_screening.plan import coverage, format_report
from bagel.mosaic_screening.protocol import correlation_upper95, spearman

HYDROPHOBIC = set('AILMFVWY')


def fake_prediction(design_sequence: str, n_total: int, sign: float = 1.0) -> MosaicPrediction:
    """Confidences that depend on the designed sequence (more hydrophobic = more confident)."""
    h = np.array([c in HYDROPHOBIC for c in design_sequence], dtype=float)
    plddt = np.full(n_total, 0.5)
    plddt[: len(design_sequence)] += (
        sign * 0.4 * h / max(1.0, h.sum() + 1.0) * 3 + 0.01 * np.arange(len(design_sequence)) % 0.05
    )
    plddt = np.clip(plddt, 0.0, 1.0)
    pae = np.full((n_total, n_total), 20.0 - sign * 10.0 * h.mean())
    return MosaicPrediction(
        plddt=plddt, pae=np.clip(pae, 1.0, 30.0), ptm=float(np.clip(0.4 + sign * 0.3 * h.mean(), 0, 1))
    )


class FakeBackend:
    """Mosaic stand-in: deterministic function of the designed sequence; records the calls."""

    def __init__(self, sign: float = 1.0) -> None:
        self.sign = sign
        self.calls: list[tuple[MosaicModelSpec, list[str], list[str]]] = []
        self.structure_flags: list[bool] = []

    def predict(self, spec, target_sequences, design_sequences, structure=False):  # noqa: ANN001
        self.calls.append((spec, list(target_sequences), list(design_sequences)))
        self.structure_flags.append(structure)
        n_total = len(design_sequences[0]) + sum(len(s) for s in target_sequences)
        return [fake_prediction(s, n_total, self.sign) for s in design_sequences]

    def versions(self):  # noqa: ANN001
        return {'mosaic_commit': 'fake'}


def make_realistic(oracle: bg.oracles.folding.ESMFold2) -> None:
    """Make the fake oracle answer like the fake mosaic (same function of the sequence), so real and pseudo agree."""

    def predict(chains):  # noqa: ANN001
        design = next(c for c in chains if any(r.mutable for r in c.residues))
        ordered = [design] + [c for c in chains if c is not design]
        n = sum(c.length for c in ordered)
        return build_result(oracle, ordered, fake_prediction(design.sequence, n))

    oracle.predict = predict  # type: ignore[method-assign]


def _system(oracle, binder='ACDEFGHIKL', target='MKVLAWQRST', target_first=False, unsupported=True):
    binder_chain = bg.Chain([bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate(binder)])
    target_chain = bg.Chain([bg.Residue(a, 'T', i, mutable=False) for i, a in enumerate(target)])
    hotspot = [target_chain.residues[i] for i in (2, 3, 4)]
    terms = [
        bg.energies.PTMEnergy(oracle=oracle, weight=1.0),
        bg.energies.PLDDTEnergy(oracle=oracle, residues=binder_chain.residues, weight=2.0, name='binder'),
        bg.energies.OverallPLDDTEnergy(oracle=oracle, weight=0.5),
        bg.energies.PAEEnergy(oracle=oracle, residues=[hotspot, binder_chain.residues], weight=3.0),
        bg.energies.ipSAEEnergy(oracle=oracle, residues=[binder_chain.residues, hotspot], weight=4.0, direction='max'),
        bg.energies.SaltBridgeEnergy(
            oracle=oracle, residues=[binder_chain.residues, target_chain.residues], weight=5.0
        ),  # not supported
    ]
    if not unsupported:
        terms = terms[:-1]
    chains = [target_chain, binder_chain] if target_first else [binder_chain, target_chain]
    return bg.System(states=[bg.State(name='s', chains=chains, energy_terms=terms)])


# ---------------------------------------------------------------------------------------------------------------------
# plan: terms, order, coverage, guardrails
# ---------------------------------------------------------------------------------------------------------------------
def test_plan_lists_used_and_unused_terms_with_coverage(fake_esmfold2) -> None:
    plan = build_plans(_system(fake_esmfold2))[0]
    assert plan is not None
    used = [s.name for s in plan.statuses if s.used]
    assert used == ['pTM', 'local_pLDDT_binder', 'global_pLDDT', 'cross_PAE', 'ipSAE']
    unused = [s for s in plan.statuses if not s.used]
    assert len(unused) == 1 and 'SaltBridgeEnergy' in unused[0].reason
    cov = coverage(plan.statuses)
    assert (cov.n_used, cov.n_total) == (5, 6)
    assert cov.weight_used == pytest.approx(10.5) and cov.weight_total == pytest.approx(15.5)
    text = format_report([plan])
    assert 'USED' in text and 'UNUSED' in text and '5/6' in text and 'SaltBridge' in text


def test_plan_puts_designed_chain_first_even_if_listed_second(fake_esmfold2) -> None:
    plan = build_plans(_system(fake_esmfold2, target_first=True))[0]
    assert plan.chain_order == ['B', 'T'] and plan.target_sequences == ['MKVLAWQRST']
    assert plan.mutable_positions == list(range(10))


def test_plan_skips_states_without_designed_chain(fake_esmfold2) -> None:
    system = _system(fake_esmfold2)
    fixed = bg.Chain([bg.Residue(a, 'F', i, mutable=False) for i, a in enumerate('MKVLA')])
    other = bg.State(name='fixed', chains=[fixed], energy_terms=[bg.energies.PTMEnergy(oracle=fake_esmfold2)])
    plans = build_plans(bg.System(states=[system.states[0], other]))
    assert plans[0] is not None and plans[1] is None


def test_plan_guardrails(fake_esmfold2) -> None:
    system = _system(fake_esmfold2)
    system.states[0].chains[1].residues[0].mutable = True  # a second designed chain
    with pytest.raises(MosaicIncompatibleError, match='one designed chain per State'):
        build_plans(system)

    system = _system(fake_esmfold2)
    system.states[0].energy_terms = [
        bg.energies.SaltBridgeEnergy(
            oracle=fake_esmfold2, residues=[system.states[0].chains[0].residues, system.states[0].chains[1].residues]
        )
    ]
    with pytest.raises(MosaicIncompatibleError, match='nothing to screen'):
        build_plans(system)

    system = _system(fake_esmfold2)
    system.states[0].chains[0].residues[3].index = 7  # residues were added/removed: masks would be wrong
    with pytest.raises(MosaicIncompatibleError, match='masks'):
        build_plans(system)

    with pytest.raises(MosaicIncompatibleError, match='No State'):
        fixed = bg.Chain([bg.Residue(a, 'F', i, mutable=False) for i, a in enumerate('MKVLA')])
        build_plans(
            bg.System(
                states=[bg.State(name='x', chains=[fixed], energy_terms=[bg.energies.PTMEnergy(oracle=fake_esmfold2)])]
            )
        )


def test_oracle_without_mosaic_counterpart_is_unused(fake_esmfold2, fake_esmfold) -> None:
    system = _system(fake_esmfold2)
    chain = system.states[0].chains[0]
    system.states[0].energy_terms.append(
        bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=chain.residues, name='esm1')
    )
    plan = build_plans(system)[0]
    status = next(s for s in plan.statuses if s.name.endswith('esm1'))
    assert not status.used and 'no mosaic counterpart' in status.reason
    assert default_spec_for(fake_esmfold) is None
    assert default_spec_for(fake_esmfold2) == MosaicModelSpec.esmfold2()


def test_models_override_and_validation(fake_esmfold2) -> None:
    custom = MosaicModelSpec.esmfold2(recycling_steps=1, sampling_steps=5, checkpoint='fast')
    plan = build_plans(_system(fake_esmfold2), models={fake_esmfold2: custom})[0]
    assert plan.groups[0].spec == custom and len(plan.groups) == 1
    with pytest.raises(ValueError):
        MosaicModelSpec('alphafold9', 1, 1)
    assert MosaicModelSpec.boltz2().sampling_steps == 25 and MosaicModelSpec.boltz2().recycling_steps == 1


# ---------------------------------------------------------------------------------------------------------------------
# adapter: bagel's own term code on a mosaic prediction
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('target_first', [False, True])
def test_adapter_applies_residue_groups_to_the_right_residues(fake_esmfold2, target_first) -> None:
    """The easiest place for a silent bug: masks must hit the same residues whatever the State's chain order."""
    system = _system(fake_esmfold2, target_first=target_first)
    state = system.states[0]
    plan = build_plans(system)[0]
    chains = plan.ordered_chains(state)  # mosaic order: B then T
    n = 20
    plddt = np.linspace(0.1, 0.9, n)  # distinct value per residue, in mosaic order
    prediction = MosaicPrediction(plddt=plddt, pae=np.full((n, n), 12.0), ptm=0.6)
    terms = {t.name: t for t in state.energy_terms}
    value, parts = pseudo_energy([terms['local_pLDDT_binder']], chains, prediction)
    assert parts['local_pLDDT_binder'] == pytest.approx(-plddt[:10].mean()), 'binder residues are the first 10 tokens'
    assert value == pytest.approx(2.0 * -plddt[:10].mean())
    # a residue group inside the (fixed) target chain must map to tokens 10..19
    target_residues = [state.chains[0 if target_first else 1].residues[i] for i in (2, 3)]
    term = bg.energies.PLDDTEnergy(oracle=fake_esmfold2, residues=target_residues, name='t')
    _, parts = pseudo_energy([term], chains, prediction)
    assert parts[term.name] == pytest.approx(-plddt[[12, 13]].mean())


def test_adapter_weights_pae_and_ipsae_like_bagel(fake_esmfold2) -> None:
    system = _system(fake_esmfold2)
    state = system.states[0]
    plan = build_plans(system)[0]
    chains = plan.ordered_chains(state)
    prediction = fake_prediction('ACDEFGHIKL', 20)
    group = plan.groups[0]
    terms = plan.terms_of(state, group)
    total, parts = pseudo_energy(terms, chains, prediction)
    assert total == pytest.approx(sum(t.weight * parts[t.name] for t in terms))
    assert parts['cross_PAE'] == pytest.approx(prediction.pae[0, 12] / 30.0)  # constant PAE matrix: mean / max_pae
    assert parts['pTM'] == pytest.approx(-prediction.ptm)


def test_adapter_rejects_inconsistent_shapes(fake_esmfold2) -> None:
    system = _system(fake_esmfold2)
    chains = build_plans(system)[0].ordered_chains(system.states[0])
    with pytest.raises(ValueError, match='Mosaic returned'):
        build_result(fake_esmfold2, chains, fake_prediction('ACDEFGHIKL', 19))


# ---------------------------------------------------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------------------------------------------------
def test_selection_is_the_lowest_pseudo_energy_candidate(fake_esmfold2) -> None:
    np.random.seed(0)
    backend = FakeBackend()
    system = _system(fake_esmfold2)
    protocol = MosaicScreening(backend=backend, calibration_steps=0)
    mutated, record = protocol.one_step(system)
    ((spec, targets, batch),) = backend.calls
    assert targets == ['MKVLAWQRST'] and batch[0] == 'ACDEFGHIKL'
    assert len(batch) == protocol.n_candidates + 1, 'constant batch size'
    # each candidate differs from the current sequence in exactly one position
    assert all(sum(a != b for a, b in zip(batch[0], s)) == 1 for s in batch[1:])
    deltas = protocol.last_info['candidate_deltas']
    mutation = record.mutations[0]
    assert (
        mutated.states[0].chains[0].sequence
        == batch[0][: mutation.residue_index] + mutation.new_amino_acid + batch[0][mutation.residue_index + 1 :]
    )
    assert protocol.last_info['predicted_delta'] == pytest.approx(min(deltas))
    assert system.states[0].chains[0].sequence == 'ACDEFGHIKL', 'the input system must not be modified'


def test_selection_matches_manual_ranking_and_temperature(fake_esmfold2) -> None:
    system = _system(fake_esmfold2)
    # T=0 always proposes the argmin; a large T proposes other candidates too
    picked_best, picked_other = 0, 0
    for seed in range(40):
        np.random.seed(seed)
        p0 = MosaicScreening(backend=FakeBackend(), calibration_steps=0)
        p0.one_step(system)
        assert p0.last_info['predicted_delta'] == pytest.approx(min(p0.last_info['candidate_deltas']))
        np.random.seed(seed)
        p1 = MosaicScreening(backend=FakeBackend(), calibration_steps=0, selection_temperature=1e6)
        p1.one_step(system)
        if p1.last_info['predicted_delta'] == pytest.approx(min(p1.last_info['candidate_deltas'])):
            picked_best += 1
        else:
            picked_other += 1
    assert picked_other > picked_best, 'a very high temperature should behave like a random choice among 8 candidates'


def test_scores_are_cached_and_rejected_moves_are_not_repeated(fake_esmfold2) -> None:
    np.random.seed(1)
    backend = FakeBackend()
    system = _system(fake_esmfold2)
    protocol = MosaicScreening(backend=backend, calibration_steps=0, n_candidates=4)
    proposed = []
    for _ in range(6):  # the same sequence every time: all proposals were "rejected"
        _, record = protocol.one_step(system)
        proposed.append((record.mutations[0].residue_index, record.mutations[0].new_amino_acid))
    assert len(set(proposed)) == 6, 'a move already proposed from this sequence is not proposed again'
    batches = [batch for _, _, batch in backend.calls]
    assert sum('ACDEFGHIKL' in b for b in batches) == 1, (
        'the current sequence is scored once, then served from the cache'
    )
    assert all(len(b) == protocol.n_candidates + 1 for b in batches), 'constant batch size (padding repeats a sequence)'
    unique = [set(b) for b in batches]
    assert sum(len(u) for u in unique) == len(set().union(*unique)), 'no sequence is scored twice'


def test_screening_steps_limits_the_calls_to_mosaic(fake_esmfold2) -> None:
    np.random.seed(2)
    backend = FakeBackend()
    system = _system(fake_esmfold2)
    protocol = MosaicScreening(backend=backend, calibration_steps=0, screening_steps=2)
    infos = []
    for _ in range(5):
        protocol.one_step(system)
        infos.append(protocol.last_info['screened'])
    assert infos == [True, True, False, False, False]
    assert len(backend.calls) == 2 and protocol.stats['fallback_steps'] == 3


def test_multi_state_sums_the_pseudo_energy_over_states_containing_the_chain(fake_esmfold2) -> None:
    np.random.seed(3)
    binder = bg.Chain([bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate('ACDEFGHIKL')])
    t1 = bg.Chain([bg.Residue(a, 'T', i, mutable=False) for i, a in enumerate('MKVLAWQRST')])
    t2 = bg.Chain([bg.Residue(a, 'U', i, mutable=False) for i, a in enumerate('MKVLAWQ')])
    f = bg.Chain([bg.Residue(a, 'F', i, mutable=False) for i, a in enumerate('MKVLA')])
    s1 = bg.State(name='s1', chains=[binder, t1], energy_terms=[bg.energies.OverallPLDDTEnergy(oracle=fake_esmfold2)])
    s2 = bg.State(
        name='s2', chains=[binder, t2], energy_terms=[bg.energies.OverallPLDDTEnergy(oracle=fake_esmfold2, weight=2.0)]
    )
    s3 = bg.State(name='s3', chains=[f], energy_terms=[bg.energies.PTMEnergy(oracle=fake_esmfold2)])
    system = bg.System(states=[s1, s2, s3])
    backend = FakeBackend()
    protocol = MosaicScreening(backend=backend, calibration_steps=0)
    protocol.one_step(system)
    assert [c[1] for c in backend.calls] == [['MKVLAWQRST'], ['MKVLAWQ']], (
        'one call per State that contains the chain; none for s3'
    )
    # the delta is the weighted sum over the two States
    seqs = backend.calls[0][2]
    d = []
    for (_, targets, batch), weight in zip(backend.calls, (1.0, 2.0)):
        n = 10 + len(targets[0])
        vals = [-weight * fake_prediction(s, n).plddt.mean() for s in batch]
        d.append(np.array(vals[1:]) - vals[0])
    assert np.allclose(protocol.last_info['candidate_deltas'], d[0] + d[1])


def test_grand_canonical_and_length_changes_are_rejected_before_mosaic(fake_esmfold2) -> None:
    with pytest.raises(MosaicIncompatibleError, match='GrandCanonical'):
        MosaicScreening(backend=FakeBackend(), fallback=bg.mutation.GrandCanonical())
    backend = FakeBackend()
    system = _system(fake_esmfold2)
    protocol = MosaicScreening(backend=backend, calibration_steps=0)
    protocol.one_step(system)
    n_calls = len(backend.calls)
    grown = system.__copy__()
    chain = grown.states[0].chains[0]
    chain.residues.append(bg.Residue('A', 'B', chain.length, mutable=True))
    with pytest.raises(MosaicIncompatibleError, match='length'):
        protocol.one_step(grown)
    assert len(backend.calls) == n_calls, 'no call to mosaic after the gate failed'


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        MosaicScreening(backend=FakeBackend(), n_candidates=0)
    with pytest.raises(ValueError):
        MosaicScreening(backend=FakeBackend(), calibration_rule='median')
    with pytest.raises(ValueError):
        MosaicScreening(backend=FakeBackend(), n_candidates=4, calibration_mutants=8)
    with pytest.raises(ValueError):
        MosaicScreening(backend=FakeBackend(), selection_temperature=-1.0)
    MosaicScreening(
        backend=FakeBackend(), calibration_steps=0, n_candidates=2, calibration_mutants=8
    )  # off: not checked


# ---------------------------------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------------------------------
def test_spearman_and_upper_bound() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    # 40 pairs: standard error 1/sqrt(37) = 0.164, so a point estimate of 0.10 has an upper end near 0.40
    assert correlation_upper95(0.10, 40) == pytest.approx(0.40, abs=0.01)
    assert math.isnan(correlation_upper95(0.5, 3))


def test_calibration_passes_when_pseudo_and_real_energy_agree(fake_esmfold2, tmp_path) -> None:
    np.random.seed(4)
    make_realistic(fake_esmfold2)
    system = _system(fake_esmfold2, unsupported=False)
    protocol = MosaicScreening(backend=FakeBackend(), calibration_steps=3, report_dir=tmp_path)
    for _ in range(4):
        protocol.one_step(system)
    assert protocol.calibration['passed'] and protocol.calibration['rho'] > 0.9
    assert protocol.calibration['n'] == 24
    assert protocol._screening_on
    assert protocol.stats['calibration_folds'] == 24


def test_calibration_failure_switches_to_random_sampling_and_is_reported(fake_esmfold2, tmp_path) -> None:
    np.random.seed(5)
    make_realistic(fake_esmfold2)
    system = _system(fake_esmfold2, unsupported=False)
    backend = FakeBackend(sign=-1.0)  # mosaic ranks the opposite way to the real energy
    protocol = MosaicScreening(backend=backend, calibration_steps=3, report_dir=tmp_path)
    for _ in range(3):
        protocol.one_step(system)
    assert protocol.calibration['passed'] is False and protocol.calibration['rho'] < 0
    calls = len(backend.calls)
    protocol.one_step(system)
    assert protocol.last_info['screened'] is False and len(backend.calls) == calls, 'random sampling, mosaic not called'
    text = (tmp_path / 'Mosaic_pseudo_energy.txt').read_text()
    assert 'passed' in text and 'False' in text and 'random sampling' in text


def _weakly_correlated_pairs(n: int = 40) -> list[tuple[float, float]]:
    """Deterministic pairs with a Spearman correlation between 0.08 and 0.2 (a low but positive estimate)."""
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        y = 0.15 * x + rng.normal(size=n)
        if 0.08 < spearman(list(x), list(y)) < 0.2:
            return list(zip(x, y))
    raise AssertionError('no seed found')


def test_calibration_rules(fake_esmfold2, tmp_path) -> None:
    """With 40 pairs, rho = ~0.14 is below 0.25 ('point' fails) but its 95% upper end (~0.45) is not ('upper95' passes)."""
    system = _system(fake_esmfold2)
    for rule, expected in (('point', False), ('upper95', True)):
        protocol = MosaicScreening(backend=FakeBackend(), calibration_rule=rule, report_dir=tmp_path)
        protocol.one_step(system)  # builds the plans
        protocol._pairs = _weakly_correlated_pairs()
        protocol._finish_calibration()
        assert protocol.calibration['passed'] is expected, (rule, protocol.calibration)
        assert protocol._screening_on is expected


def test_calibration_off(fake_esmfold2) -> None:
    np.random.seed(6)
    protocol = MosaicScreening(backend=FakeBackend(), calibration_steps=0)
    system = _system(fake_esmfold2)
    for _ in range(3):
        protocol.one_step(system)
    assert protocol.calibration == {} and protocol.stats['calibration_folds'] == 0


# ---------------------------------------------------------------------------------------------------------------------
# report and minimizer integration
# ---------------------------------------------------------------------------------------------------------------------
def test_report_is_written_to_the_log_folder_of_the_minimizer(fake_esmfold2, tmp_path) -> None:
    np.random.seed(7)
    make_realistic(fake_esmfold2)
    system = _system(fake_esmfold2, unsupported=False)
    protocol = MosaicScreening(backend=FakeBackend(), calibration_steps=1)
    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=protocol,
        initial_temperature=0.1,
        final_temperature=0.01,
        n_steps=3,
        log_path=tmp_path,
        experiment_name='run',
        callbacks=[],
    )
    minimizer.minimize_system(system)
    report = tmp_path / 'run' / 'Mosaic_pseudo_energy.txt'
    assert report.exists()
    text = report.read_text()
    assert 'Coverage by number of terms : 5/5' in text and 'Coverage by weight' in text
    assert protocol.stats['screened_steps'] == 3


# ---------------------------------------------------------------------------------------------------------------------
# transport: the real subprocess + JSON-lines wire format, with the fake worker (no GPU)
# ---------------------------------------------------------------------------------------------------------------------
def test_subprocess_backend_round_trip(tmp_path) -> None:
    import sys

    from bagel.mosaic_screening.backends import WORKER_SCRIPT

    backend = SubprocessMosaicBackend([sys.executable, str(WORKER_SCRIPT), '--fake'], timeout=60)
    try:
        spec = MosaicModelSpec.esmfold2()
        predictions = backend.predict(spec, ['MKVLAWQRST'], ['ACDEFGHIKL', 'AAAAAAAAAA'])
        assert len(predictions) == 2 and predictions[0].plddt.shape == (20,) and predictions[0].pae.shape == (20, 20)
        assert predictions[1].ptm > predictions[0].ptm  # more hydrophobic
        assert backend.versions()['mosaic_commit'] == 'fake'
    finally:
        backend.close()


def test_subprocess_backend_reports_a_dead_worker() -> None:
    import sys

    backend = SubprocessMosaicBackend([sys.executable, '-c', 'import sys; sys.exit(3)'], timeout=30)
    with pytest.raises(MosaicBackendError):
        backend.predict(MosaicModelSpec.esmfold2(), ['MKV'], ['AAA'])


def test_end_to_end_with_subprocess_backend(fake_esmfold2) -> None:
    import sys

    from bagel.mosaic_screening.backends import WORKER_SCRIPT

    np.random.seed(8)
    backend = SubprocessMosaicBackend([sys.executable, str(WORKER_SCRIPT), '--fake'], timeout=60)
    try:
        protocol = MosaicScreening(backend=backend, calibration_steps=0)
        mutated, record = protocol.one_step(_system(fake_esmfold2))
        assert len(record.mutations) == 1 and protocol.stats['mosaic_calls'] == 1
    finally:
        backend.close()


# ---------------------------------------------------------------------------------------------------------------------
# all-atom structures rebuilt from mosaic's atom37 output
# ---------------------------------------------------------------------------------------------------------------------
def _atom37_from_structure(chains, structure):  # noqa: ANN001
    """Inverse of structure_from_atom37, for a structure whose atoms are heavy atoms of standard residues."""
    from bagel.mosaic_screening.adapter import ATOM37_NAMES

    residues = [r for c in chains for r in c.residues]
    index = {(r.chain_ID, r.index): i for i, r in enumerate(residues)}
    coords = np.zeros((len(residues), 37, 3))
    mask = np.zeros((len(residues), 37))
    for atom in structure:
        if atom.atom_name in ATOM37_NAMES and atom.atom_name != 'OXT':
            i = index[(str(atom.chain_id), int(atom.res_id))]
            coords[i, ATOM37_NAMES.index(atom.atom_name)] = atom.coord
            mask[i, ATOM37_NAMES.index(atom.atom_name)] = 1.0
    return coords, mask


@pytest.fixture
def two_chain_structure():  # noqa: ANN201
    """Two chains of a real protein (chain A designed, B fixed), renumbered from 0, heavy atoms without OXT."""
    from biotite.structure.io import load_structure

    path = Path(__file__).resolve().parents[1] / 'structures' / 'example_protein.pdb'
    atoms = load_structure(str(path))
    atoms = atoms[(atoms.element != 'H') & (atoms.atom_name != 'OXT') & ~atoms.hetero]
    chains, pieces = [], []
    for chain_id in ('A', 'B'):
        sub = atoms[atoms.chain_id == chain_id].copy()
        ids = sub.res_id
        _, first = np.unique(ids, return_index=True)
        order = ids[np.sort(first)]
        renumber = {old: new for new, old in enumerate(order)}
        sub.res_id = np.array([renumber[i] for i in ids])
        names = [sub.res_name[sub.res_id == k][0] for k in range(len(order))]
        from bagel.constants import aa_dict

        three_to_one = {v: k for k, v in aa_dict.items()}
        chains.append(
            bg.Chain([bg.Residue(three_to_one[n], chain_id, k, mutable=chain_id == 'A') for k, n in enumerate(names)])
        )
        pieces.append(sub)
    structure = pieces[0] + pieces[1]
    return chains, structure


def test_rebuilt_structure_matches_the_original_atom_for_atom(two_chain_structure) -> None:
    from bagel.mosaic_screening.adapter import structure_from_atom37

    chains, structure = two_chain_structure
    coords, mask = _atom37_from_structure(chains, structure)
    rebuilt = structure_from_atom37(chains, coords, mask)
    assert len(rebuilt) == len(structure)
    for field in ('chain_id', 'res_id', 'res_name', 'atom_name', 'element'):
        assert list(getattr(rebuilt, field)) == list(getattr(structure, field)), field
    assert np.allclose(rebuilt.coord, structure.coord, atol=1e-4)


def test_structure_terms_give_the_same_value_on_the_rebuilt_structure(two_chain_structure, fake_esmfold2) -> None:
    from bagel.oracles import OraclesResultDict

    chains, structure = two_chain_structure
    n = sum(c.length for c in chains)
    coords, mask = _atom37_from_structure(chains, structure)
    prediction = MosaicPrediction(
        plddt=np.full(n, 0.8),
        pae=np.full((n, n), 10.0),
        ptm=0.7,
        atom37_coords=coords.astype(np.float32).astype(np.float64),
        atom37_mask=mask,
    )
    original = fake_esmfold2.result_class(
        input_chains=chains,
        structure=structure,
        local_plddt=np.full((1, n), 0.8),
        ptm=np.full((1, 1), 0.7),
        pae=np.full((1, n, n), 10.0),
    )
    rebuilt = build_result(fake_esmfold2, chains, prediction)
    E = bg.energies
    A, B = chains[0].residues, chains[1].residues
    terms = [
        E.SurfaceAreaEnergy(oracle=fake_esmfold2),
        E.SurfaceAreaEnergy(oracle=fake_esmfold2, residues=A, name='A'),
        E.HydrophobicEnergy(oracle=fake_esmfold2),
        E.HydrophobicEnergy(oracle=fake_esmfold2, residues=A, surface_only=True, name='surf'),
        E.HydropathyEnergy(oracle=fake_esmfold2),
        E.GlobularEnergy(oracle=fake_esmfold2),
        E.BuriedHistidineEnergy(oracle=fake_esmfold2),
        E.HistidineCarboxylateContactEnergy(oracle=fake_esmfold2),
    ]
    from bagel.mosaic_screening.plan import STRUCTURE_TERMS

    assert all(type(t) in STRUCTURE_TERMS for t in terms)
    for term in terms:
        a = OraclesResultDict()
        a[fake_esmfold2] = original
        b = OraclesResultDict()
        b[fake_esmfold2] = rebuilt
        va, vb = term.compute(a)[0], term.compute(b)[0]
        assert np.allclose(va, vb, rtol=1e-4, atol=1e-6), (term.name, va, vb)


def test_structure_is_requested_only_when_a_term_needs_it(fake_esmfold2) -> None:
    system = _system(fake_esmfold2, unsupported=False)
    assert not build_plans(system)[0].groups[0].needs_structure
    system.states[0].energy_terms.append(bg.energies.HydrophobicEnergy(oracle=fake_esmfold2))
    off = build_plans(system)[0]  # structure terms are opt-in
    assert not off.groups[0].needs_structure
    assert any(not s.used and 'use_structure_terms' in s.reason for s in off.statuses)
    plan = build_plans(system, structure_terms=True)[0]
    assert plan.groups[0].needs_structure
    backend = FakeBackend()
    np.random.seed(0)
    MosaicScreening(backend=backend, calibration_steps=0, use_structure_terms=True).one_step(system)
    assert backend.structure_flags == [True]
    backend = FakeBackend()
    MosaicScreening(backend=backend, calibration_steps=0).one_step(system)
    assert backend.structure_flags == [False]


def test_subprocess_backend_returns_structures_on_request() -> None:
    import sys

    from bagel.mosaic_screening.backends import WORKER_SCRIPT

    backend = SubprocessMosaicBackend([sys.executable, str(WORKER_SCRIPT), '--fake'], timeout=60)
    try:
        spec = MosaicModelSpec.esmfold2()
        plain = backend.predict(spec, ['MKV'], ['ACDE'])[0]
        full = backend.predict(spec, ['MKV'], ['ACDE'], structure=True)[0]
        assert plain.atom37_coords is None
        assert full.atom37_coords.shape == (7, 37, 3) and full.atom37_mask.shape == (7, 37)
    finally:
        backend.close()
