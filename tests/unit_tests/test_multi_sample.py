"""Tests for n_samples (several structures per prediction) and energy terms with several oracles."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import bagel as bg
from boileroom.models.alphafold.types import AlphaFold2MultimerOutput
from boileroom.models.rf3.types import RF3Output

from test_energies import _FakeBackboneOracle, _binder_and_target
from test_folding_oracle_wrappers import FakeModel, fake_atom_array, prediction_metadata
from test_folding_oracles_all_models import five_sample_output

ConfidenceResult = bg.oracles.folding.base.ConfidenceFoldingResult


def make_result(chains: list[bg.Chain], plddt: float, shift: float = 0.0) -> ConfidenceResult:
    n = sum(len(chain.residues) for chain in chains)
    structure = fake_atom_array(chains)
    structure.coord[:, 0] += shift
    return ConfidenceResult(
        input_chains=chains,
        structure=structure,
        local_plddt=np.full((1, n), plddt),
        ptm=np.full((1, 1), plddt),
        pae=np.zeros((1, n, n)),
    )


class SampledOracle(bg.oracles.folding.FoldingOracle):
    """Folding oracle whose k-th sample has pLDDT ``plddts[k]`` (best first) and is shifted by ``shift`` in x."""

    result_class = ConfidenceResult

    def __init__(self, plddts: list[float], shift: float = 0.0) -> None:
        self.plddts = plddts
        self.shift = shift
        self.n_samples = len(plddts)
        self.calls = 0

    def fold(self, chains: list[bg.Chain]) -> ConfidenceResult:
        self.calls += 1
        return make_result(chains, self.plddts[0], self.shift)

    def predict_samples(self, chains: list[bg.Chain]) -> list[bg.oracles.folding.FoldingResult]:
        self.calls += 1
        return [make_result(chains, p, self.shift + k) for k, p in enumerate(self.plddts)]


class SquaredMeanPLDDT(bg.energies.EnergyTerm):
    """Nonlinear in the structure: the mean over samples of the energy differs from the energy of the mean."""

    def __init__(self, oracle, weight: float = 1.0, name: str = 'squared_mean_plddt') -> None:
        super().__init__(name=name, oracle=oracle, inheritable=False, weight=weight)

    def compute(self, oracles_result):
        value = float(np.mean(oracles_result[self.oracle].local_plddt)) ** 2
        return value, value * self.weight


def chains_of(sequence: str = 'ACDEF', chain_id: str = 'A', mutable: bool = True) -> list[bg.Chain]:
    return [bg.Chain([bg.Residue(aa, chain_id, i, mutable=mutable) for i, aa in enumerate(sequence)])]


# ---------------------------------------------------------------------------------------------------------------------
# State energy
# ---------------------------------------------------------------------------------------------------------------------
def test_one_sample_reproduces_the_previous_computation_exactly() -> None:
    oracle = SampledOracle([0.6])
    chains = chains_of()
    term = SquaredMeanPLDDT(oracle, weight=2.0)
    state = bg.State('s', chains, [term])

    previous = bg.oracles.OraclesResultDict()
    previous[oracle] = oracle.predict(chains)
    unweighted, weighted = term.compute(previous)
    oracle.calls = 0

    assert state.energy == weighted  # bit-identical, not just close
    assert state.energy_term_values == {term.name: unweighted}
    assert state.energy_term_stds == {term.name: 0.0}
    assert oracle.calls == 1
    assert state._oracles_result[oracle] is state._sample_results[oracle][0]
    assert len(state._sample_results[oracle]) == 1


def test_energy_is_the_mean_over_samples_of_the_energy_of_each_sample() -> None:
    plddts = [0.8, 0.5, 0.2]
    oracle = SampledOracle(plddts)
    term = SquaredMeanPLDDT(oracle, weight=3.0)
    state = bg.State('s', chains_of(), [term])

    per_sample = np.array(plddts) ** 2
    assert state.energy_term_values[term.name] == pytest.approx(per_sample.mean())
    assert state.energy == pytest.approx(3.0 * per_sample.mean())
    assert state.energy_term_stds[term.name] == pytest.approx(per_sample.std(ddof=1))
    assert state.energy_term_values[term.name] != pytest.approx(np.mean(plddts) ** 2)  # not the energy of the mean
    assert oracle.calls == 1  # one run of the model gives all the samples
    assert [r.local_plddt[0, 0] for r in state._sample_results[oracle]] == pytest.approx(plddts)
    assert state._oracles_result[oracle].local_plddt[0, 0] == pytest.approx(0.8)  # the best sample represents the State


def test_samples_are_cached_and_invalidated_with_the_sequence() -> None:
    oracle = SampledOracle([0.9, 0.3])
    chains = chains_of()
    state = bg.State('s', chains, [SquaredMeanPLDDT(oracle)])
    first = state.energy
    assert state.energy == first and oracle.calls == 1  # reused

    chains[0].mutate_residue(index=0, amino_acid='W')
    assert state.energy == pytest.approx(first)  # same fake pLDDTs ...
    assert oracle.calls == 2  # ... but it was recomputed
    assert len(state._sample_results[oracle]) == 2


def test_weight_changes_only_the_weighted_value() -> None:
    oracle = SampledOracle([0.9, 0.3])
    term = SquaredMeanPLDDT(oracle, weight=1.0)
    state = bg.State('s', chains_of(), [term])
    unweighted = state.energy_term_values[term.name]
    assert state.energy == pytest.approx(unweighted)
    other = bg.State('s2', chains_of(), [SquaredMeanPLDDT(oracle, weight=4.0)])
    assert other.energy_term_values['squared_mean_plddt'] == pytest.approx(unweighted)
    assert other.energy == pytest.approx(4 * unweighted)


def test_an_oracle_with_samples_next_to_one_without_in_the_same_state() -> None:
    sampled, single = SampledOracle([0.8, 0.4]), SampledOracle([0.5])
    state = bg.State(
        's',
        chains_of(),
        [SquaredMeanPLDDT(sampled, name='a'), SquaredMeanPLDDT(single, name='b')],
    )
    assert state.energy_term_values['a'] == pytest.approx((0.64 + 0.16) / 2)
    assert state.energy_term_values['b'] == pytest.approx(0.25)
    assert state.energy_term_stds['b'] == 0.0 and state.energy_term_stds['a'] > 0.0


# ---------------------------------------------------------------------------------------------------------------------
# several oracles in one term
# ---------------------------------------------------------------------------------------------------------------------
def test_term_with_several_oracles_averages_every_structure_of_every_oracle() -> None:
    first, second = SampledOracle([0.2, 0.4]), SampledOracle([0.9])
    term = SquaredMeanPLDDT([first, second], weight=2.0)
    state = bg.State('s', chains_of(), [term])

    assert term.oracle is first and term.oracles == [first, second]
    assert state.oracles_list == [first, second]
    squares = np.array([0.2, 0.4, 0.9]) ** 2  # each structure counts the same (pooled, not a mean of per-oracle means)
    assert state.energy_term_values[term.name] == pytest.approx(squares.mean())
    assert state.energy == pytest.approx(2.0 * squares.mean())
    assert state.energy_term_stds[term.name] == pytest.approx(squares.std(ddof=1))
    assert first.calls == second.calls == 1
    assert term.oracle is first  # the term is left bound to its first oracle


def test_term_with_several_oracles_one_sample_each() -> None:
    first, second = SampledOracle([0.2]), SampledOracle([0.8])
    state = bg.State('s', chains_of(), [SquaredMeanPLDDT([first, second])])
    assert state.energy == pytest.approx((0.04 + 0.64) / 2)


def test_oracle_lists_are_validated() -> None:
    oracle = SampledOracle([0.5])
    with pytest.raises(ValueError, match='at least one oracle'):
        SquaredMeanPLDDT([])
    with pytest.raises(ValueError, match='same oracle more than once'):
        SquaredMeanPLDDT([oracle, oracle])
    with pytest.raises(TypeError, match='must be an Oracle'):
        SquaredMeanPLDDT([oracle, 'esmfold'])
    term = SquaredMeanPLDDT(oracle)
    with pytest.raises(ValueError, match='not one of the oracles'):
        with term.bound_to(SampledOracle([0.1])):
            pass


def test_term_binding_is_restored_even_if_compute_fails() -> None:
    first, second = SampledOracle([0.5]), SampledOracle([0.6])

    class Failing(SquaredMeanPLDDT):
        def compute(self, oracles_result):
            raise RuntimeError('boom')

    term = Failing([first, second])
    state = bg.State('s', chains_of(), [term])
    with pytest.raises(RuntimeError, match='boom'):
        state.energy
    assert term.oracle is first


# ---------------------------------------------------------------------------------------------------------------------
# reference state: every structure against every reference structure
# ---------------------------------------------------------------------------------------------------------------------
class SampledBackboneOracle(_FakeBackboneOracle):
    """Backbone oracle with one displacement profile per sample (best first)."""

    def __init__(self, profiles: list[dict[tuple[str, int], float]]) -> None:
        super().__init__(profiles[0], lambda n: np.full(n, 0.5))
        self.profiles = profiles
        self.n_samples = len(profiles)

    def predict_samples(self, chains):
        results = []
        for profile in self.profiles:
            self.displacements = profile
            results.append(self.fold(chains))
        self.displacements = self.profiles[0]
        return results


def _rmsd_state(complex_profiles, reference_profiles):
    binder, target = _binder_and_target()
    complex_oracle = SampledBackboneOracle(complex_profiles)
    reference_oracle = SampledBackboneOracle(reference_profiles)
    reference_state = bg.State('reference', [binder], [bg.energies.PLDDTEnergy(reference_oracle, None)])
    term = bg.energies.BinderRMSDEnergy(complex_oracle, binder.residues, reference_state)
    return bg.State('complex', [binder, target], [term])


def test_binder_rmsd_is_the_mean_over_all_structure_pairs() -> None:
    complex_profiles = [{('A', 2): 4.0}, {('A', 4): -2.0}, {('A', 1): 1.5, ('A', 3): -3.0}]
    reference_profiles = [{}, {('A', 0): 2.5}]
    # the value of every pair, from the single-structure path (which the n_samples=1 tests pin to the old behaviour)
    pair_values = [
        _rmsd_state([c], [r]).energy_term_values['binder_rmsd']
        for c, r in product(complex_profiles, reference_profiles)
    ]
    assert len(set(np.round(pair_values, 8))) > 1  # the pairs differ, so pairing and averaging matter

    state = _rmsd_state(complex_profiles, reference_profiles)
    assert state.energy_term_values['binder_rmsd'] == pytest.approx(np.mean(pair_values))


def test_binder_rmsd_with_one_structure_each_is_unchanged() -> None:
    from test_energies import BINDER_CA, BINDER_DISPLACEMENTS, _kabsch_deviations, _displaced_binder_ca

    state = _rmsd_state([BINDER_DISPLACEMENTS], [{}])
    expected = float(np.sqrt(np.mean(_kabsch_deviations(BINDER_CA, _displaced_binder_ca()) ** 2)))  # independent
    assert state.energy_term_values['binder_rmsd'] == pytest.approx(expected)
    assert state.energy_term_stds == {'binder_rmsd': 0.0}


# ---------------------------------------------------------------------------------------------------------------------
# loggers
# ---------------------------------------------------------------------------------------------------------------------
def test_default_logger_writes_the_std_only_when_asked(tmp_path: Path) -> None:
    oracle = SampledOracle([0.9, 0.3])
    system = bg.System([bg.State('s', chains_of(), [SquaredMeanPLDDT(oracle)])])

    for with_std in (False, True):
        folder = tmp_path / str(with_std)
        folder.mkdir()
        bg.callbacks.DefaultLogger(log_interval=1, log_sample_std=with_std)._dump_system(0, system, folder)
        header = (folder / 'energies.csv').read_text().splitlines()[0].split(',')
        assert ('s/squared_mean_plddt/std' in header) == with_std
        assert 's/squared_mean_plddt' in header


class SampledOracleA(SampledOracle):
    pass


class SampledOracleB(SampledOracle):
    pass


def test_folding_logger_writes_one_cif_per_oracle_with_the_best_sample(tmp_path: Path) -> None:
    a, b = SampledOracleA([0.9, 0.3], shift=0.0), SampledOracleB([0.8], shift=100.0)
    system = bg.System([bg.State('s', chains_of(), [SquaredMeanPLDDT([a, b])])])
    system.get_total_energy()

    logger = bg.callbacks.FoldingLogger(None, log_interval=1)  # None: every folding oracle
    logger._dump_folding_results_for_system(step=3, system=system, folding_path=tmp_path)

    assert sorted(p.name for p in tmp_path.glob('*.cif')) == ['s_SampledOracleA_3.cif', 's_SampledOracleB_3.cif']
    assert 'CA' in (tmp_path / 's_SampledOracleA_3.cif').read_text()

    only_b = tmp_path / 'only_b'
    bg.callbacks.FoldingLogger(b, log_interval=1)._dump_folding_results_for_system(3, system, only_b)
    assert [p.name for p in only_b.glob('*.cif')] == ['s_SampledOracleB_3.cif']  # the previous single-oracle behaviour


def test_folding_logger_indexes_oracles_of_the_same_class(tmp_path: Path) -> None:
    first, second = SampledOracleA([0.9]), SampledOracleA([0.5])
    system = bg.System([bg.State('s', chains_of(), [SquaredMeanPLDDT([first, second])])])
    system.get_total_energy()
    bg.callbacks.FoldingLogger([first, second], log_interval=1)._dump_folding_results_for_system(0, system, tmp_path)
    assert sorted(p.name for p in tmp_path.glob('*.cif')) == ['s_SampledOracleA_0_0.cif', 's_SampledOracleA_1_0.cif']


def test_folding_logger_argument_validation() -> None:
    with pytest.raises(ValueError, match='empty sequence'):
        bg.callbacks.FoldingLogger([], log_interval=1)
    with pytest.raises(TypeError, match='FoldingOracle'):
        bg.callbacks.FoldingLogger(['x'], log_interval=1)


# ---------------------------------------------------------------------------------------------------------------------
# inverse folding and mosaic
# ---------------------------------------------------------------------------------------------------------------------
def test_inverse_folding_uses_the_best_sample_of_the_oracle() -> None:
    oracle = SampledOracle([0.9, 0.3], shift=0.0)  # sample k is shifted by k: sample 0 is at x = 0
    chains = chains_of()
    state = bg.State('s', chains, [SquaredMeanPLDDT(oracle)])
    structure = bg.mutation.InverseFoldingSampling._get_structure(state)
    assert np.allclose(structure.coord[:, 0], fake_atom_array(chains).coord[:, 0])  # unshifted = best sample
    assert oracle.calls == 1


def test_inverse_folding_reuses_the_samples_of_the_energy_computation() -> None:
    oracle = SampledOracle([0.9, 0.3])
    state = bg.State('s', chains_of(), [SquaredMeanPLDDT(oracle)])
    state.energy
    bg.mutation.InverseFoldingSampling._get_structure(state)
    assert oracle.calls == 1


def test_mosaic_screening_refuses_terms_averaging_several_structures(fake_esmfold2) -> None:
    from bagel.mosaic_screening import MosaicIncompatibleError
    from bagel.mosaic_screening.plan import build_plans

    from test_mosaic_screening import _system

    system = _system(fake_esmfold2, unsupported=False)
    build_plans(system)  # fine with one structure per term

    fake_esmfold2.n_samples = 2
    with pytest.raises(MosaicIncompatibleError, match='averages over several structures'):
        build_plans(system)
    fake_esmfold2.n_samples = 1

    other = bg.oracles.folding.ESMFold2.__new__(bg.oracles.folding.ESMFold2)
    other.n_samples = 1
    system.states[0].energy_terms[0].oracles = [fake_esmfold2, other]
    with pytest.raises(MosaicIncompatibleError, match='2 oracles'):
        build_plans(system)


# ---------------------------------------------------------------------------------------------------------------------
# oracles: n_samples
# ---------------------------------------------------------------------------------------------------------------------
def test_n_samples_is_validated(monkeypatch) -> None:
    monkeypatch.setattr(bg.oracles.folding.RF3, '_load', lambda self, config=None: None)
    for bad in (0, -1, True, 2.0, '2'):
        with pytest.raises(ValueError, match='n_samples must be a positive integer'):
            bg.oracles.folding.RF3(n_samples=bad)
    assert bg.oracles.folding.RF3().n_samples == 1  # the default keeps a single structure per prediction


def test_deterministic_esmfold_refuses_several_samples(monkeypatch) -> None:
    monkeypatch.setattr(bg.oracles.folding.ESMFold, '_load', lambda self, config=None: None)
    with pytest.raises(ValueError, match='at most 1 sample.*deterministic'):
        bg.oracles.folding.ESMFold(n_samples=2)
    assert bg.oracles.folding.ESMFold().n_samples == 1


def test_alphafold2_has_five_models(monkeypatch) -> None:
    monkeypatch.setattr(bg.oracles.folding.AlphaFold2Multimer, '_load', lambda self, config=None: None)
    assert bg.oracles.folding.AlphaFold2Multimer(n_samples=5).n_samples == 5
    with pytest.raises(ValueError, match='at most 5'):
        bg.oracles.folding.AlphaFold2Multimer(n_samples=6)


@pytest.mark.parametrize(
    ('oracle_name', 'module', 'key', 'default'),
    [
        ('RF3', 'boileroom.models.rf3.rf3', 'diffusion_batch_size', 5),
        ('Protenix', 'boileroom.models.protenix.protenix', 'sample', 5),
        ('OpenDDE', 'boileroom.models.opendde.opendde', 'sample', 5),
        ('Boltz2', 'boileroom.models.boltz.boltz2', 'diffusion_samples', 1),
        ('AlphaFold2Multimer', 'boileroom.models.alphafold.alphafold2_multimer', 'num_models', 5),
    ],
)
def test_the_model_is_asked_for_enough_samples(monkeypatch, oracle_name, module, key, default) -> None:
    import importlib

    captured = {}

    class FakeBoiler:
        def __init__(self, backend, device, config):
            captured['config'] = config

    monkeypatch.setattr(importlib.import_module(module), oracle_name, FakeBoiler)
    oracle_class = getattr(bg.oracles.folding, oracle_name)

    n = default + 2 if default + 2 <= 5 or oracle_name != 'AlphaFold2Multimer' else 5
    oracle_class(n_samples=n)
    assert (captured['config'] or {}).get(key, default) >= n  # raised if the model's default is too small
    if n > default:
        assert captured['config'][key] == n
    with pytest.raises(ValueError, match='fewer samples than n_samples'):
        oracle_class(n_samples=n, config={key: n - 1})


def test_boltz2_msa_cache_is_off_unless_the_user_turns_it_on(monkeypatch) -> None:
    """boileroom's Boltz-2 MSA cache reuses stale MSAs in no-MSA mode and crashes on mixed chains: off by default."""
    import importlib

    captured = {}

    class FakeBoiler:
        def __init__(self, backend, device, config):
            captured['config'] = config

    monkeypatch.setattr(importlib.import_module('boileroom.models.boltz.boltz2'), 'Boltz2', FakeBoiler)

    bg.oracles.folding.Boltz2()
    assert captured['config'] == {'msa_cache_enabled': False}
    bg.oracles.folding.Boltz2(config={'use_msa_server': False})  # no-MSA mode keeps the cache off
    assert captured['config'] == {'use_msa_server': False, 'msa_cache_enabled': False}
    bg.oracles.folding.Boltz2(config={'sampling_steps': 10}, n_samples=3)
    assert captured['config'] == {'sampling_steps': 10, 'diffusion_samples': 3, 'msa_cache_enabled': False}
    bg.oracles.folding.Boltz2(config={'msa_cache_enabled': True})  # an explicit choice is respected
    assert captured['config'] == {'msa_cache_enabled': True}


@pytest.mark.parametrize('n_samples', [1, 2, 3, 5])
def test_rf3_keeps_the_n_best_ranked_samples(monkeypatch, monomer, n_samples) -> None:
    output = five_sample_output(RF3Output, monomer, best=3)
    monkeypatch.setattr(
        bg.oracles.folding.RF3, '_load', lambda self, config=None: setattr(self, 'model', FakeModel(output))
    )
    oracle = bg.oracles.folding.RF3(n_samples=n_samples)

    samples = oracle.fold_samples(monomer)

    assert len(samples) == n_samples == len(oracle.predict_samples(monomer))
    # five_sample_output gives sample i a pLDDT of 0.1 * (i + 1) and lists the best (3) first, then 0, 1, 2, 4
    assert [round(float(s.local_plddt[0, 0]), 1) for s in samples] == [0.4, 0.1, 0.2, 0.3, 0.5][:n_samples]
    assert oracle.fold(monomer).local_plddt[0, 0] == pytest.approx(0.4)  # fold() is the best one


def test_fewer_samples_than_requested_is_an_error(monkeypatch, monomer) -> None:
    output = five_sample_output(RF3Output, monomer, best=0)
    monkeypatch.setattr(
        bg.oracles.folding.RF3, '_load', lambda self, config=None: setattr(self, 'model', FakeModel(output))
    )
    with pytest.raises(ValueError, match='returned 5 sample.*fewer than n_samples=6'):
        bg.oracles.folding.RF3(n_samples=6).fold_samples(monomer)


def test_alphafold2_keeps_the_first_models(monkeypatch, monomer) -> None:
    output = five_sample_output(AlphaFold2MultimerOutput, monomer, best=2)
    monkeypatch.setattr(
        bg.oracles.folding.AlphaFold2Multimer,
        '_load',
        lambda self, config=None: setattr(self, 'model', FakeModel(output)),
    )
    samples = bg.oracles.folding.AlphaFold2Multimer(n_samples=2).fold_samples(monomer)
    assert [round(float(s.local_plddt[0, 0]), 1) for s in samples] == [0.3, 0.1]  # models are listed best first


@pytest.mark.parametrize('oracle_name', ['ESMFold2', 'Boltz2'])
def test_models_without_ranking_are_ranked_by_ptm_then_plddt(monkeypatch, monomer, oracle_name) -> None:
    n = len(monomer[0].residues)
    ptm = [0.5, 0.9, 0.9, 0.7]
    mean_plddt = [0.99, 0.6, 0.8, 0.9]  # samples 1 and 2 tie on pTM: the higher pLDDT (2) wins
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer) for _ in ptm],
        plddt=[np.full(n, p) for p in mean_plddt],
        pae=[np.zeros((n, n)) for _ in ptm],
        ptm=[np.array(p) for p in ptm],
    )
    oracle_class = getattr(bg.oracles.folding, oracle_name)
    monkeypatch.setattr(oracle_class, '_load', lambda self, config=None: setattr(self, 'model', FakeModel(output)))

    samples = oracle_class(n_samples=4).fold_samples(monomer)

    assert [round(float(s.local_plddt[0, 0]), 2) for s in samples] == [0.8, 0.6, 0.9, 0.99]  # sample order 2, 1, 3, 0
    assert oracle_class().fold(monomer).local_plddt[0, 0] == pytest.approx(0.8)  # one sample: still the best


def test_esmfold2_asks_for_n_samples_only_when_more_than_one(monkeypatch, monomer) -> None:
    n = len(monomer[0].residues)
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)] * 2,
        plddt=[np.full(n, 0.8)] * 2,
        pae=[np.zeros((n, n))] * 2,
        ptm=[np.array(0.7)] * 2,
    )
    model = FakeModel(output)
    monkeypatch.setattr(bg.oracles.folding.ESMFold2, '_load', lambda self, config=None: setattr(self, 'model', model))

    bg.oracles.folding.ESMFold2().fold(monomer)
    bg.oracles.folding.ESMFold2(n_samples=2).fold_samples(monomer)

    assert 'num_diffusion_samples' not in model.calls[0][1]  # the options of a single prediction are unchanged
    assert model.calls[1][1]['num_diffusion_samples'] == 2


def test_state_runs_a_multi_sample_oracle_once_and_keeps_its_best_sample(monkeypatch, monomer) -> None:
    output = five_sample_output(RF3Output, monomer, best=4)
    model = FakeModel(output)
    monkeypatch.setattr(bg.oracles.folding.RF3, '_load', lambda self, config=None: setattr(self, 'model', model))
    oracle = bg.oracles.folding.RF3(n_samples=3)
    state = bg.State('s', monomer, [bg.energies.OverallPLDDTEnergy(oracle=oracle, weight=1.0)])

    state.energy

    assert len(model.calls) == 1
    assert len(state._sample_results[oracle]) == 3
    assert state._oracles_result[oracle].local_plddt[0, 0] == pytest.approx(0.5)  # the best of the five
    assert len(state.energy_term_stds) == 1


def test_copying_a_system_keeps_the_samples_and_the_oracles() -> None:
    """System copies deep-copy the cached samples; the oracles themselves stay shared, so nothing is refolded."""
    oracle = SampledOracle([0.9, 0.3])
    term = SquaredMeanPLDDT([oracle], weight=2.0)
    system = bg.System([bg.State('s', chains_of(), [term])])
    energy = system.get_total_energy()

    copied = system.__copy__()
    state = copied.states[0]

    assert state.energy_terms[0].oracles[0] is oracle  # shared, not copied
    assert list(state._sample_results) == [oracle]
    assert len(state._sample_results[oracle]) == 2
    assert state.energy == energy and oracle.calls == 1  # served from the copied cache


@pytest.mark.parametrize(
    'make_term',
    [
        lambda oracles: bg.energies.PTMEnergy(oracle=oracles),
        lambda oracles: bg.energies.OverallPLDDTEnergy(oracle=oracles),
        lambda oracles: bg.energies.PLDDTEnergy(oracle=oracles, residues=None),
        lambda oracles: bg.energies.SurfaceAreaEnergy(oracle=oracles, residues=None),
    ],
    ids=['PTM', 'OverallPLDDT', 'PLDDT', 'SurfaceArea'],
)
def test_every_oracle_of_a_term_is_checked_when_the_term_is_built(fake_esm2, make_term) -> None:
    """An oracle of the wrong kind later in the list must fail at construction, not when the energy is computed."""
    folding = SampledOracle([0.5])
    make_term([folding])  # a valid list is fine
    with pytest.raises(AssertionError):
        make_term([folding, fake_esm2])
    with pytest.raises(AssertionError):
        make_term([fake_esm2, folding])


def _sae_oracle(model_id: str = 'esmc-6b-2024-12-sae-layer60-k64-codebook16384'):
    from bagel.oracles.embedding.sae import SAE, SAEResult

    oracle = SAE.__new__(SAE)
    oracle.result_class = SAEResult
    oracle.sae_model_id = model_id
    oracle.sae_identity_tokens = ''
    return oracle


def _residues():
    chains = chains_of('ACDEFGH')
    return chains[0].residues


@pytest.mark.parametrize(
    ('make_term', 'good', 'bad'),
    [
        (
            lambda o: bg.energies.PAEEnergy(oracle=o, residues=[_residues()[:3], _residues()[3:]]),
            'folding',
            'embedding',
        ),
        (
            lambda o: bg.energies.ipSAEEnergy(oracle=o, residues=[_residues()[:3], _residues()[3:]]),
            'folding',
            'embedding',
        ),
        (lambda o: bg.energies.SAEnergy(oracle=o, feature_indices=[0, 1]), 'sae', 'folding'),
        (lambda o: bg.energies.SAEnergy(oracle=o, feature_indices=[0, 1]), 'sae', 'unsupported_sae'),
    ],
    ids=['PAE', 'ipSAE', 'SAEnergy-not-an-embedding-oracle', 'SAEnergy-unsupported-sae-model'],
)
def test_every_oracle_is_checked_by_pae_ipsae_and_sae_terms(fake_esm2, make_term, good, bad) -> None:
    oracles = {
        'folding': SampledOracle([0.5]),
        'embedding': fake_esm2,
        'sae': _sae_oracle(),
        'unsupported_sae': _sae_oracle(model_id='some-other-sae'),
    }
    make_term([oracles[good]])
    with pytest.raises((AssertionError, ValueError)):
        make_term([oracles[good], oracles[bad]])  # a bad oracle after a good one
    with pytest.raises((AssertionError, ValueError)):
        make_term([oracles[bad], oracles[good]])


def test_binder_rmsd_checks_every_oracle(fake_esm2) -> None:
    binder, _ = _binder_and_target()
    reference = bg.State('reference', [binder], [bg.energies.PLDDTEnergy(SampledOracle([0.5]), None)])
    good = SampledOracle([0.5])
    bg.energies.BinderRMSDEnergy([good], binder.residues, reference)
    with pytest.raises(AssertionError):
        bg.energies.BinderRMSDEnergy([good, fake_esm2], binder.residues, reference)


def test_invalidation_clears_the_sample_caches() -> None:
    oracle = SampledOracle([0.9, 0.3])
    chains = chains_of()
    state = bg.State('s', chains, [SquaredMeanPLDDT(oracle)])
    state.energy
    assert state._sample_results and state._energy_term_stds

    chains[0].mutate_residue(index=0, amino_acid='W')
    state._invalidate_cache_if_needed()

    assert state._sample_results == {} and state._energy_term_stds == {} and len(state._oracles_result) == 0


def test_samples_are_ignored_when_the_best_result_was_replaced_directly() -> None:
    """A result written straight into ``_oracles_result`` must not be shadowed by older samples."""
    oracle = SampledOracle([0.9, 0.3])
    chains = chains_of()
    state = bg.State('s', chains, [SquaredMeanPLDDT(oracle)])
    state.energy
    replacement = make_result(chains, 0.1)
    state._oracles_result[oracle] = replacement
    assert state._samples_of(oracle) == [replacement]


def test_numpy_integers_are_valid_n_samples(monkeypatch) -> None:
    monkeypatch.setattr(bg.oracles.folding.RF3, '_load', lambda self, config=None: None)
    oracle = bg.oracles.folding.RF3(n_samples=np.int64(3))
    assert oracle.n_samples == 3 and type(oracle.n_samples) is int


def test_ranking_refuses_non_finite_scores_and_inconsistent_ranks(monkeypatch, monomer) -> None:
    n = len(monomer[0].residues)

    def output(ptm, plddt=0.8, **extra):
        return SimpleNamespace(
            atom_array=[fake_atom_array(monomer)] * len(ptm),
            plddt=[np.full(n, plddt)] * len(ptm),
            pae=[np.zeros((n, n))] * len(ptm),
            ptm=[np.array(p) for p in ptm],
            **extra,
        )

    monkeypatch.setattr(bg.oracles.folding.Boltz2, '_load', lambda self, config=None: None)
    boltz = bg.oracles.folding.Boltz2(n_samples=2)
    with pytest.raises(ValueError, match='non-finite pTM or pLDDT'):
        boltz._reduce_samples(output([0.5, float('nan')]), monomer)
    with pytest.raises(ValueError, match='non-finite pTM or pLDDT'):
        boltz._reduce_samples(output([0.5, 0.6], plddt=float('nan')), monomer)

    monkeypatch.setattr(bg.oracles.folding.RF3, '_load', lambda self, config=None: None)
    with pytest.raises(ValueError, match='reported 3 sample ranks for 2 structure'):
        bg.oracles.folding.RF3(n_samples=2)._reduce_samples(output([0.5, 0.6], sample_ranks=[0, 1, 2]), monomer)


class _NoPaeResult(bg.oracles.folding.FoldingResult):
    """A folding result that has a structure but no ``pae``."""

    def save_attributes(self, filepath: Path) -> None:
        return


class _NoStructureResult(bg.oracles.OracleResult):
    """A result of a folding oracle without a ``structure`` field."""

    def save_attributes(self, filepath: Path) -> None:
        return


class _StubFolding(bg.oracles.folding.FoldingOracle):
    def __init__(self, result_class) -> None:
        self.result_class = result_class

    def fold(self, chains):
        raise NotImplementedError


def test_the_result_fields_a_term_needs_are_checked_for_every_oracle() -> None:
    """A folding oracle whose result lacks the needed field fails the term's construction, wherever it is listed."""
    good = SampledOracle([0.5])
    no_pae, no_structure = _StubFolding(_NoPaeResult), _StubFolding(_NoStructureResult)
    groups = [_residues()[:3], _residues()[3:]]
    binder, _ = _binder_and_target()
    reference = bg.State('reference', [binder], [bg.energies.PLDDTEnergy(SampledOracle([0.5]), None)])

    needs_pae = [
        lambda o: bg.energies.PAEEnergy(oracle=o, residues=groups),
        lambda o: bg.energies.ipSAEEnergy(oracle=o, residues=groups),
    ]
    for make in needs_pae:
        make([good])
        for oracles in ([good, no_pae], [no_pae, good]):
            with pytest.raises(AssertionError, match='pae'):
                make(oracles)
    for oracles in ([good, no_structure], [no_structure, good]):
        with pytest.raises(AssertionError, match='structure'):
            bg.energies.SurfaceAreaEnergy(oracle=oracles, residues=None)
        with pytest.raises(AssertionError, match='structure'):
            bg.energies.BinderRMSDEnergy(oracles, binder.residues, reference)


def test_single_structure_output_is_not_ranked_or_validated(monkeypatch, monomer) -> None:
    """With one structure the output is passed on as before, even with values the ranking would refuse."""
    n = len(monomer[0].residues)
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(n, 0.8)],
        pae=[np.zeros((n, n))],
        ptm=[np.array(0.7)],
        sample_ranks=[0, 1, 2],  # inconsistent with one structure, but nothing needs ranking
    )
    monkeypatch.setattr(bg.oracles.folding.RF3, '_load', lambda self, config=None: None)
    results = bg.oracles.folding.RF3()._reduce_samples(output, monomer)
    assert len(results) == 1 and results[0].ptm[0, 0] == pytest.approx(0.7)
