"""Tests for the RF3, Protenix, OpenDDE and AlphaFold2Multimer oracles and for user-supplied MSAs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import bagel as bg
from boileroom.base import PredictionMetadata
from boileroom.models.alphafold.types import AlphaFold2MultimerOutput
from boileroom.models.opendde.types import OpenDDEOutput
from boileroom.models.protenix.types import ProtenixOutput
from boileroom.models.rf3.types import RF3Output

from test_folding_oracle_wrappers import FakeModel, fake_atom_array, prediction_metadata

MODELS = [
    (bg.oracles.folding.RF3, bg.oracles.folding.RF3Result, RF3Output),
    (bg.oracles.folding.Protenix, bg.oracles.folding.ProtenixResult, ProtenixOutput),
    (bg.oracles.folding.OpenDDE, bg.oracles.folding.OpenDDEResult, OpenDDEOutput),
    (bg.oracles.folding.AlphaFold2Multimer, bg.oracles.folding.AlphaFold2MultimerResult, AlphaFold2MultimerOutput),
]
MODEL_IDS = [m[0].__name__ for m in MODELS]


def five_sample_output(output_class, chains: list[bg.Chain], best: int):
    """An output with 5 samples whose pLDDT is ``0.1 * (i + 1)``; sample ``best`` is the top-ranked one."""
    n = sum(len(chain.residues) for chain in chains)
    kwargs = dict(
        metadata=prediction_metadata(n),
        atom_array=[fake_atom_array(chains) for _ in range(5)],
        plddt=[np.full(n, 0.1 * (i + 1)) for i in range(5)],
        pae=[np.full((n, n), float(i)) for i in range(5)],
        ptm=[np.array(0.1 * (i + 1)) for i in range(5)],
    )
    if output_class is RF3Output:
        # RF3 lists samples best first; sample_ranks says so explicitly
        order = [best] + [i for i in range(5) if i != best]
        kwargs = {k: ([v[i] for i in order] if isinstance(v, list) else v) for k, v in kwargs.items()}
        kwargs['sample_ranks'] = list(range(5))
    elif output_class in (ProtenixOutput, OpenDDEOutput):
        # Protenix lists samples by (seed, rank); sample_ranks gives the confidence rank of each listed sample
        kwargs['sample_ranks'] = [(i - best) % 5 for i in range(5)]
        kwargs['seeds'] = [101] * 5
    else:
        # AlphaFold lists models best first, without explicit ranks
        order = [best] + [i for i in range(5) if i != best]
        kwargs = {k: ([v[i] for i in order] if isinstance(v, list) else v) for k, v in kwargs.items()}
    return output_class(**kwargs)


@pytest.fixture
def make_oracle(monkeypatch):
    def _make(oracle_class, output, **kwargs):
        fake_model = FakeModel(output)

        def mock_load(self, config=None):
            self.model = fake_model
            self.loaded_config = config

        monkeypatch.setattr(oracle_class, '_load', mock_load)
        return oracle_class(backend='modal', **kwargs), fake_model

    return _make


@pytest.mark.parametrize(('oracle_class', 'result_class', 'output_class'), MODELS, ids=MODEL_IDS)
def test_oracle_takes_top_ranked_sample(make_oracle, monomer: list[bg.Chain], oracle_class, result_class, output_class):
    best = 3
    output = five_sample_output(output_class, monomer, best=best)
    oracle, model = make_oracle(oracle_class, output)

    result = oracle.fold(monomer)

    n = len(monomer[0].residues)
    assert isinstance(result, result_class)
    assert model.calls == [([monomer[0].sequence], {'include_fields': ['plddt', 'pae', 'ptm']})]
    # the best sample has pLDDT 0.1 * (best + 1), PAE equal to `best`
    assert np.allclose(result.local_plddt, 0.1 * (best + 1))
    assert np.allclose(result.pae, float(best))
    assert result.ptm[0, 0] == pytest.approx(0.1 * (best + 1))
    assert result.local_plddt.shape == (1, n)
    assert result.pae.shape == (1, n, n)
    assert result.ptm.shape == (1, 1)
    assert np.all(result.structure.chain_id == monomer[0].chain_ID)


@pytest.mark.parametrize(('oracle_class', 'result_class', 'output_class'), MODELS, ids=MODEL_IDS)
def test_oracle_accepts_single_sample_output(make_oracle, monomer, oracle_class, result_class, output_class):
    n = len(monomer[0].residues)
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(n, 0.8)],
        pae=[np.zeros((n, n))],
        ptm=[np.array(0.7)],
    )
    oracle, _ = make_oracle(oracle_class, output)
    result = oracle.fold(monomer)
    assert np.allclose(result.local_plddt, 0.8)


@pytest.mark.parametrize(('oracle_class', 'result_class', 'output_class'), MODELS, ids=MODEL_IDS)
def test_oracle_rejects_inconsistent_sample_counts(make_oracle, monomer, oracle_class, result_class, output_class):
    n = len(monomer[0].residues)
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)] * 2,
        plddt=[np.full(n, 0.8)],
        pae=[np.zeros((n, n))] * 2,
        ptm=[np.array(0.7)] * 2,
    )
    oracle, _ = make_oracle(oracle_class, output)
    with pytest.raises(ValueError, match='plddt has 1 samples; expected 2'):
        oracle.fold(monomer)


@pytest.mark.parametrize(('oracle_class', 'result_class', 'output_class'), MODELS, ids=MODEL_IDS)
def test_oracle_folds_multimers(make_oracle, complex_chains, oracle_class, result_class, output_class):
    output = five_sample_output(output_class, complex_chains, best=0)
    oracle, model = make_oracle(oracle_class, output)
    result = oracle.fold(complex_chains)
    sequences, _ = model.calls[0]
    assert sequences == [':'.join(chain.sequence for chain in complex_chains)]
    assert set(result.structure.chain_id) == {chain.chain_ID for chain in complex_chains}


def a3m_for(sequence: str) -> str:
    return f'>query\n{sequence}\n>hit1\n{sequence}\n>hit2\n{sequence[:-1]}-\n'


@pytest.fixture
def complex_chains() -> list[bg.Chain]:
    return [
        bg.Chain([bg.Residue(name=aa, chain_ID='T', index=i) for i, aa in enumerate('ACDEF')]),
        bg.Chain([bg.Residue(name=aa, chain_ID='B', index=i, mutable=True) for i, aa in enumerate('GHIK')]),
    ]


@pytest.mark.parametrize(('oracle_class', 'result_class', 'output_class'), MODELS, ids=MODEL_IDS)
def test_msa_is_forwarded_per_chain_with_none_for_designed_chain(
    make_oracle, complex_chains, oracle_class, result_class, output_class
):
    target = complex_chains[0]
    output = five_sample_output(output_class, complex_chains, best=0)
    oracle, model = make_oracle(oracle_class, output, msa={'T': a3m_for(target.sequence)})

    oracle.fold(complex_chains)

    _, options = model.calls[0]
    assert options['msa'] == [a3m_for(target.sequence), None]
    assert options['include_fields'] == ['plddt', 'pae', 'ptm']


def test_msa_is_read_from_file_once_at_construction(make_oracle, complex_chains, tmp_path: Path) -> None:
    target = complex_chains[0]
    path = tmp_path / 'target.a3m'
    path.write_text(a3m_for(target.sequence))
    output = five_sample_output(RF3Output, complex_chains, best=0)
    oracle, model = make_oracle(bg.oracles.folding.RF3, output, msa={'T': path})

    path.unlink()  # a later fold must not touch the file: the text was stored at construction
    oracle.fold(complex_chains)
    oracle.fold(complex_chains)

    assert [call[1]['msa'] for call in model.calls] == [[a3m_for(target.sequence), None]] * 2


def test_msa_for_a_mutated_chain_is_refused(make_oracle, complex_chains) -> None:
    output = five_sample_output(RF3Output, complex_chains, best=0)
    oracle, model = make_oracle(bg.oracles.folding.RF3, output, msa={'B': a3m_for('GHIK')})
    complex_chains[1].mutate_residue(0, 'W')
    with pytest.raises(ValueError, match="MSA of chain 'B' does not match"):
        oracle.fold(complex_chains)
    assert model.calls == []


def test_msa_for_absent_chain_folds_without_msa(make_oracle, monomer) -> None:
    n = len(monomer[0].residues)
    output = five_sample_output(RF3Output, monomer, best=0)
    oracle, model = make_oracle(bg.oracles.folding.RF3, output, msa={'other': a3m_for('ACDEF')})
    oracle.fold(monomer)
    assert 'msa' not in model.calls[0][1]


def test_missing_msa_file_and_invalid_a3m_fail_at_construction(make_oracle, monomer, tmp_path: Path) -> None:
    output = five_sample_output(RF3Output, monomer, best=0)
    with pytest.raises(FileNotFoundError):
        make_oracle(bg.oracles.folding.RF3, output, msa={'A': tmp_path / 'missing.a3m'})
    bad = tmp_path / 'bad.a3m'
    bad.write_text('not an alignment')
    with pytest.raises(FileNotFoundError):  # not starting with '>' and not a file: treated as a (missing) path
        make_oracle(bg.oracles.folding.RF3, output, msa={'A': 'not an alignment'})
    with pytest.raises(ValueError, match='not valid A3M'):
        make_oracle(bg.oracles.folding.RF3, output, msa={'A': '>only header\n'})


@pytest.mark.parametrize(
    'oracle_class', [bg.oracles.folding.Boltz2, bg.oracles.folding.Chai1, bg.oracles.folding.ESMFold]
)
def test_msa_is_refused_by_models_without_user_msa(monkeypatch, oracle_class) -> None:
    monkeypatch.setattr(oracle_class, '_load', lambda self, config=None: None)
    with pytest.raises(ValueError, match='does not support a user-supplied MSA'):
        oracle_class(msa={'A': '>q\nAAA\n'})


def test_esmfold2_forwards_msa_per_chain(monkeypatch, complex_chains) -> None:
    n = sum(len(chain.residues) for chain in complex_chains)
    output = SimpleNamespace(
        atom_array=[fake_atom_array(complex_chains)],
        plddt=[np.full(n, 0.8)],
        pae=[np.zeros((n, n))],
        ptm=[np.array(0.7)],
    )
    fake_model = FakeModel(output)
    monkeypatch.setattr(
        bg.oracles.folding.ESMFold2, '_load', lambda self, config=None: setattr(self, 'model', fake_model)
    )
    msa = a3m_for(complex_chains[0].sequence)

    bg.oracles.folding.ESMFold2(msa={'T': msa}).fold(complex_chains)

    fold_input, options = fake_model.calls[0]
    assert [protein.id for protein in fold_input.sequences] == ['T', 'B']
    assert options['msa'] == [msa, None]


@pytest.mark.parametrize(
    ('oracle_name', 'module', 'with_msa', 'expected'),
    [
        ('Protenix', 'boileroom.models.protenix.protenix', False, {'use_msa': False}),
        ('Protenix', 'boileroom.models.protenix.protenix', True, {'use_msa': True}),
        ('OpenDDE', 'boileroom.models.opendde.opendde', False, {'use_msa': False}),
        ('OpenDDE', 'boileroom.models.opendde.opendde', True, {'use_msa': True}),
        ('AlphaFold2Multimer', 'boileroom.models.alphafold.alphafold2_multimer', False, {'use_msa_server': False}),
        ('AlphaFold2Multimer', 'boileroom.models.alphafold.alphafold2_multimer', True, {'use_msa_server': False}),
        ('RF3', 'boileroom.models.rf3.rf3', False, {}),
    ],
)
def test_oracles_never_search_an_msa_by_default(monkeypatch, oracle_name, module, with_msa, expected) -> None:
    import importlib

    captured = {}

    class FakeBoiler:
        def __init__(self, backend, device, config):
            captured['config'] = config

    monkeypatch.setattr(importlib.import_module(module), oracle_name, FakeBoiler)
    oracle_class = getattr(bg.oracles.folding, oracle_name)
    oracle_class(msa={'A': a3m_for('ACD')} if with_msa else None)
    assert captured['config'] == expected


def test_user_config_overrides_msa_defaults(monkeypatch) -> None:
    import importlib

    captured = {}

    class FakeBoiler:
        def __init__(self, backend, device, config):
            captured['config'] = config

    monkeypatch.setattr(importlib.import_module('boileroom.models.protenix.protenix'), 'Protenix', FakeBoiler)
    bg.oracles.folding.Protenix(config={'use_msa': True, 'optimization': 'exact'})
    assert captured['config'] == {'use_msa': True, 'optimization': 'exact'}


@pytest.mark.parametrize('oracle_name', ['ESMFold2'])
def test_esmfold2_forwards_optimization_config(monkeypatch, oracle_name) -> None:
    import importlib

    captured = {}

    class FakeBoiler:
        def __init__(self, backend, device, config):
            captured['config'] = config

    monkeypatch.setattr(importlib.import_module('boileroom.models.esmfold2.esmfold2'), 'ESMFold2', FakeBoiler)
    bg.oracles.folding.ESMFold2(config={'optimization': 'exact'})
    assert captured['config'] == {'optimization': 'exact'}


@pytest.mark.parametrize('oracle_name', ['Protenix', 'OpenDDE'])
def test_several_seeds_are_refused(monkeypatch, oracle_name) -> None:
    import importlib

    module = {'Protenix': 'boileroom.models.protenix.protenix', 'OpenDDE': 'boileroom.models.opendde.opendde'}[
        oracle_name
    ]
    monkeypatch.setattr(importlib.import_module(module), oracle_name, lambda **kwargs: None)
    oracle_class = getattr(bg.oracles.folding, oracle_name)
    with pytest.raises(ValueError, match='exactly one seed'):
        oracle_class(config={'seeds': '1,2'})
    oracle_class(config={'seeds': '7'})


@pytest.mark.parametrize(
    ('oracle_class', 'output_class'),
    [(bg.oracles.folding.Protenix, ProtenixOutput), (bg.oracles.folding.OpenDDE, OpenDDEOutput)],
)
def test_unmatched_msa_is_passed_as_explicit_no_msa(make_oracle, monomer, oracle_class, output_class) -> None:
    """use_msa is on when MSAs are given; a state without those chains must say 'no MSA' or the model would search."""
    output = five_sample_output(output_class, monomer, best=0)
    oracle, model = make_oracle(oracle_class, output, msa={'other': a3m_for('ACDEF')})
    oracle.fold(monomer)
    assert model.calls[0][1]['msa'] == [None]


def test_alphafold2_refuses_asymmetric_msa_for_identical_chains(make_oracle, dimer: list[bg.Chain]) -> None:
    """AF2 shares one MSA between equal sequences, so an MSA on only one copy would silently apply to both."""
    a3m = a3m_for(dimer[0].sequence)
    output = five_sample_output(AlphaFold2MultimerOutput, dimer, best=0)

    oracle, model = make_oracle(bg.oracles.folding.AlphaFold2Multimer, output, msa={dimer[0].chain_ID: a3m})
    with pytest.raises(ValueError, match='same sequence as another chain'):
        oracle.fold(dimer)
    assert model.calls == []

    both = {chain.chain_ID: a3m for chain in dimer}
    oracle, model = make_oracle(bg.oracles.folding.AlphaFold2Multimer, output, msa=both)
    oracle.fold(dimer)
    assert model.calls[0][1]['msa'] == [a3m, a3m]


def test_other_models_allow_asymmetric_msa_for_identical_chains(make_oracle, dimer: list[bg.Chain]) -> None:
    a3m = a3m_for(dimer[0].sequence)
    output = five_sample_output(RF3Output, dimer, best=0)
    oracle, model = make_oracle(bg.oracles.folding.RF3, output, msa={dimer[0].chain_ID: a3m})
    oracle.fold(dimer)
    assert model.calls[0][1]['msa'] == [a3m, None]
