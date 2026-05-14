from types import SimpleNamespace

import numpy as np
import pytest
import bagel as bg
from biotite.structure import Atom, array


class FakeModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def fold(self, sequences, options=None):
        self.calls.append((sequences, options))
        return self.output


def fake_atom_array(chains: list[bg.Chain]):
    atoms = []
    for chain_index, chain in enumerate(chains):
        model_chain_id = chr(ord('A') + chain_index)
        for residue in chain.residues:
            atoms.append(
                Atom(
                    coord=[0.0, 0.0, 0.0],
                    chain_id=model_chain_id,
                    res_id=residue.index + 1,
                    res_name=residue.three_letter_name,
                    atom_name='CA',
                    element='C',
                )
            )
    return array(atoms)


@pytest.mark.parametrize(
    ('oracle_class', 'result_class', 'output_fields', 'expected_fields', 'expected_ptm'),
    [
        (
            bg.oracles.folding.Boltz2,
            bg.oracles.folding.Boltz2Result,
            {'plddt': [np.full(20, 80.0)], 'pae': [np.zeros((20, 20))], 'confidence': [{'ptm': 0.7}]},
            ['plddt', 'pae', 'confidence'],
            0.7,
        ),
        (
            bg.oracles.folding.Chai1,
            bg.oracles.folding.Chai1Result,
            {'plddt': [np.full(20, 80.0)], 'pae': [np.zeros((20, 20))], 'ptm': [0.7]},
            ['plddt', 'pae', 'ptm'],
            0.7,
        ),
    ],
)
def test_folding_oracle_wrappers_request_required_fields(
    monkeypatch,
    monomer: list[bg.Chain],
    oracle_class,
    result_class,
    output_fields,
    expected_fields,
    expected_ptm,
) -> None:
    output = SimpleNamespace(atom_array=[fake_atom_array(monomer)], **output_fields)
    fake_model = FakeModel(output)

    def mock_load(self, config={}):
        self.model = fake_model

    monkeypatch.setattr(oracle_class, '_load', mock_load)

    oracle = oracle_class(backend='modal')
    result = oracle.fold(monomer)

    assert isinstance(result, result_class)
    assert fake_model.calls == [([monomer[0].sequence], {'include_fields': expected_fields})]
    assert np.all(result.structure.chain_id == monomer[0].chain_ID)
    assert np.array_equal(np.unique(result.structure.res_id), np.arange(len(monomer[0].residues)))
    assert np.allclose(result.local_plddt, 0.8)
    assert result.pae.shape == (1, len(monomer[0].residues), len(monomer[0].residues))
    assert result.ptm.shape == (1, 1)
    assert result.ptm[0, 0] == pytest.approx(expected_ptm)


def test_chai1_wrapper_defaults_missing_ptm(monkeypatch, monomer: list[bg.Chain]) -> None:
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(20, 80.0)],
        pae=[np.zeros((20, 20))],
        ptm=None,
    )
    fake_model = FakeModel(output)

    def mock_load(self, config={}):
        self.model = fake_model

    monkeypatch.setattr(bg.oracles.folding.Chai1, '_load', mock_load)

    result = bg.oracles.folding.Chai1(backend='modal').fold(monomer)

    assert isinstance(result, bg.oracles.folding.Chai1Result)
    assert np.array_equal(result.ptm, np.array([[0.0]]))


@pytest.mark.parametrize(
    'ptm_input',
    [
        pytest.param([np.asarray(0.73)], id='zero_dim_ndarray'),
        pytest.param([np.array([0.73])], id='one_dim_single_element'),
        pytest.param([0.73], id='python_scalar'),
    ],
)
def test_chai1_wrapper_handles_ptm_shapes(monkeypatch, monomer: list[bg.Chain], ptm_input) -> None:
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(20, 80.0)],
        pae=[np.zeros((20, 20))],
        ptm=ptm_input,
    )
    fake_model = FakeModel(output)

    def mock_load(self, config={}):
        self.model = fake_model

    monkeypatch.setattr(bg.oracles.folding.Chai1, '_load', mock_load)

    result = bg.oracles.folding.Chai1(backend='modal').fold(monomer)

    assert result.ptm.shape == (1, 1)
    assert result.ptm[0, 0] == pytest.approx(0.73)


@pytest.mark.parametrize(
    ('confidence', 'message'),
    [
        (None, 'Boltz2 output does not contain confidence'),
        ([None], 'Boltz2 output does not contain confidence'),
        ([{}], "Boltz2 confidence dict does not contain 'ptm'"),
    ],
)
def test_boltz2_rejects_missing_ptm(monkeypatch, monomer: list[bg.Chain], confidence, message: str) -> None:
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(20, 80.0)],
        pae=[np.zeros((20, 20))],
        confidence=confidence,
    )

    def mock_load(self, config={}):
        self.model = FakeModel(output)

    monkeypatch.setattr(bg.oracles.folding.Boltz2, '_load', mock_load)
    oracle = bg.oracles.folding.Boltz2(backend='modal')

    with pytest.raises(ValueError, match=message):
        oracle._reduce_output(output, monomer)


@pytest.mark.parametrize(
    ('field_name', 'message'),
    [
        ('plddt', 'ESMFold output does not contain plddt'),
        ('pae', 'ESMFold output does not contain pae'),
        ('ptm', 'ESMFold output does not contain ptm'),
    ],
)
def test_esmfold_rejects_empty_required_fields(
    monkeypatch, monomer: list[bg.Chain], field_name: str, message: str
) -> None:
    fields = {
        'plddt': np.ones((1, len(monomer[0].residues), 37)),
        'pae': np.zeros((1, len(monomer[0].residues), len(monomer[0].residues))),
        'ptm': np.array([0.7]),
    }
    fields[field_name] = []
    output = SimpleNamespace(atom_array=[fake_atom_array(monomer)], **fields)

    def mock_load(self, config={}):
        pass

    monkeypatch.setattr(bg.oracles.folding.ESMFold, '_load', mock_load)
    oracle = bg.oracles.folding.ESMFold(backend='modal')

    with pytest.raises(ValueError, match=message):
        oracle._reduce_output(output, monomer)
