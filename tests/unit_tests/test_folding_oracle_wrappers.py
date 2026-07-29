from types import SimpleNamespace

import numpy as np
import pytest
import bagel as bg
from biotite.structure import Atom, array
from boileroom.base import PredictionMetadata
from boileroom.models.boltz.types import Boltz2Output
from boileroom.models.esm.types import ESMFoldOutput


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


def prediction_metadata(num_residues: int) -> PredictionMetadata:
    return PredictionMetadata(model_name='test', model_version='0.4.1', sequence_lengths=[num_residues])


def test_esmfold_reduces_boilerroom_041_output_contract(monkeypatch, monomer: list[bg.Chain]) -> None:
    num_residues = len(monomer[0].residues)
    output = ESMFoldOutput(
        metadata=prediction_metadata(num_residues),
        atom_array=[fake_atom_array(monomer)],
        plddt=np.full((1, num_residues, 37), 80.0),
        pae=np.zeros((1, num_residues, num_residues)),
        ptm=np.array([0.7]),
    )
    monkeypatch.setattr(bg.oracles.folding.ESMFold, '_load', lambda self, config=None: None)

    result = bg.oracles.folding.ESMFold()._reduce_output(output, monomer)

    assert np.allclose(result.local_plddt, 0.8)
    assert result.ptm[0, 0] == pytest.approx(0.7)


def test_boltz2_reduces_boilerroom_041_output_contract(monkeypatch, monomer: list[bg.Chain]) -> None:
    num_residues = len(monomer[0].residues)
    output = Boltz2Output(
        metadata=prediction_metadata(num_residues),
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(num_residues, 80.0)],
        pae=[np.zeros((num_residues, num_residues))],
        confidence=[{'ptm': 0.7}],
    )
    monkeypatch.setattr(bg.oracles.folding.Boltz2, '_load', lambda self, config=None: None)

    result = bg.oracles.folding.Boltz2()._reduce_output(output, monomer)

    assert output.confidence is None
    assert np.allclose(result.local_plddt, 0.8)
    assert result.ptm[0, 0] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ('oracle_class', 'result_class', 'output_fields', 'expected_fields', 'expected_ptm'),
    [
        (
            bg.oracles.folding.Boltz2,
            bg.oracles.folding.Boltz2Result,
            {'plddt': [np.full(20, 0.8)], 'pae': [np.zeros((20, 20))], 'ptm': [np.array([0.7])]},
            ['plddt', 'pae', 'ptm'],
            0.7,
        ),
        (
            bg.oracles.folding.Chai1,
            bg.oracles.folding.Chai1Result,
            {'plddt': [np.full(20, 0.8)], 'pae': [np.zeros((20, 20))], 'ptm': [np.array([0.7])]},
            ['plddt', 'pae', 'ptm'],
            0.7,
        ),
        (
            bg.oracles.folding.ESMFold,
            bg.oracles.folding.ESMFoldResult,
            {
                'plddt': [np.full(20, 0.8)],
                'pae': np.zeros((1, 20, 20)),
                'ptm': [np.array([0.7])],
            },
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

    def mock_load(self, config=None):
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


def test_esmfold2_wrapper_uses_typed_input_and_requests_confidence(monkeypatch, monomer: list[bg.Chain]) -> None:
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(20, 0.8)],
        pae=[np.zeros((20, 20))],
        ptm=[np.array([0.7])],
    )
    fake_model = FakeModel(output)

    def mock_load(self, config=None):
        self.model = fake_model

    monkeypatch.setattr(bg.oracles.folding.ESMFold2, '_load', mock_load)

    result = bg.oracles.folding.ESMFold2(backend='modal').fold(monomer)
    fold_input, options = fake_model.calls[0]

    assert isinstance(result, bg.oracles.folding.ESMFold2Result)
    assert [protein.id for protein in fold_input.sequences] == [monomer[0].chain_ID]
    assert [protein.sequence for protein in fold_input.sequences] == [monomer[0].sequence]
    assert options == {'include_fields': ['plddt', 'pae', 'ptm']}


def test_chai1_wrapper_rejects_missing_ptm(monkeypatch, monomer: list[bg.Chain]) -> None:
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(20, 0.8)],
        pae=[np.zeros((20, 20))],
        ptm=None,
    )
    fake_model = FakeModel(output)

    def mock_load(self, config=None):
        self.model = fake_model

    monkeypatch.setattr(bg.oracles.folding.Chai1, '_load', mock_load)

    with pytest.raises(ValueError, match='Chai1 output does not contain ptm'):
        bg.oracles.folding.Chai1(backend='modal').fold(monomer)


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
        plddt=[np.full(20, 0.8)],
        pae=[np.zeros((20, 20))],
        ptm=ptm_input,
    )
    fake_model = FakeModel(output)

    def mock_load(self, config=None):
        self.model = fake_model

    monkeypatch.setattr(bg.oracles.folding.Chai1, '_load', mock_load)

    result = bg.oracles.folding.Chai1(backend='modal').fold(monomer)

    assert result.ptm.shape == (1, 1)
    assert result.ptm[0, 0] == pytest.approx(0.73)


@pytest.mark.parametrize(
    ('ptm', 'message'),
    [
        (None, 'Boltz2 output does not contain ptm'),
        ([None], 'Boltz2 output does not contain ptm'),
        ([], 'Boltz2 ptm must contain exactly one sample'),
    ],
)
def test_boltz2_rejects_missing_ptm(monkeypatch, monomer: list[bg.Chain], ptm, message: str) -> None:
    output = SimpleNamespace(
        atom_array=[fake_atom_array(monomer)],
        plddt=[np.full(20, 0.8)],
        pae=[np.zeros((20, 20))],
        ptm=ptm,
    )

    def mock_load(self, config=None):
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
        'plddt': [np.ones(len(monomer[0].residues))],
        'pae': np.zeros((1, len(monomer[0].residues), len(monomer[0].residues))),
        'ptm': [np.array([0.7])],
    }
    fields[field_name] = None
    output = SimpleNamespace(atom_array=[fake_atom_array(monomer)], **fields)

    def mock_load(self, config=None):
        pass

    monkeypatch.setattr(bg.oracles.folding.ESMFold, '_load', mock_load)
    oracle = bg.oracles.folding.ESMFold(backend='modal')

    with pytest.raises(ValueError, match=message):
        oracle._reduce_output(output, monomer)
