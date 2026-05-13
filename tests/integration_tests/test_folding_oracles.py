import numpy as np
import pytest
import bagel as bg
from biotite.structure import AtomArray


@pytest.mark.parametrize(
    ('oracle_fixture', 'result_class'),
    [
        ('esmfold', bg.oracles.folding.ESMFoldResult),
        ('boltz2', bg.oracles.folding.Boltz2Result),
        ('chai1', bg.oracles.folding.Chai1Result),
    ],
)
def test_folding_oracle_modal_smoke(
    request,
    oracle_fixture: str,
    result_class: type[bg.oracles.folding.FoldingResult],
    monomer: list[bg.Chain],
) -> None:
    oracle = request.getfixturevalue(oracle_fixture)
    result = oracle.fold(monomer)

    num_residues = sum(len(chain.residues) for chain in monomer)

    assert isinstance(result, result_class)
    assert isinstance(result.structure, AtomArray)
    assert len(result.structure) > 0
    assert isinstance(result.local_plddt, np.ndarray)
    assert result.local_plddt.shape == (1, num_residues)
    assert isinstance(result.pae, np.ndarray)
    assert result.pae.shape == (1, num_residues, num_residues)
    assert isinstance(result.ptm, np.ndarray)

    unique_residue_ids = np.unique(result.structure.res_id)
    assert np.array_equal(unique_residue_ids, np.arange(num_residues))
    assert np.array_equal(np.unique(result.structure.chain_id), [monomer[0].chain_ID])
