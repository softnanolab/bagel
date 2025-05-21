import numpy as np
import bagel as bg
from biotite.structure import AtomArray  # type: ignore


def test_monomer_folding(esmfold, monomer):
    results = esmfold.fold(monomer)

    assert results is not None, 'Output should not be None'
    assert isinstance(results, bg.oracles.folding.ESMFoldingResults), (
        'Output should be an bg.oracles.folding.ESMFoldingResults'
    )
    assert results.structure is not None, 'Output should not be None'
    assert isinstance(results.structure, AtomArray), 'Output should be an AtomArray'
    assert results.local_plddt is not None, 'Output should not be None'
    assert isinstance(results.local_plddt, np.ndarray), 'Output should be an np.ndarray'
    assert results.ptm is not None, 'Output should not be None'
    assert isinstance(results.ptm, np.ndarray), 'Output should be an np.ndarray'


def test_dimer_folding(esmfold, dimer):
    results = esmfold.fold(dimer)

    assert results is not None, 'Output should not be None'
    assert isinstance(results, bg.oracles.folding.ESMFoldingResults), (
        'Output should be an bg.oracles.folding.ESMFoldingResults'
    )
    assert results.structure is not None, 'Output should not be None'
    assert isinstance(results.structure, AtomArray), 'Output should be an AtomArray'
    assert results.local_plddt is not None, 'Output should not be None'
    assert isinstance(results.local_plddt, np.ndarray), 'Output should be an np.ndarray'
    assert results.ptm is not None, 'Output should not be None'
    assert isinstance(results.ptm, np.ndarray), 'Output should be an np.ndarray'


def test_trimer_folding(esmfold, trimer):
    results = esmfold.fold(trimer)

    assert results is not None, 'Output should not be None'
    assert isinstance(results, bg.oracles.folding.ESMFoldingResults), (
        'Output should be an bg.oracles.folding.ESMFoldingResults'
    )
    assert results.structure is not None, 'Output should not be None'
    assert isinstance(results.structure, AtomArray), 'Output should be an AtomArray'
    assert results.local_plddt is not None, 'Output should not be None'
    assert isinstance(results.local_plddt, np.ndarray), 'Output should be an np.ndarray'
    assert results.ptm is not None, 'Output should not be None'
    assert isinstance(results.ptm, np.ndarray), 'Output should be an np.ndarray'

    # Test custom chain IDs
    assert np.all(np.unique(results.structure.chain_id) == ['C-A', 'C-B', 'C-C']), 'Chain IDs should be C-A, C-B, C-C'
