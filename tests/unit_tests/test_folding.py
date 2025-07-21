import numpy as np
import bagel as bg
from biotite.structure import AtomArray  # type: ignore


def test_monomer_folding(esmfold, monomer):
    results = esmfold.fold(monomer)

    assert results is not None, 'Output should not be None'
    assert isinstance(results, bg.oracles.folding.ESMFoldResult), 'Output should be an bg.oracles.folding.ESMFoldResult'
    assert results.structure is not None, 'Output should not be None'
    assert isinstance(results.structure, AtomArray), 'Output should be an AtomArray'
    assert results.local_plddt is not None, 'Output should not be None'
    assert isinstance(results.local_plddt, np.ndarray), 'Output should be an np.ndarray'
    assert results.ptm is not None, 'Output should not be None'
    assert isinstance(results.ptm, np.ndarray), 'Output should be an np.ndarray'

    # Our codebase assumes 0-indexing for residues, so we need to check that the residue IDs are 0-indexed
    num_residues = len(np.unique(results.structure.res_id))
    assert np.all(np.unique(results.structure.res_id) == np.arange(0, num_residues)), 'Residue IDs should be 0-indexed'


def test_dimer_folding(esmfold, dimer):
    results = esmfold.fold(dimer)

    assert results is not None, 'Output should not be None'
    assert isinstance(results, bg.oracles.folding.ESMFoldResult), 'Output should be an bg.oracles.folding.ESMFoldResult'
    assert results.structure is not None, 'Output should not be None'
    assert isinstance(results.structure, AtomArray), 'Output should be an AtomArray'
    assert results.local_plddt is not None, 'Output should not be None'
    assert isinstance(results.local_plddt, np.ndarray), 'Output should be an np.ndarray'
    assert results.ptm is not None, 'Output should not be None'
    assert isinstance(results.ptm, np.ndarray), 'Output should be an np.ndarray'


def test_trimer_folding(esmfold, trimer):
    results = esmfold.fold(trimer)

    assert results is not None, 'Output should not be None'
    assert isinstance(results, bg.oracles.folding.ESMFoldResult), 'Output should be an bg.oracles.folding.ESMFoldResult'
    assert results.structure is not None, 'Output should not be None'
    assert isinstance(results.structure, AtomArray), 'Output should be an AtomArray'
    assert results.local_plddt is not None, 'Output should not be None'
    assert isinstance(results.local_plddt, np.ndarray), 'Output should be an np.ndarray'
    assert results.ptm is not None, 'Output should not be None'
    assert isinstance(results.ptm, np.ndarray), 'Output should be an np.ndarray'

    # Test custom chain IDs
    assert np.all(np.unique(results.structure.chain_id) == ['C-A', 'C-B', 'C-C']), 'Chain IDs should be C-A, C-B, C-C'
