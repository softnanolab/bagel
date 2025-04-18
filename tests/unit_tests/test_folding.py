import numpy as np
import desprot as dp
from biotite.structure import AtomArray  # type: ignore


def test_monomer_folding(folder, monomer):
    atoms, metrics = folder.fold(monomer)

    assert atoms is not None, 'Output should not be None'
    assert isinstance(atoms, AtomArray), 'Output should be an AtomArray'
    assert metrics is not None, 'Output should not be None'
    assert isinstance(metrics, dp.folding.ESMFoldingMetrics), 'Output should be an dp.folding.ESMFoldingMetrics'


def test_dimer_folding(folder, dimer):
    atoms, metrics = folder.fold(dimer)

    assert atoms is not None, 'Output should not be None'
    assert isinstance(atoms, AtomArray), 'Output should be an AtomArray'
    assert metrics is not None, 'Output should not be None'
    assert isinstance(metrics, dp.folding.ESMFoldingMetrics), 'Output should be an dp.folding.ESMFoldingMetrics'


def test_trimer_folding(folder, trimer):
    atoms, metrics = folder.fold(trimer)

    assert atoms is not None, 'Output should not be None'
    assert isinstance(atoms, AtomArray), 'Output should be an AtomArray'
    assert metrics is not None, 'Output should not be None'
    assert isinstance(metrics, dp.folding.ESMFoldingMetrics), 'Output should be an dp.folding.ESMFoldingMetrics'

    # Test custom chain IDs
    assert np.all(np.unique(atoms.chain_id) == ['C-A', 'C-B', 'C-C']), 'Chain IDs should be C-A, C-B, C-C'
