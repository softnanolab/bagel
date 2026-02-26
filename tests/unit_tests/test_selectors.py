import pytest

import bagel as bg
from bagel.config.selectors import SelectorCompilationError, compile_residue_selectors


def _chain_registry() -> dict[str, bg.Chain]:
    residues = [bg.Residue(name='A', chain_ID='A', index=i, mutable=True) for i in range(5)]
    return {'design': bg.Chain(residues=residues)}


def test_indices_rejects_boolean_values() -> None:
    with pytest.raises(SelectorCompilationError, match='list of integers'):
        compile_residue_selectors({'chain': 'design', 'indices': [0, True, 2]}, _chain_registry())


def test_start_end_reject_booleans() -> None:
    with pytest.raises(SelectorCompilationError, match='start/end must be integers'):
        compile_residue_selectors({'chain': 'design', 'start': True, 'end': 3}, _chain_registry())

    with pytest.raises(SelectorCompilationError, match='start/end must be integers'):
        compile_residue_selectors({'chain': 'design', 'start': 0, 'end': False}, _chain_registry())
