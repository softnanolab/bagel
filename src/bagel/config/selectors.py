"""Residue selector compiler utilities for YAML configs."""

from __future__ import annotations

from typing import Any

from bagel.chain import Chain, Residue


class SelectorCompilationError(ValueError):
    """Raised when residue selector compilation fails."""


def _is_plain_int(value: Any) -> bool:
    """Return True only for int values that are not booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _select_from_dict(selector: dict[str, Any], chain_registry: dict[str, Chain], path: str) -> list[Residue]:
    chain_name = selector.get('chain')
    if not isinstance(chain_name, str) or not chain_name:
        raise SelectorCompilationError(f"{path}: selector must include non-empty 'chain'")

    if chain_name not in chain_registry:
        known = ', '.join(chain_registry.keys())
        raise SelectorCompilationError(f"{path}: unknown chain '{chain_name}'. Known chains: {known}")

    chain = chain_registry[chain_name]
    chain_len = chain.length

    has_all = selector.get('all', False) is True
    has_indices = 'indices' in selector
    has_range = 'start' in selector or 'end' in selector

    if sum([has_all, has_indices, has_range]) == 0:
        raise SelectorCompilationError(
            f"{path}: selector must define one of: all=true, indices=[...], or start/end"
        )
    if sum([has_all, has_indices, has_range]) > 1:
        raise SelectorCompilationError(
            f"{path}: selector cannot mix all/indices/start-end in the same selector"
        )

    if has_all:
        return list(chain.residues)

    if has_indices:
        indices = selector.get('indices')
        if not isinstance(indices, list) or not all(_is_plain_int(i) for i in indices):
            raise SelectorCompilationError(f"{path}: 'indices' must be a list of integers")
        residues: list[Residue] = []
        for idx in indices:
            if idx < 0 or idx >= chain_len:
                raise SelectorCompilationError(
                    f"{path}: residue index {idx} out of bounds for chain '{chain_name}' length {chain_len}"
                )
            residues.append(chain.residues[idx])
        return residues

    start = selector.get('start')
    end = selector.get('end')

    if start is None:
        start = 0
    if end is None:
        end = chain_len - 1

    if not _is_plain_int(start) or not _is_plain_int(end):
        raise SelectorCompilationError(f"{path}: start/end must be integers")
    if start < 0 or end < 0:
        raise SelectorCompilationError(f"{path}: start/end must be >= 0")
    if start > end:
        raise SelectorCompilationError(f"{path}: start ({start}) cannot be greater than end ({end})")
    if end >= chain_len:
        raise SelectorCompilationError(
            f"{path}: end index {end} out of bounds for chain '{chain_name}' length {chain_len}"
        )

    return list(chain.residues[start : end + 1])


def compile_residue_selectors(value: Any, chain_registry: dict[str, Chain], path: str = 'residues') -> Any:
    """
    Compile YAML residue selectors to Residue objects.

    Rules:
    - dict selector -> list[Residue]
    - list of dict selectors -> merged list[Residue]
    - list with nested structures -> recursive list[...] preserving group structure
    """
    if isinstance(value, dict):
        return _select_from_dict(value, chain_registry, path)

    if isinstance(value, list):
        if all(isinstance(item, dict) for item in value):
            merged: list[Residue] = []
            for i, selector in enumerate(value):
                merged.extend(_select_from_dict(selector, chain_registry, f'{path}[{i}]'))
            return merged

        return [
            compile_residue_selectors(item, chain_registry, path=f'{path}[{idx}]') for idx, item in enumerate(value)
        ]

    raise SelectorCompilationError(f"{path}: expected selector dict or list, got {type(value).__name__}")
