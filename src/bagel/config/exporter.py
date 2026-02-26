"""Run configuration export utilities for YAML and runtime-driven BAGEL runs."""

from __future__ import annotations

import inspect
import json
import pathlib as pl
from typing import Any

import numpy as np
import yaml

from bagel.callbacks import Callback
from bagel.energies import EnergyTerm
from bagel.minimizer import Minimizer
from bagel.oracles.base import Oracle
from bagel.system import System


class ConfigExportError(ValueError):
    """Raised when exporting run configuration fails."""


def _class_path(obj: Any) -> str:
    return f'{obj.__class__.__module__}:{obj.__class__.__name__}'


def _callable_ref(fn: Any) -> str:
    return f'{fn.__module__}:{getattr(fn, "__qualname__", getattr(fn, "__name__", "callable"))}'


def _safe_value(value: Any, ref_by_id: dict[int, str]) -> Any:
    value_id = id(value)
    if value_id in ref_by_id:
        return ref_by_id[value_id]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, pl.Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.size <= 128:
            return value.tolist()
        return {'__ndarray__': {'shape': list(value.shape), 'dtype': str(value.dtype)}}

    if isinstance(value, tuple):
        return [_safe_value(item, ref_by_id) for item in value]

    if isinstance(value, list):
        return [_safe_value(item, ref_by_id) for item in value]

    if isinstance(value, dict):
        return {str(k): _safe_value(v, ref_by_id) for k, v in value.items()}

    if callable(value):
        return {'__callable__': _callable_ref(value)}

    return {'__python_object__': _class_path(value), 'repr': repr(value)}


def _extract_constructor_params(obj: Any, ref_by_id: dict[int, str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    signature = inspect.signature(obj.__class__.__init__)

    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if name == 'callbacks' and hasattr(obj, 'callback_manager'):
            params[name] = [_safe_value(cb, ref_by_id) for cb in obj.callback_manager.callbacks]
            continue

        if hasattr(obj, name):
            params[name] = _safe_value(getattr(obj, name), ref_by_id)

    return params


def _residue_groups_to_yaml(term: EnergyTerm, chain_id_to_key: dict[str, str]) -> Any:
    if not hasattr(term, 'residue_groups') or not term.residue_groups:
        return None

    groups_yaml: list[list[dict[str, Any]]] = []
    for group in term.residue_groups:
        chain_ids = np.array(group[0])
        res_indices = np.array(group[1])
        group_yaml: list[dict[str, Any]] = []
        for chain_id in np.unique(chain_ids):
            chain_mask = chain_ids == chain_id
            indices = sorted(int(i) for i in np.unique(res_indices[chain_mask]).tolist())
            group_yaml.append({'chain': chain_id_to_key.get(str(chain_id), str(chain_id)), 'indices': indices})
        groups_yaml.append(group_yaml)

    if len(groups_yaml) == 1:
        return groups_yaml[0]
    return groups_yaml


def make_runtime_snapshot(system: System, minimizer: Minimizer) -> dict[str, Any]:
    """Best-effort runtime snapshot for script-driven runs without YAML input."""
    chain_by_id: dict[int, str] = {}
    chains_section: dict[str, Any] = {}

    chain_id_counts: dict[str, int] = {}
    chain_id_to_key: dict[str, str] = {}
    unique_chains: list[Any] = []

    for state in system.states:
        for chain in state.chains:
            if id(chain) not in chain_by_id:
                unique_chains.append(chain)

    for chain in unique_chains:
        base_key = chain.chain_ID
        count = chain_id_counts.get(base_key, 0) + 1
        chain_id_counts[base_key] = count
        chain_key = base_key if count == 1 else f'{base_key}_{count}'
        chain_by_id[id(chain)] = chain_key
        chain_id_to_key.setdefault(chain.chain_ID, chain_key)
        chains_section[chain_key] = {
            'sequence': chain.sequence,
            'mutable': [bool(res.mutable) for res in chain.residues],
            'chain_id': chain.chain_ID,
        }

    oracle_by_id: dict[int, str] = {}
    oracles_section: dict[str, Any] = {}
    unique_oracles: list[Oracle] = []

    for state in system.states:
        for term in state.energy_terms:
            if id(term.oracle) not in oracle_by_id:
                unique_oracles.append(term.oracle)

    for idx, oracle in enumerate(unique_oracles, start=1):
        oracle_key = f'oracle_{idx}'
        oracle_by_id[id(oracle)] = oracle_key
        oracles_section[oracle_key] = {
            'type': _class_path(oracle),
            'params': _extract_constructor_params(oracle, ref_by_id={}),
        }

    mutator_key = 'mutator_main'
    mutator = minimizer.mutator
    mutators_section = {
        mutator_key: {'type': _class_path(mutator), 'params': _extract_constructor_params(mutator, ref_by_id={})}
    }

    callbacks_section: dict[str, Any] = {}
    callbacks = list(minimizer.callback_manager.callbacks)
    for idx, callback in enumerate(callbacks, start=1):
        callback_key = f'callback_{idx}'
        callbacks_section[callback_key] = {
            'type': _class_path(callback),
            'params': _extract_constructor_params(callback, ref_by_id={}),
        }

    ref_by_id: dict[int, str] = {}
    for chain in unique_chains:
        ref_by_id[id(chain)] = f"@chains.{chain_by_id[id(chain)]}"
    for idx, oracle in enumerate(unique_oracles, start=1):
        ref_by_id[id(oracle)] = f'@oracles.oracle_{idx}'
    ref_by_id[id(mutator)] = f'@mutators.{mutator_key}'
    for idx, callback in enumerate(callbacks, start=1):
        ref_by_id[id(callback)] = f'@callbacks.callback_{idx}'

    # Rebuild callback params now that refs exist
    for idx, callback in enumerate(callbacks, start=1):
        callback_key = f'callback_{idx}'
        callbacks_section[callback_key]['params'] = _extract_constructor_params(callback, ref_by_id=ref_by_id)

    states_section: list[dict[str, Any]] = []
    for state in system.states:
        energies_yaml: list[dict[str, Any]] = []
        for term in state.energy_terms:
            params = _extract_constructor_params(term, ref_by_id=ref_by_id)
            residues_yaml = _residue_groups_to_yaml(term, chain_id_to_key)
            energies_yaml.append(
                {
                    'type': _class_path(term),
                    'oracle': ref_by_id.get(id(term.oracle), f'@oracles.{oracle_by_id[id(term.oracle)]}'),
                    'residues': residues_yaml,
                    'params': params,
                }
            )

        states_section.append(
            {
                'name': state.name,
                'chains': [chain_by_id[id(chain)] for chain in state.chains],
                'energies': energies_yaml,
            }
        )

    minimizer_params = _extract_constructor_params(minimizer, ref_by_id=ref_by_id)
    minimizer_params.pop('run_config', None)

    snapshot = {
        'schema_version': 1,
        'run': {
            'experiment_name': minimizer.experiment_name,
            'log_path': str(minimizer.log_path),
        },
        'chains': chains_section,
        'oracles': oracles_section,
        'mutators': mutators_section,
        'states': states_section,
        'callbacks': callbacks_section,
        'minimizer': {
            'type': _class_path(minimizer),
            'params': minimizer_params,
        },
    }
    return snapshot


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pl.Path):
        return str(value)
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if value.size <= 128:
            return value.tolist()
        return {'__ndarray__': {'shape': list(value.shape), 'dtype': str(value.dtype)}}
    if callable(value):
        return {'__callable__': _callable_ref(value)}
    return {'__python_object__': _class_path(value), 'repr': repr(value)}


def export_run_config(
    log_path: pl.Path,
    run_config: dict[str, Any] | None = None,
    system: System | None = None,
    minimizer: Minimizer | None = None,
) -> tuple[pl.Path, pl.Path]:
    """Export run config snapshots to minimizer log directory."""
    log_path.mkdir(parents=True, exist_ok=True)

    if run_config is None:
        if system is None or minimizer is None:
            raise ConfigExportError('system and minimizer are required when run_config is None')
        snapshot = make_runtime_snapshot(system=system, minimizer=minimizer)
        raw_data = snapshot
        resolved_data = snapshot
    else:
        raw_data = run_config.get('raw', run_config)
        resolved_data = run_config.get('resolved', run_config)

    yaml_path = log_path / 'run_config.yaml'
    json_path = log_path / 'run_config.resolved.json'

    with open(yaml_path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(_to_json_safe(raw_data), handle, sort_keys=False, allow_unicode=False)

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(_to_json_safe(resolved_data), handle, indent=2, sort_keys=True)
        handle.write('\n')

    return yaml_path, json_path
