"""Compile BAGEL YAML configs into runtime objects and execute runs."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import pathlib as pl
import random
from typing import Any

import numpy as np

import bagel as bg
from bagel.callbacks import Callback, WandBLogger
from bagel.energies import EnergyTerm
from bagel.minimizer import Minimizer
from bagel.mutation import MutationProtocol
from bagel.oracles.base import Oracle

from .aliases import is_builtin_class_path, resolve_alias
from .imports import ImportResolutionError, resolve_callable, resolve_class
from .loader import LoadedConfig, load_config
from .selectors import SelectorCompilationError, compile_residue_selectors


class ConfigCompilationError(ValueError):
    """Raised when config compilation or static validation fails."""


@dataclass(frozen=True)
class CompiledRun:
    """Runtime objects produced from YAML config compilation."""

    config_path: pl.Path
    base_dir: pl.Path
    raw_config: dict[str, Any]
    resolved_config: dict[str, Any]
    seed: int | None
    system: bg.System
    minimizer: Minimizer


def _coerce_explicit_class_path(type_name: str) -> str:
    """Accept module:Class and module.Class forms (for explicit paths)."""
    if ':' in type_name:
        return type_name

    # Allow dotted class path form as a convenience for explicit paths
    if '.' in type_name and not type_name.endswith('.py'):
        module, attr = type_name.rsplit('.', 1)
        if module and attr:
            return f'{module}:{attr}'

    raise ConfigCompilationError(
        f"Unknown component type '{type_name}'. Use a built-in alias or '<module_or_file>:<ClassName>'."
    )


def _resolve_component_class(section: str, type_name: str, base_dir: pl.Path) -> tuple[type[Any], str, bool]:
    alias_class_path = resolve_alias(section, type_name)
    class_path = alias_class_path if alias_class_path is not None else _coerce_explicit_class_path(type_name)

    try:
        cls = resolve_class(class_path, base_dir)
    except ImportResolutionError as exc:
        raise ConfigCompilationError(f"Failed to resolve type '{type_name}' in section '{section}': {exc}") from exc

    builtin = alias_class_path is not None or is_builtin_class_path(section, class_path)
    return cls, class_path, builtin


def _ensure_subclass(cls: type[Any], expected: type[Any], context: str) -> None:
    if not issubclass(cls, expected):
        raise ConfigCompilationError(
            f"{context}: resolved class '{cls.__module__}:{cls.__name__}' must subclass {expected.__name__}"
        )


def _strict_validate_kwargs(cls: type[Any], params: dict[str, Any], context: str) -> None:
    sig = inspect.signature(cls.__init__)
    allowed: set[str] = set()

    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        allowed.add(name)

    unknown = sorted(set(params.keys()) - allowed)
    if unknown:
        raise ConfigCompilationError(
            f"{context}: unknown params for built-in '{cls.__module__}:{cls.__name__}': {unknown}. "
            f'Allowed params: {sorted(allowed)}'
        )


def _parse_ref(ref: str, context: str) -> tuple[str, str]:
    if not ref.startswith('@'):
        raise ConfigCompilationError(f"{context}: reference must start with '@', got '{ref}'")

    body = ref[1:]
    if '.' not in body:
        raise ConfigCompilationError(f"{context}: invalid reference '{ref}', expected '@section.id'")

    section, identifier = body.split('.', 1)
    if not section or not identifier:
        raise ConfigCompilationError(f"{context}: invalid reference '{ref}', expected '@section.id'")

    return section, identifier


def _validate_ref_exists(ref: str, known_ids: dict[str, set[str]], context: str) -> None:
    section, identifier = _parse_ref(ref, context)
    if identifier not in known_ids.get(section, set()):
        known = sorted(known_ids.get(section, set()))
        raise ConfigCompilationError(f"{context}: unknown reference '{ref}'. Known '{section}' ids: {known}")


def _resolve_ref_object(ref: str, object_maps: dict[str, dict[str, Any]], context: str) -> Any:
    section, identifier = _parse_ref(ref, context)
    if identifier not in object_maps.get(section, {}):
        known = sorted(object_maps.get(section, {}).keys())
        raise ConfigCompilationError(f"{context}: unknown reference '{ref}'. Known '{section}' ids: {known}")
    return object_maps[section][identifier]


def _resolve_value(value: Any, object_maps: dict[str, dict[str, Any]], base_dir: pl.Path, context: str) -> Any:
    if isinstance(value, dict):
        if '__callable__' in value:
            if set(value.keys()) != {'__callable__'}:
                raise ConfigCompilationError(
                    f"{context}: callable marker dict must only contain '__callable__' key"
                )
            target = value['__callable__']
            if not isinstance(target, str):
                raise ConfigCompilationError(f"{context}: '__callable__' value must be a string")
            try:
                return resolve_callable(target, base_dir)
            except ImportResolutionError as exc:
                raise ConfigCompilationError(f"{context}: failed to resolve callable '{target}': {exc}") from exc

        return {
            key: _resolve_value(sub_value, object_maps, base_dir, context=f'{context}.{key}')
            for key, sub_value in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_value(item, object_maps, base_dir, context=f'{context}[{idx}]') for idx, item in enumerate(value)
        ]

    if isinstance(value, str) and value.startswith('@'):
        return _resolve_ref_object(value, object_maps, context)

    return value


def _validate_value(value: Any, known_ids: dict[str, set[str]], base_dir: pl.Path, context: str) -> None:
    if isinstance(value, dict):
        if '__callable__' in value:
            if set(value.keys()) != {'__callable__'}:
                raise ConfigCompilationError(
                    f"{context}: callable marker dict must only contain '__callable__' key"
                )
            target = value['__callable__']
            if not isinstance(target, str):
                raise ConfigCompilationError(f"{context}: '__callable__' value must be a string")
            try:
                resolve_callable(target, base_dir)
            except ImportResolutionError as exc:
                raise ConfigCompilationError(f"{context}: failed to resolve callable '{target}': {exc}") from exc
            return

        for key, sub_value in value.items():
            _validate_value(sub_value, known_ids, base_dir, context=f'{context}.{key}')
        return

    if isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_value(item, known_ids, base_dir, context=f'{context}[{idx}]')
        return

    if isinstance(value, str) and value.startswith('@'):
        _validate_ref_exists(value, known_ids, context)


def _build_chain_registry(parsed: Any) -> dict[str, bg.Chain]:
    chain_registry: dict[str, bg.Chain] = {}

    for chain_name, spec in parsed.chains.items():
        residues = [
            bg.Residue(name=aa, chain_ID=spec.chain_id, index=i, mutable=bool(spec.mutable[i]))
            for i, aa in enumerate(spec.sequence)
        ]
        chain_registry[chain_name] = bg.Chain(residues=residues)

    return chain_registry


def _build_known_ids(parsed: Any) -> dict[str, set[str]]:
    return {
        'chains': set(parsed.chains.keys()),
        'oracles': set(parsed.oracles.keys()),
        'mutators': set(parsed.mutators.keys()),
        'callbacks': set(parsed.callbacks.keys()),
        'states': {state.name for state in parsed.states},
    }


def _build_resolved_config(parsed: Any, base_dir: pl.Path) -> dict[str, Any]:
    resolved = parsed.model_dump(mode='python')

    for oracle_name, spec in parsed.oracles.items():
        _, class_path, _ = _resolve_component_class('oracles', spec.type, base_dir)
        resolved['oracles'][oracle_name]['type'] = class_path

    for mutator_name, spec in parsed.mutators.items():
        _, class_path, _ = _resolve_component_class('mutators', spec.type, base_dir)
        resolved['mutators'][mutator_name]['type'] = class_path

    for callback_name, spec in parsed.callbacks.items():
        _, class_path, _ = _resolve_component_class('callbacks', spec.type, base_dir)
        resolved['callbacks'][callback_name]['type'] = class_path

    _, min_class_path, _ = _resolve_component_class('minimizers', parsed.minimizer.type, base_dir)
    resolved['minimizer']['type'] = min_class_path

    for state_idx, state in enumerate(parsed.states):
        for energy_idx, energy_spec in enumerate(state.energies):
            _, class_path, _ = _resolve_component_class('energies', energy_spec.type, base_dir)
            resolved['states'][state_idx]['energies'][energy_idx]['type'] = class_path

    return resolved


def validate_loaded_config(loaded: LoadedConfig) -> None:
    """Validate configuration statically (imports/refs/schema) without object instantiation."""
    parsed = loaded.parsed
    known_ids = _build_known_ids(parsed)
    chain_registry = _build_chain_registry(parsed)

    for oracle_name, oracle_spec in parsed.oracles.items():
        cls, _, builtin = _resolve_component_class('oracles', oracle_spec.type, loaded.base_dir)
        _ensure_subclass(cls, Oracle, context=f'oracles.{oracle_name}')
        _validate_value(oracle_spec.params, known_ids, loaded.base_dir, context=f'oracles.{oracle_name}.params')
        if builtin:
            _strict_validate_kwargs(cls, oracle_spec.params, context=f'oracles.{oracle_name}')

    for mutator_name, mutator_spec in parsed.mutators.items():
        cls, _, builtin = _resolve_component_class('mutators', mutator_spec.type, loaded.base_dir)
        _ensure_subclass(cls, MutationProtocol, context=f'mutators.{mutator_name}')
        _validate_value(mutator_spec.params, known_ids, loaded.base_dir, context=f'mutators.{mutator_name}.params')
        if builtin:
            _strict_validate_kwargs(cls, mutator_spec.params, context=f'mutators.{mutator_name}')

    for state_idx, state_spec in enumerate(parsed.states):
        for chain_name in state_spec.chains:
            if chain_name not in parsed.chains:
                known = sorted(parsed.chains.keys())
                raise ConfigCompilationError(
                    f"states[{state_idx}].chains references unknown chain '{chain_name}'. "
                    f'Known chains: {known}'
                )

        for energy_idx, energy_spec in enumerate(state_spec.energies):
            cls, _, builtin = _resolve_component_class('energies', energy_spec.type, loaded.base_dir)
            context = f'states[{state_idx}].energies[{energy_idx}]'
            _ensure_subclass(cls, EnergyTerm, context=context)

            _validate_ref_exists(energy_spec.oracle, known_ids, context=f'{context}.oracle')
            if energy_spec.residues is not None:
                try:
                    compile_residue_selectors(energy_spec.residues, chain_registry, path=f'{context}.residues')
                except SelectorCompilationError as exc:
                    raise ConfigCompilationError(str(exc)) from exc

            _validate_value(energy_spec.params, known_ids, loaded.base_dir, context=f'{context}.params')

            if builtin:
                strict_params = dict(energy_spec.params)
                strict_params['oracle'] = '@oracles.placeholder'
                if energy_spec.residues is not None:
                    strict_params['residues'] = []
                _strict_validate_kwargs(cls, strict_params, context=context)

    for callback_name, callback_spec in parsed.callbacks.items():
        cls, _, builtin = _resolve_component_class('callbacks', callback_spec.type, loaded.base_dir)
        _ensure_subclass(cls, Callback, context=f'callbacks.{callback_name}')
        _validate_value(callback_spec.params, known_ids, loaded.base_dir, context=f'callbacks.{callback_name}.params')
        if builtin:
            _strict_validate_kwargs(cls, callback_spec.params, context=f'callbacks.{callback_name}')

    min_cls, _, min_builtin = _resolve_component_class('minimizers', parsed.minimizer.type, loaded.base_dir)
    _ensure_subclass(min_cls, Minimizer, context='minimizer')
    _validate_value(parsed.minimizer.params, known_ids, loaded.base_dir, context='minimizer.params')
    if min_builtin:
        _strict_validate_kwargs(min_cls, parsed.minimizer.params, context='minimizer')

    if parsed.run.wandb.enabled and parsed.run.wandb.project is None and os.environ.get('WANDB_PROJECT') is None:
        raise ConfigCompilationError(
            'run.wandb.enabled=true requires run.wandb.project unless WANDB_PROJECT env var is set'
        )


def _ensure_wandb_callback(
    callbacks: list[Callback],
    wandb_enabled: bool,
    wandb_project: str | None,
    config_payload: dict[str, Any],
) -> list[Callback]:
    if not wandb_enabled:
        return callbacks

    any_wandb = any(isinstance(cb, WandBLogger) for cb in callbacks)
    if not any_wandb:
        project = wandb_project or os.environ.get('WANDB_PROJECT')
        if not project:
            raise ConfigCompilationError(
                'run.wandb.enabled=true requires run.wandb.project unless WANDB_PROJECT env var is set'
            )
        callbacks = [*callbacks, WandBLogger(project=project, config={'bagel_config': config_payload})]
    else:
        for cb in callbacks:
            if isinstance(cb, WandBLogger):
                cb.config.setdefault('bagel_config', config_payload)

    return callbacks


def compile_loaded_config(loaded: LoadedConfig) -> CompiledRun:
    """Compile loaded config into concrete BAGEL objects."""
    parsed = loaded.parsed
    validate_loaded_config(loaded)

    resolved_config = _build_resolved_config(parsed, loaded.base_dir)
    object_maps: dict[str, dict[str, Any]] = {
        'chains': {},
        'oracles': {},
        'mutators': {},
        'callbacks': {},
        'states': {},
    }

    chain_registry = _build_chain_registry(parsed)
    object_maps['chains'] = chain_registry

    oracle_registry: dict[str, Oracle] = {}
    for oracle_name, oracle_spec in parsed.oracles.items():
        cls, _, _ = _resolve_component_class('oracles', oracle_spec.type, loaded.base_dir)
        params = _resolve_value(oracle_spec.params, object_maps, loaded.base_dir, context=f'oracles.{oracle_name}.params')
        oracle_registry[oracle_name] = cls(**params)
    object_maps['oracles'] = oracle_registry

    mutator_registry: dict[str, MutationProtocol] = {}
    for mutator_name, mutator_spec in parsed.mutators.items():
        cls, _, _ = _resolve_component_class('mutators', mutator_spec.type, loaded.base_dir)
        params = _resolve_value(
            mutator_spec.params,
            object_maps,
            loaded.base_dir,
            context=f'mutators.{mutator_name}.params',
        )
        mutator_registry[mutator_name] = cls(**params)
    object_maps['mutators'] = mutator_registry

    states: list[bg.State] = []
    for state_idx, state_spec in enumerate(parsed.states):
        energies: list[EnergyTerm] = []
        for energy_idx, energy_spec in enumerate(state_spec.energies):
            context = f'states[{state_idx}].energies[{energy_idx}]'
            cls, _, _ = _resolve_component_class('energies', energy_spec.type, loaded.base_dir)
            params = _resolve_value(energy_spec.params, object_maps, loaded.base_dir, context=f'{context}.params')
            params['oracle'] = _resolve_ref_object(energy_spec.oracle, object_maps, context=f'{context}.oracle')
            if energy_spec.residues is not None:
                params['residues'] = compile_residue_selectors(
                    energy_spec.residues,
                    chain_registry,
                    path=f'{context}.residues',
                )

            energies.append(cls(**params))

        state_chains = [chain_registry[name] for name in state_spec.chains]
        state = bg.State(name=state_spec.name, chains=state_chains, energy_terms=energies)
        states.append(state)
        object_maps['states'][state_spec.name] = state

    callback_registry: dict[str, Callback] = {}
    for callback_name, callback_spec in parsed.callbacks.items():
        cls, _, _ = _resolve_component_class('callbacks', callback_spec.type, loaded.base_dir)
        params = _resolve_value(
            callback_spec.params,
            object_maps,
            loaded.base_dir,
            context=f'callbacks.{callback_name}.params',
        )
        callback_registry[callback_name] = cls(**params)
    object_maps['callbacks'] = callback_registry

    min_cls, _, _ = _resolve_component_class('minimizers', parsed.minimizer.type, loaded.base_dir)
    min_params = _resolve_value(parsed.minimizer.params, object_maps, loaded.base_dir, context='minimizer.params')

    if 'callbacks' not in min_params and callback_registry:
        min_params['callbacks'] = list(callback_registry.values())

    if parsed.run.experiment_name is not None:
        min_params.setdefault('experiment_name', parsed.run.experiment_name)
    if parsed.run.log_path is not None:
        min_params.setdefault('log_path', parsed.run.log_path)

    wandb_mode = parsed.run.wandb.config_mode
    if wandb_mode == 'raw':
        wandb_payload: dict[str, Any] = loaded.raw
    elif wandb_mode == 'both':
        wandb_payload = {'raw': loaded.raw, 'resolved': resolved_config}
    else:
        wandb_payload = resolved_config

    callbacks_list: list[Callback] = list(min_params.get('callbacks', []))
    callbacks_list = _ensure_wandb_callback(
        callbacks=callbacks_list,
        wandb_enabled=parsed.run.wandb.enabled,
        wandb_project=parsed.run.wandb.project,
        config_payload=wandb_payload,
    )
    if callbacks_list:
        min_params['callbacks'] = callbacks_list

    minimizer = min_cls(**min_params)
    minimizer.run_config = {
        'schema_version': 1,
        'raw': loaded.raw,
        'resolved': resolved_config,
        'config_path': str(loaded.path),
    }

    system = bg.System(states=states, name=parsed.run.experiment_name)

    return CompiledRun(
        config_path=loaded.path,
        base_dir=loaded.base_dir,
        raw_config=loaded.raw,
        resolved_config=resolved_config,
        seed=parsed.run.seed,
        system=system,
        minimizer=minimizer,
    )


def compile_config_file(path: str | pl.Path) -> CompiledRun:
    loaded = load_config(path)
    return compile_loaded_config(loaded)


def run_compiled(compiled: CompiledRun) -> bg.System:
    if compiled.seed is not None:
        random.seed(compiled.seed)
        np.random.seed(compiled.seed)

    return compiled.minimizer.minimize_system(compiled.system)


def run_from_config_file(path: str | pl.Path) -> bg.System:
    compiled = compile_config_file(path)
    return run_compiled(compiled)
