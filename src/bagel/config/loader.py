"""YAML loading and schema validation for BAGEL run configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pathlib as pl

import yaml

from .schema_v1 import RunConfigV1

CURRENT_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = {1}


class ConfigLoadError(ValueError):
    """Raised when loading or validating YAML config fails."""


@dataclass(frozen=True)
class LoadedConfig:
    path: pl.Path
    base_dir: pl.Path
    raw: dict[str, Any]
    parsed: RunConfigV1


def _read_yaml(path: pl.Path) -> dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            content = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigLoadError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML in '{path}': {exc}") from exc

    if not isinstance(content, dict):
        raise ConfigLoadError(f"Config '{path}' must deserialize to a top-level mapping")

    return content


def load_config(path: str | pl.Path) -> LoadedConfig:
    config_path = pl.Path(path).expanduser().resolve()
    raw = _read_yaml(config_path)

    schema_version = raw.get('schema_version')
    if not isinstance(schema_version, int):
        raise ConfigLoadError("Field 'schema_version' is required and must be an integer")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ', '.join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ConfigLoadError(
            f"Unsupported schema_version={schema_version}. Supported versions: {supported}."
        )

    try:
        parsed = RunConfigV1.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise ConfigLoadError(f"Config validation failed for '{config_path}': {exc}") from exc

    return LoadedConfig(path=config_path, base_dir=config_path.parent, raw=raw, parsed=parsed)


def schema_json(version: int) -> dict[str, Any]:
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ', '.join(str(v) for v in sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ConfigLoadError(f"Unsupported schema version {version}. Supported versions: {supported}")

    if version == 1:
        return RunConfigV1.model_json_schema()

    raise ConfigLoadError(f'No schema available for version {version}')
