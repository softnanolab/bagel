import json
import pathlib as pl

import pytest

from bagel.config.loader import ConfigLoadError, load_config, schema_json


def test_schema_version_required(tmp_path: pl.Path) -> None:
    config_path = tmp_path / 'missing_schema.yaml'
    config_path.write_text('run: {}\nchains: {}\noracles: {}\nmutators: {}\nstates: []\nminimizer: {type: x}\n')

    with pytest.raises(ConfigLoadError, match='schema_version'):
        load_config(config_path)


def test_schema_version_unsupported(tmp_path: pl.Path) -> None:
    config_path = tmp_path / 'unsupported_schema.yaml'
    config_path.write_text('schema_version: 999\n')

    with pytest.raises(ConfigLoadError, match='Unsupported schema_version'):
        load_config(config_path)


def test_schema_export_contains_expected_fields() -> None:
    schema = schema_json(1)
    dumped = json.dumps(schema)
    assert 'schema_version' in dumped
    assert 'minimizer' in dumped
    assert 'states' in dumped
