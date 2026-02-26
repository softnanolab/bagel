import pathlib as pl

import pytest

from bagel.config.compiler import ConfigCompilationError, compile_loaded_config, run_compiled, validate_loaded_config
from bagel.config.loader import load_config


FIXTURES_DIR = pl.Path(__file__).resolve().parents[1] / 'fixtures'


def test_validate_fails_on_unknown_builtin_param(tmp_path: pl.Path) -> None:
    plugins_file = (FIXTURES_DIR / 'plugins' / 'custom_components.py').as_posix()
    config_path = tmp_path / 'bad_builtin.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'schema_version: 1',
                'run: {}',
                'chains:',
                '  design:',
                '    sequence: ACDE',
                '    mutable: [true, true, true, true]',
                '    chain_id: D',
                'oracles:',
                '  oracle_main:',
                f'    type: {plugins_file}:DummyOracle',
                '    params: {}',
                'mutators:',
                '  mut_main:',
                '    type: canonical',
                '    params:',
                '      n_mutations: 1',
                '      typo_param: 123',
                'states:',
                '  - name: state_A',
                '    chains: [design]',
                '    energies:',
                f'      - type: {plugins_file}:DummyEnergy',
                '        oracle: "@oracles.oracle_main"',
                '        params: {}',
                'minimizer:',
                f'  type: {plugins_file}:DummyMinimizer',
                '  params:',
                '    mutator: "@mutators.mut_main"',
            ]
        )
    )

    loaded = load_config(config_path)
    with pytest.raises(ConfigCompilationError, match='unknown params'):
        validate_loaded_config(loaded)


def test_compile_and_run_custom_stack_exports_config(tmp_path: pl.Path) -> None:
    config_path = FIXTURES_DIR / 'configs' / 'custom_stack.yaml'
    loaded = load_config(config_path)

    # Redirect logs to temporary directory so test remains isolated.
    loaded.raw['run']['log_path'] = str(tmp_path)
    loaded.parsed.run.log_path = str(tmp_path)

    compiled = compile_loaded_config(loaded)
    result = run_compiled(compiled)

    assert result is not None

    run_dir = compiled.minimizer.log_path
    assert (run_dir / 'run_config.yaml').exists()
    assert (run_dir / 'run_config.resolved.json').exists()


def test_nested_callable_marker_validates(tmp_path: pl.Path) -> None:
    plugins_file = (FIXTURES_DIR / 'plugins' / 'custom_components.py').as_posix()
    config_path = tmp_path / 'callable.yaml'
    config_path.write_text(
        f"""
schema_version: 1
run: {{}}
chains:
  design:
    sequence: ACDE
    mutable: [true, true, true, true]
    chain_id: D
oracles:
  oracle_main:
    type: {plugins_file}:DummyOracle
    params: {{}}
mutators:
  mut_main:
    type: {plugins_file}:DummyMutator
    params:
      n_mutations: 1
states:
  - name: state_A
    chains: [design]
    energies:
      - type: {plugins_file}:DummyEnergy
        oracle: "@oracles.oracle_main"
        params: {{}}
minimizer:
  type: {plugins_file}:DummyMinimizer
  params:
    mutator: "@mutators.mut_main"
    nested:
      level1:
        fn:
          __callable__: {plugins_file}:square_distance
"""
    )

    loaded = load_config(config_path)
    validate_loaded_config(loaded)
