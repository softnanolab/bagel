import pathlib as pl

from bagel.cli import main


def _write_custom_config(path: pl.Path, log_path: pl.Path, plugin_path: str) -> None:
    path.write_text(
        f"""
schema_version: 1
run:
  experiment_name: cli_run
  log_path: {log_path.as_posix()}
chains:
  design:
    sequence: ACDE
    mutable: [true, true, true, true]
    chain_id: D
oracles:
  oracle_main:
    type: {plugin_path}:DummyOracle
    params: {{}}
mutators:
  mut_main:
    type: {plugin_path}:DummyMutator
    params:
      n_mutations: 1
states:
  - name: state_A
    chains: [design]
    energies:
      - type: {plugin_path}:DummyEnergy
        oracle: "@oracles.oracle_main"
        params: {{}}
minimizer:
  type: {plugin_path}:DummyMinimizer
  params:
    mutator: "@mutators.mut_main"
"""
    )


def test_cli_validate_schema_and_run(tmp_path: pl.Path) -> None:
    plugin_path = (
        pl.Path(__file__).resolve().parents[1] / 'fixtures' / 'plugins' / 'custom_components.py'
    ).as_posix()
    config_path = tmp_path / 'cli_config.yaml'
    log_path = tmp_path / 'logs'
    _write_custom_config(config_path, log_path=log_path, plugin_path=plugin_path)

    assert main(['validate', str(config_path)]) == 0
    assert main(['schema', '--version', '1']) == 0
    assert main(['run', str(config_path)]) == 0

    run_dir = log_path / 'cli_run'
    assert (run_dir / 'run_config.yaml').exists()
    assert (run_dir / 'run_config.resolved.json').exists()
