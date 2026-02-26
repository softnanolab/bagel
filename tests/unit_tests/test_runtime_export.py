import pathlib as pl
from unittest.mock import Mock

import bagel as bg


def test_minimizer_exports_runtime_config_without_yaml(tmp_path: pl.Path, fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    residues = [bg.Residue(name='A', chain_ID='A', index=i, mutable=True) for i in range(4)]
    state = bg.State(
        name='state_A',
        chains=[bg.Chain(residues=residues)],
        energy_terms=[bg.energies.PTMEnergy(oracle=fake_esmfold, weight=1.0)],
    )
    system = bg.System(states=[state])

    mock_mutator = Mock()
    mock_mutator.one_step.return_value = (system.__copy__(), None)

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=1.0,
        n_steps=1,
        experiment_name='runtime_export_test',
        log_path=tmp_path,
    )

    minimizer.minimize_system(system)

    run_dir = tmp_path / 'runtime_export_test'
    assert (run_dir / 'run_config.yaml').exists()
    assert (run_dir / 'run_config.resolved.json').exists()
