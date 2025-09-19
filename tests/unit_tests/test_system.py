import bagel as bg
import pandas as pd
import pathlib as pl
import shutil
from biotite.sequence.io.fasta import FastaFile
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np
from unittest.mock import Mock


def test_system_dump_config_file_is_correct(mixed_system: bg.System) -> None:
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mock_experiment = test_system_dump_config_file_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)

    mixed_system.dump_config(experiment_folder)

    file = mock_output_folder / mock_experiment / 'config.csv'
    assert file.exists(), 'config file not present in expected location'

    # Check the version.txt file
    version_file = mock_output_folder / mock_experiment / 'version.txt'
    assert version_file.exists(), 'version.txt file not present in expected location'
    with open(version_file, 'r') as vfile:
        version_line = vfile.readline().strip()
    assert version_line == str(bg.__version__), f'version.txt does not match version: {version_line}'

    # Now read the CSV
    config = pd.read_csv(file)

    assert all(config.columns == ['state', 'energy', 'weight']), 'config does not contain correct columns'
    correct_config = pd.DataFrame(
        {
            'state': ['small', 'small', 'mixed', 'mixed'],
            'energy': [
                'pTM',
                'selective_surface_area',
                'local_pLDDT',
                'cross_PAE',
                #'pTM',
                #'normalized_globular',
            ],
            'weight': [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert all(config == correct_config), 'data within config file is incorrect'  # checks non index cols

    shutil.rmtree(experiment_folder)


def test_system_dump_logs_folder_is_correct(mixed_system: bg.System) -> None:
    mock_step = 0
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mock_experiment = test_system_dump_logs_folder_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)

    # Ensure system energy is calculated
    assert mixed_system.total_energy is not None, 'System energy should be calculated before dumping logs'

    # Call dump_logs with the parameters aligned with the current implementation
    mixed_system.dump_logs(step=mock_step, path=experiment_folder, save_structure=True)

    sequences = {
        header: sequence
        for header, sequence in FastaFile.read_iter(file=experiment_folder / f'{mixed_system.states[1].name}.fasta')
    }
    assert sequences == {'0': 'G:VV:GVVV'}, 'incorrect sequence information saved'

    masks = {
        header: mask
        for header, mask in FastaFile.read_iter(file=experiment_folder / f'{mixed_system.states[1].name}.mask.fasta')
    }
    assert masks == {'0': 'M:MM:MIIM'}, 'incorrect mutability mask information saved'

    oracle = mixed_system.states[0].oracles_list[0]
    oracle_name = type(oracle).__name__
    assert oracle_name == 'ESMFold', 'incorrect oracle information saved'
    assert oracle == mixed_system.states[1].oracles_list[0], 'inconsistent oracles between states'

    structures = {
        'small': get_structure(
            CIFFile().read(file=experiment_folder / 'structures' / f'small_{oracle_name}_{mock_step}.cif')
        )[0],
        'mixed': get_structure(
            CIFFile().read(file=experiment_folder / 'structures' / f'mixed_{oracle_name}_{mock_step}.cif')
        )[0],
    }
    correct_structures = {
        'small': mixed_system.states[0]._oracles_result[oracle].structure,
        'mixed': mixed_system.states[1]._oracles_result[oracle].structure,
    }

    energies = pd.read_csv(experiment_folder / 'energies.csv')
    correct_energies = pd.DataFrame(
        {
            'step': [mock_step],
            'small:pTM': [-0.7],
            'small:selective_surface_area': [0.2],
            'mixed:local_pLDDT': [-0.4],
            'mixed:cross_PAE': [0.5],
            'small:state_energy': [-0.5],
            'mixed:state_energy': [0.1],
            'system_energy': [-0.4],
        }
    )

    assert structures == correct_structures, 'incorrect structure information saved'

    # Check column names match
    assert set(energies.columns) == set(correct_energies.columns), "DataFrame columns don't match"

    # Sort columns to ensure same order and compare values only
    assert np.array_equal(energies.sort_index(axis=1).values, correct_energies.sort_index(axis=1).values), (
        'incorrect energy information saved'
    )

    # load the pae and plddt files
    small_pae = np.loadtxt(experiment_folder / 'structures' / f'small_{oracle_name}_{mock_step}.pae')
    small_plddt = np.loadtxt(experiment_folder / 'structures' / f'small_{oracle_name}_{mock_step}.plddt')
    mixed_pae = np.loadtxt(experiment_folder / 'structures' / f'mixed_{oracle_name}_{mock_step}.pae')
    mixed_plddt = np.loadtxt(experiment_folder / 'structures' / f'mixed_{oracle_name}_{mock_step}.plddt')
    assert np.array_equal(small_pae, mixed_system.states[0]._oracles_result[oracle].pae[0]), (
        'incorrect pae information saved'
    )
    assert np.array_equal(small_plddt, mixed_system.states[0]._oracles_result[oracle].local_plddt[0]), (
        'incorrect plddt information saved'
    )
    assert np.array_equal(mixed_pae, mixed_system.states[1]._oracles_result[oracle].pae[0]), (
        'incorrect pae information saved'
    )
    assert np.array_equal(mixed_plddt, mixed_system.states[1]._oracles_result[oracle].local_plddt[0]), (
        'incorrect plddt information saved'
    )

    shutil.rmtree(experiment_folder)


def test_copied_system_is_independant_of_original_system(mixed_system: bg.System) -> None:
    copied_system = mixed_system.__copy__()
    mixed_system.states[0].chains[0].add_residue(amino_acid='A', index=0)
    assert mixed_system.states[0].chains[0] != copied_system.states[0].chains[0]


def test_system_states_still_reference_shared_chain_object_after_copy_method(shared_chain_system: bg.System) -> None:
    copied_system = shared_chain_system.__copy__()
    copied_system.states[0].chains[0].add_residue(amino_acid='A', index=0)
    assert copied_system.states[0].chains[0] == copied_system.states[1].chains[0]


def test_system_get_total_energy_gives_correct_output(mixed_system: bg.System) -> None:
    for state in mixed_system.states:
        state.get_energy = Mock()  # disable method for easier testing
    total_energy = mixed_system.get_total_energy()
    # state 0: energy=-0.5, state 1: energy=0.1
    assert np.isclose(total_energy, (-0.5 + 0.1))  # system energy is sum of state energies
