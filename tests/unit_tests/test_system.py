# test add_chain

import desprot as dp
import pandas as pd
import pathlib as pl
import shutil
from biotite.sequence.io.fasta import FastaFile
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np


def test_system_dump_config_file_is_correct(mixed_system: dp.System) -> None:
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mixed_system.output_folder = mock_output_folder
    mock_experiment = test_system_dump_config_file_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)

    mixed_system.dump_config(experiment=mock_experiment)
    file = mock_output_folder / mock_experiment / 'config.csv'
    assert file.exists(), 'config file not present in expected location'
    config = pd.read_csv(file)

    assert all(config.columns[1:] == ['state_ID', 'energy_name']), 'config does not contain correct columns'
    correct_config = pd.DataFrame(
        {
            'state_ID': ('small', 'small', 'mixed', 'mixed'),
            'energy_name': (
                'Predicted Template Modelling Score Energy',
                'Selective Normalized Surface Area Energy',
                'Predicted Template Modelling Score Energy',
                'Normalized Globular Energy',
            ),
        }
    )
    assert all(config.iloc[:, 1:] == correct_config), 'data within config file is incorrect'  # checks non index cols

    shutil.rmtree(experiment_folder)


def test_system_dump_logs_folder_is_correct(mixed_system: dp.System) -> None:
    mock_step = 8
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mixed_system.output_folder = mock_output_folder
    mock_experiment = test_system_dump_logs_folder_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)

    mixed_system.dump_logs(step=mock_step, experiment=mock_experiment)
    folder_path = mock_output_folder / mock_experiment
    assert folder_path.is_dir(), 'log folder not present in expected location'

    state = mixed_system.states[0]
    sequences = {
        header: sequence for header, sequence in FastaFile.read_iter(file=folder_path / f'{state.state_ID}.fasta')
    }
    assert sequences == {'8': 'GV:V'}, 'incorrect sequence information saved'

    state = mixed_system.states[1]
    sequences = {
        header: sequence for header, sequence in FastaFile.read_iter(file=folder_path / f'{state.state_ID}.fasta')
    }
    assert sequences == {'8': 'G:VV:GVVV'}, 'incorrect sequence information saved'

    structures = {
        'small': get_structure(CIFFile().read(file=folder_path / 'CIF' / f'small_{mock_step}.cif'))[0],
        'mixed': get_structure(CIFFile().read(file=folder_path / 'CIF' / f'mixed_{mock_step}.cif'))[0],
    }
    correct_structures = {'small': mixed_system.states[0]._structure, 'mixed': mixed_system.states[1]._structure}

    energies = pd.read_csv(folder_path / 'energies.csv')
    correct_energies = pd.DataFrame(
        {
            'step': [mock_step],
            'small:pTM': [-0.7],
            'small:selective_surface_area': [0.2],
            'mixed:pTM': [-0.4],
            'mixed:normalized_globular': [0.5],
            'small:total_energy': [-0.5],
            'mixed:total_energy': [0.1],
            'total_energy': [-0.4],
        }
    )

    assert structures == correct_structures, 'incorrect structure information saved'

    # Check column names match
    assert set(energies.columns) == set(correct_energies.columns), "DataFrame columns don't match"

    # Sort columns to ensure same order and compare values only
    assert np.array_equal(energies.sort_index(axis=1).values, correct_energies.sort_index(axis=1).values), (
        'incorrect energy information saved'
    )
