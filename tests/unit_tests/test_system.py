# test add_chain

import bl.cklane as bl
import pandas as pd
import pathlib as pl
import shutil
from biotite.sequence.io.fasta import FastaFile
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np


def test_system_dump_config_file_is_correct(mixed_system: bl.System) -> None:
    # TODO: this should be likely also tested through the Minimizer somehow, as that initializes the folder
    mock_output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / mixed_system.name
    mock_experiment = test_system_dump_config_file_is_correct.__name__

    experiment_folder = mock_output_folder / mock_experiment
    if experiment_folder.exists():  # clean data folder if it exists
        shutil.rmtree(experiment_folder)
    experiment_folder.mkdir(parents=True, exist_ok=True)

    mixed_system.dump_config(experiment_folder)

    file = mock_output_folder / mock_experiment / 'config.csv'
    assert file.exists(), 'config file not present in expected location'
    config = pd.read_csv(file)

    assert all(config.columns == ['state', 'energy', 'weight']), 'config does not contain correct columns'
    correct_config = pd.DataFrame(
        {
            'state': ['small', 'small', 'mixed', 'mixed'],
            'energy': [
                'pTM',
                'selective_surface_area',
                'pTM',
                'normalized_globular',
            ],
            'weight': [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert all(config == correct_config), 'data within config file is incorrect'  # checks non index cols

    shutil.rmtree(experiment_folder)


def test_system_dump_logs_folder_is_correct(mixed_system: bl.System) -> None:
    # TODO: this should be likely also tested through the Minimizer somehow, as that initializes the folder
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

    structures = {
        'small': get_structure(CIFFile().read(file=experiment_folder / 'structures' / f'small_{mock_step}.cif'))[0],
        'mixed': get_structure(CIFFile().read(file=experiment_folder / 'structures' / f'mixed_{mock_step}.cif'))[0],
    }
    correct_structures = {'small': mixed_system.states[0]._structure, 'mixed': mixed_system.states[1]._structure}

    energies = pd.read_csv(experiment_folder / 'energies.csv')
    correct_energies = pd.DataFrame(
        {
            'step': [mock_step],
            'small:pTM': [-0.7],
            'small:selective_surface_area': [0.2],
            'mixed:pTM': [-0.4],
            'mixed:normalized_globular': [0.5],
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

    shutil.rmtree(experiment_folder)



# WARNING
# WARNING
# WARNING
# From here on added copying Ayham tests


from unittest.mock import Mock

def test_copied_system_is_independant_of_original_system(mixed_system: bl.System) -> None:
    copied_system = mixed_system.copy()
    mixed_system.states[0].chains[0].add_residue(amino_acid='A', index=0)
    assert mixed_system.states[0].chains[0] != copied_system.states[0].chains[0]

def test_system_states_still_reference_shared_chain_object_after_copy_method(shared_chain_system: bl.System) -> None:
    copied_system = shared_chain_system.copy()
    copied_system.states[0].chains[0].add_residue(amino_acid='A', index=0)
    assert copied_system.states[0].chains[0] == copied_system.states[1].chains[0]


def test_system_calculate_system_energies_method_gives_correct_output(mixed_system: bl.System) -> None:
    for state in mixed_system.states:
        state.get_energy() = Mock()  # disable method for easier testing
    total_system_energy = mixed_system.get_total_energy(folder=None)
    # state 0: energy=-0.5, chem_potential=1.0, n_residues=3. state 1: energy=0.1, chem_potential=2.0, n_residues=7
    assert np.isclose(total_system_energy, (-0.5 + 1 * 3 + 0.1 + 2 * 7) / 2)  # system energy is mean of state energies

def test_system_dump_logs_folder_is_correct(mixed_system: bl.System, temp_path: pl.Path) -> None:
    mock_step = 0
    mixed_system.is_calculated = True  # required for dump logs method to run
    mixed_system.dump_logs(step=mock_step, path=temp_path, save_structure=True)
    assert temp_path.is_dir(), 'system results folder not created'
    structure_path = temp_path / 'structures'
    assert structure_path.is_dir(), 'structure results folder not created'
    state = mixed_system.states[0]
    sequences = {header: sequence for header, sequence in FastaFile.read_iter(file=temp_path / f'{state.name}.fasta')}
    assert sequences == {'0': 'GV:V'}, 'incorrect sequence information saved for state 0'
    state = mixed_system.states[1]
    sequences = {header: sequence for header, sequence in FastaFile.read_iter(file=temp_path / f'{state.name}.fasta')}
    assert sequences == {'0': 'G:VV:GVVV'}, 'incorrect sequence information saved for state 1'

    structures = {
        'small': get_structure(CIFFile().read(file=structure_path / f'small_{mock_step}.cif'))[0],
        'mixed': get_structure(CIFFile().read(file=structure_path / f'mixed_{mock_step}.cif'))[0],
    }
    correct_structures = {'small': mixed_system.states[0].structure, 'mixed': mixed_system.states[1].structure}
    assert structures == correct_structures, 'incorrect structure information saved'

    energies = pd.read_csv(temp_path / 'energies.csv')
    correct_energies = pd.DataFrame(
        {
            'step': [mock_step],
            'small_chemical_potential_energy': [3.0],
            'small_pTM': [-0.7],
            'small_selective_surface_area': [0.2],
            'small_state_energy': [2.5],
            'mixed_chemical_potential_energy': [14.0],
            'mixed_local_pLDDT': [-0.4],
            'mixed_selective_surface_area': [0.5],
            'mixed_state_energy': [14.1],
            'system_energy': [-0.4],
        }
        )

    assert np.all(energies.columns == correct_energies.columns), 'Incorrect columns in energy.csv'
    assert np.all(energies.values == correct_energies.values), 'incorrect energy information saved'


def test_system_dump_config_file_is_correct(mixed_system: bl.System, temp_path: pl.Path) -> None:
    temp_path.mkdir()
    mixed_system.dump_config(path=temp_path)
    file_path = temp_path / 'config.csv'
    assert file_path.exists(), 'config file not present in expected location'
    config = pd.read_csv(file_path)
    assert all(config.columns == ['state_name', 'energy_name', 'weight']), 'config does not contain correct columns'
    correct_config = pd.DataFrame(
        {
            'state_name': ('small', 'small', 'small', 'mixed', 'mixed', 'mixed'),
            'energy_name': (
                'chemical_potential',
                'pTM',
                'selective_surface_area',
                'chemical_potential',
                'local_pLDDT',
                'selective_surface_area',
            ),
            'weight': (1.0, 1.0, 1.0, 2.0, 1.0, 1.0),
        }
    )

    assert all(config == correct_config), 'data within config file is incorrect'


#def test_system_unique_chains_method_gives_correct_output(mixed_system: bl.System) -> None:
#    unique_chains = mixed_system.states[0].chains + mixed_system.states[1].chains
#    sorted_unique_chains = sorted(unique_chains, key=lambda chain: chain.chain_ID)
#    mixed_system.states[0].chains += mixed_system.states[1].chains  # adding non unique chains to first state
#    assert sorted_unique_chains == sorted(mixed_system.unique_chains(), key=lambda chain: chain.chain_ID)