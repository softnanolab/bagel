"""Module used to configure pytest behaviour."""

import shutil
import pytest
import pathlib as pl
from unittest.mock import Mock
import numpy as np
from biotite.structure import AtomArray, Atom, array, concatenate
import desprot as dp


def pytest_addoption(parser):
    """Globally adds flag to pytest command line call. Used to specify how to handle tests that require folding."""
    parser.addoption(
        '--folding',
        required=True,
        action='store',
        help='What do do with tests that require folding. options: skip or local or modal',
        choices=('skip', 'local', 'modal'),
    )


"""
### PyTest Fixtures ###

These are imported implicitly in all test_*.py modules when pytest is called in the terminal.

This is confusing when just looking at said test_*.py modules, however explicitly importing these from a .py file leads
to bugs. The below fixture for example would create a new ESMFolder for each module, despite the session scope flag.
This was leading to test breaking exceptions.
"""


@pytest.fixture(scope='session')  # ensures only 1 modal container is requested per process
def folder(request) -> dp.folding.ESMFolder:
    """
    Fixture that must be called in tests that require folding. Behaviour based on  the --folding flag of the
    origional pytest call.
    """
    flag = request.config.getoption('--folding')
    if flag == 'skip':
        pytest.skip(reason='--folding flag of the origional pytest call set to skip')
    else:
        model = dp.folding.ESMFolder(use_modal=flag == 'modal')
        yield model
        del model


@pytest.fixture
def short_chain() -> dp.Chain:
    """Chain with 5 amino acids."""
    return dp.Chain([dp.Residue(name='C', chain_ID='A', index=i) for i in range(5)])


@pytest.fixture
def pdb_path() -> str:
    """Location of protein data bank file of real human protein."""
    return str(pl.Path(__file__).resolve().parent / 'example_protein.pdb')


@pytest.fixture
def residues() -> list[dp.Residue]:
    """list of 5 Residue objects."""
    return [dp.Residue(name='C', chain_ID='A', index=i) for i in range(5)] + [
        dp.Residue(name='C', chain_ID='B', index=3)
    ]


@pytest.fixture
def small_structure() -> AtomArray:
    atoms = [
        Atom(coord=[-1, 0, 0], chain_id='A', res_name='GLY', res_id=0, element='C', atom_name='C'),
        Atom(coord=[1, 0, 0], chain_id='A', res_name='GLY', res_id=0, element='C', atom_name='C'),
        Atom(coord=[0, -1, 0], chain_id='A', res_name='VAL', res_id=1, element='C', atom_name='C'),
        Atom(coord=[0, 1, 0], chain_id='A', res_name='VAL', res_id=1, element='C', atom_name='C'),
        Atom(coord=[0, 0, 0], chain_id='B', res_name='VAL', res_id=0, element='C', atom_name='C'),
    ]
    return array(atoms)


@pytest.fixture
def small_structure_residues() -> list[dp.Residue]:
    residues = [
        dp.Residue(name='G', chain_ID='A', index=0),
        dp.Residue(name='V', chain_ID='A', index=1),
        dp.Residue(name='V', chain_ID='B', index=0),
    ]
    return residues


@pytest.fixture
def small_structure_chains(small_structure_residues: list[dp.Residue]) -> list[dp.Chain]:
    return [dp.Chain(small_structure_residues[:2]), dp.Chain(small_structure_residues[-1:])]


@pytest.fixture
def small_structure_state(
    small_structure_chains: list[dp.Chain], small_structure_residues: list[dp.Residue], small_structure: AtomArray
) -> dp.State:
    energy_terms = [dp.energies.PTMEnergy(), dp.energies.SurfaceAreaEnergy(residues=small_structure_residues[1:])]
    state = dp.State(
        chains=small_structure_chains,
        energy_terms=energy_terms,
        energy_terms_weights=[1.0 for term in energy_terms],
        state_ID='small',
    )
    state._energy = -0.5
    state._structure = small_structure
    folding_metrics = Mock(dp.folding.FoldingMetrics)
    folding_metrics.ptm = 0.7
    state._folding_metrics = folding_metrics
    state.energy_terms[0].value, state.energy_terms[1].value = [-0.7, 0.2]
    state._energy_terms_value = [-0.7, 0.2]
    return state


@pytest.fixture
def line_structure() -> AtomArray:  # backbone atoms of first 2 residues form a diagonal line
    atoms = [
        Atom(coord=[0, 0, 0], chain_id='C', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[0, 0, 0], chain_id='C', atom_name='H', res_name='GLY', res_id=0, element='H'),
        Atom(coord=[1, 1, 0], chain_id='C', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[7, 7, 0], chain_id='D', atom_name='O', res_name='GLY', res_id=1, element='O'),
        Atom(coord=[2, 2, 0], chain_id='D', atom_name='CA', res_name='GLY', res_id=1, element='C'),
        Atom(coord=[6, 4, 0], chain_id='D', atom_name='CA', res_name='VAL', res_id=2, element='C'),
        Atom(coord=[9, 0, 0], chain_id='D', atom_name='CA', res_name='VAL', res_id=2, element='C'),
    ]
    return array(atoms)


@pytest.fixture
def line_structure_residues() -> list[dp.Residue]:
    residues = [
        dp.Residue(name='G', chain_ID='C', index=0),
        dp.Residue(name='V', chain_ID='D', index=1),
        dp.Residue(name='V', chain_ID='D', index=2),
    ]
    return residues


@pytest.fixture
def line_structure_chains(line_structure_residues: list[dp.Residue]) -> list[dp.Chain]:
    return [dp.Chain(residues=line_structure_residues[:1]), dp.Chain(residues=line_structure_residues[1:])]


@pytest.fixture
def square_structure() -> AtomArray:  # centroid of backbone atoms of each residue form a square of length 1
    atoms = [
        Atom(coord=[0, 0, -1], chain_id='E', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[0, 0, -1], chain_id='E', atom_name='H', res_name='GLY', res_id=0, element='H'),
        Atom(coord=[0, 0, 1], chain_id='E', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[0, 0, 2], chain_id='E', atom_name='O', res_name='GLY', res_id=0, element='O'),
        Atom(coord=[0, 1, -1], chain_id='E', atom_name='CA', res_name='GLY', res_id=1, element='C'),
        Atom(coord=[0, 1, 1], chain_id='E', atom_name='CA', res_name='GLY', res_id=1, element='C'),
        Atom(coord=[1, 1, -1], chain_id='E', atom_name='CA', res_name='VAL', res_id=2, element='C'),
        Atom(coord=[1, 1, 1], chain_id='E', atom_name='CA', res_name='VAL', res_id=2, element='C'),
        Atom(coord=[1, 0, -1], chain_id='E', atom_name='CA', res_name='VAL', res_id=3, element='C'),
        Atom(coord=[1, 3, -1], chain_id='E', atom_name='O', res_name='VAL', res_id=3, element='O'),
        Atom(coord=[1, 3, -1], chain_id='E', atom_name='H', res_name='VAL', res_id=3, element='H'),
        Atom(coord=[1, 0, 1], chain_id='E', atom_name='CA', res_name='VAL', res_id=3, element='C'),
    ]
    return array(atoms)


@pytest.fixture
def square_structure_residues() -> list[dp.Residue]:
    residues = [
        dp.Residue(name='G', chain_ID='E', index=0),
        dp.Residue(name='V', chain_ID='E', index=1),
        dp.Residue(name='V', chain_ID='E', index=2),
        dp.Residue(name='V', chain_ID='E', index=3),
    ]
    return residues


@pytest.fixture
def square_structure_chains(square_structure_residues: list[dp.Residue]) -> list[dp.Chain]:
    return [dp.Chain(residues=square_structure_residues)]


@pytest.fixture
def mixed_structure_state(
    square_structure_chains: list[dp.Chain],
    line_structure_chains: list[dp.Chain],
    square_structure: AtomArray,
    line_structure: AtomArray,
) -> dp.State:
    energy_terms = [dp.energies.PTMEnergy(), dp.energies.GlobularEnergy()]
    state = dp.State(
        chains=line_structure_chains + square_structure_chains,
        energy_terms=energy_terms,
        energy_terms_weights=[1.0 for term in energy_terms],
        state_ID='mixed',
    )
    state._energy = 0.1
    state._structure = concatenate((line_structure, square_structure))
    folding_metrics = Mock(dp.folding.FoldingMetrics)
    folding_metrics.ptm = 0.4
    state._folding_metrics = folding_metrics
    state.energy_terms[0].value, state.energy_terms[1].value = [-0.4, 0.5]
    state._energy_terms_value = [-0.4, 0.5]
    return state


@pytest.fixture
def mixed_system(small_structure_state: dp.State, mixed_structure_state: dp.State) -> dp.System:
    system = dp.System(
        states=[small_structure_state, mixed_structure_state],
        name='mixed_system',
        output_folder=pl.Path(__file__).resolve().parent / 'data' / 'mixed_system',
    )
    system._old_energy = 0.67
    system.total_energy = -0.4
    return system


@pytest.fixture
def test_output_folder() -> pl.Path:
    num = np.random.randint(low=0, high=999_999)  # ensures multiple folders can be created at the same time
    path = pl.Path(__file__).resolve().parent / f'{num} data'
    yield path
    shutil.rmtree(path)


@pytest.fixture
def base_sequence() -> str:
    return 'MKVWPQGHSTNRYLAEFCID'


@pytest.fixture
def simple_state() -> dp.State:
    sequence = np.random.choice(list(dp.constants.aa_dict.keys()), size=5)
    residues = [dp.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    state = dp.State(
        chains=[dp.Chain(residues)],
        energy_terms=[dp.energies.PTMEnergy(), dp.energies.OverallPLDDTEnergy(), dp.energies.HydrophobicEnergy()],
        energy_terms_weights=[1.0, 1.0, 5.0],
        state_ID='state_A',
        verbose=True,
    )
    return state


@pytest.fixture
def monomer(base_sequence):
    residues = [dp.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    return [dp.Chain(residues=residues)]


@pytest.fixture
def dimer(base_sequence):
    residues_A = [dp.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    residues_B = [dp.Residue(name=aa, chain_ID='C-B', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    return [dp.Chain(residues=residues_A), dp.Chain(residues=residues_B)]


@pytest.fixture
def trimer(base_sequence):
    residues_A = [dp.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    residues_B = [dp.Residue(name=aa, chain_ID='C-B', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    residues_C = [dp.Residue(name=aa, chain_ID='C-C', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    return [dp.Chain(residues=residues_A), dp.Chain(residues=residues_B), dp.Chain(residues=residues_C)]
