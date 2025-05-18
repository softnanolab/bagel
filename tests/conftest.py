"""Module used to configure pytest behaviour."""

import shutil
import pytest
import pathlib as pl
from unittest.mock import Mock
import numpy as np
from biotite.structure import AtomArray, Atom, array, concatenate
import bagel as bg


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
def folder(request) -> bg.folding.ESMFolder:
    """
    Fixture that must be called in tests that require folding. Behaviour based on  the --folding flag of the
    origional pytest call.
    """
    flag = request.config.getoption('--folding')
    if flag == 'skip':
        pytest.skip(reason='--folding flag of the origional pytest call set to skip')
    else:
        model = bg.folding.ESMFolder(use_modal=flag == 'modal')
        yield model
        del model

@pytest.fixture
def very_high_temp() -> float:
    """High temperature to make acceptance of any move 100%"""
    return 1e10 


@pytest.fixture
def short_chain() -> bg.Chain:
    """Chain with 5 amino acids."""
    return bg.Chain([bg.Residue(name='C', chain_ID='A', index=i) for i in range(5)])


@pytest.fixture
def pdb_path() -> str:
    """Location of protein data bank file of real human protein."""
    return str(pl.Path(__file__).resolve().parent / 'example_protein.pdb')


@pytest.fixture
def residues() -> list[bg.Residue]:
    """list of 5 Residue objects."""
    return [bg.Residue(name='C', chain_ID='A', index=i) for i in range(5)] + [
        bg.Residue(name='C', chain_ID='B', index=0)
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
def small_structure_residues() -> list[bg.Residue]:
    residues = [
        bg.Residue(name='G', chain_ID='A', index=0),
        bg.Residue(name='V', chain_ID='A', index=1),
        bg.Residue(name='V', chain_ID='B', index=0),
    ]
    return residues


@pytest.fixture
def small_structure_chains(small_structure_residues: list[bg.Residue]) -> list[bg.Chain]:
    return [bg.Chain(small_structure_residues[:2]), bg.Chain(small_structure_residues[-1:])]


@pytest.fixture
def small_structure_state(
    small_structure_chains: list[bg.Chain], small_structure_residues: list[bg.Residue], small_structure: AtomArray
) -> bg.State:
    energy_terms = [bg.energies.PTMEnergy(), bg.energies.SurfaceAreaEnergy(residues=small_structure_residues[1:])]
    state = bg.State(
        chains=small_structure_chains,
        energy_terms=energy_terms,
        energy_terms_weights=[1.0 for term in energy_terms],
        name='small',
    )
    state._energy = -0.5
    state._structure = small_structure
    folding_metrics = Mock(bg.folding.FoldingMetrics)
    folding_metrics.ptm = 0.7
    state._folding_metrics = folding_metrics
    state.energy_terms[0].value, state.energy_terms[1].value = [-0.7, 0.2]
    state._energy_terms_value = [-0.7, 0.2]
    state.chemical_potential = 1.0
    return state


@pytest.fixture
def line_structure() -> AtomArray:  # backbone atoms of first 2 residues form a diagonal line
    atoms = [
        Atom(coord=[0, 0, 0], chain_id='C', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[0, 0, 0], chain_id='C', atom_name='H', res_name='GLY', res_id=0, element='H'),
        Atom(coord=[1, 1, 0], chain_id='C', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[7, 7, 0], chain_id='D', atom_name='O', res_name='GLY', res_id=0, element='O'),
        Atom(coord=[2, 2, 0], chain_id='D', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[6, 4, 0], chain_id='D', atom_name='CA', res_name='VAL', res_id=1, element='C'),
        Atom(coord=[9, 0, 0], chain_id='D', atom_name='CA', res_name='VAL', res_id=1, element='C'),
    ]
    return array(atoms)


@pytest.fixture
def line_structure_residues() -> list[bg.Residue]:
    residues = [
        bg.Residue(name='G', chain_ID='C', index=0),
        bg.Residue(name='V', chain_ID='D', index=0),
        bg.Residue(name='V', chain_ID='D', index=1),
    ]
    return residues


@pytest.fixture
def line_structure_chains(line_structure_residues: list[bg.Residue]) -> list[bg.Chain]:
    return [bg.Chain(residues=line_structure_residues[:1]), bg.Chain(residues=line_structure_residues[1:])]


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
def square_structure_residues() -> list[bg.Residue]:
    residues = [
        bg.Residue(name='G', chain_ID='E', index=0),
        bg.Residue(name='V', chain_ID='E', index=1),
        bg.Residue(name='V', chain_ID='E', index=2),
        bg.Residue(name='V', chain_ID='E', index=3),
    ]
    return residues


@pytest.fixture
def square_structure_chains(square_structure_residues: list[bg.Residue]) -> list[bg.Chain]:
    return [bg.Chain(residues=square_structure_residues)]


@pytest.fixture
def mixed_structure_state(
    square_structure_chains: list[bg.Chain],
    line_structure_chains: list[bg.Chain],
    square_structure: AtomArray,
    line_structure: AtomArray,
    line_structure_residues: list[bg.Residue],
    square_structure_residues: list[bg.Residue],
) -> bg.State:
    # energy_terms = [bg.energies.PTMEnergy(), bg.energies.GlobularEnergy()]
    energy_terms = [
        bg.energies.PLDDTEnergy(residues=line_structure_residues + square_structure_residues),
        bg.energies.PAEEnergy(
            group_1_residues=line_structure_residues, group_2_residues=square_structure_residues, inheritable=False
        ),
    ]
    state = bg.State(
        chains=line_structure_chains + square_structure_chains,
        energy_terms=energy_terms,
        energy_terms_weights=[1.0 for term in energy_terms],
        name='mixed',
    )
    state._energy = 0.1
    state._structure = concatenate((line_structure, square_structure))
    folding_metrics = Mock(bg.folding.FoldingMetrics)
    folding_metrics.ptm = 0.4
    state._folding_metrics = folding_metrics
    state.energy_terms[0].value, state.energy_terms[1].value = [-0.4, 0.5]
    state._energy_terms_value = [-0.4, 0.5]
    state.chemical_potential = 2.0
    return state


@pytest.fixture
def mixed_system(small_structure_state: bg.State, mixed_structure_state: bg.State) -> bg.System:
    system = bg.System(
        states=[small_structure_state, mixed_structure_state],
        name='mixed_system',
    )
    system._old_energy = 0.67
    system.total_energy = -0.4
    return system


@pytest.fixture
def test_log_path(request) -> pl.Path:
    test_name = request.node.name
    path = pl.Path(__file__).resolve().parent / 'data' / test_name
    yield path
    shutil.rmtree(path)


@pytest.fixture
def base_sequence() -> str:
    return 'MKVWPQGHSTNRYLAEFCID'

#! @Jakub please check here
@pytest.fixture
def protein_language_model() -> bg.embedding.ESM2:
    """Returns a protein language model object."""
    # This is a mock of the ESM2 class, which is not implemented in the provided code.
    # In a real scenario, this would be replaced with the actual ESM2 class from the bagel module.
    return bg.embedding.ESM2(use_modal=True)

@pytest.fixture
def plm_only_state() -> bg.State:
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    oracle = bg.EmbeddingOracle( language_model = bg.embedding.ESM2(use_modal=True) )
    # This really should be taken from the ESM2 class
    n_features = 1280
    reference_embeddings = np.zeros( ( 2, n_features ))
    reference_embeddings[0, 0] = 1.0
    reference_embeddings[1, 0] = 1.0
    energy = bg.energies.PTMEnergy( weight = 0.5, residues=residues[:2], oracle=oracle, input_key='embedding',
                                   reference_embeddings=reference_embeddings )
    state = bg.State(
        oracles=[oracle],
        chains=[bg.Chain(residues)],
        energy_terms=[energy],
        name='state_A',
     )
    return state


@pytest.fixture
def simple_state() -> bg.State:
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)
    residues = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[bg.energies.PTMEnergy(), bg.energies.OverallPLDDTEnergy()],
        energy_terms_weights=[1.0, 1.0],
        name='state_A',
    )
    state._structure = AtomArray(length=len(residues))
    state.energy_terms[0].value = -1.0
    state.energy_terms[1].value = -0.5
    state._energy_terms_value = {
        state.energy_terms[0].name: state.energy_terms[0].value,
        state.energy_terms[1].name: state.energy_terms[1].value,
    }
    return state


@pytest.fixture
def shared_chain_system() -> bg.State:
    """System where each state references the same chain"""
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    shared_chain = bg.Chain(residues)

    A_state = bg.State(
        chains=[shared_chain],
        energy_terms=[bg.energies.PLDDTEnergy(residues), bg.energies.SurfaceAreaEnergy(residues)],
        energy_terms_weights=[1.0, 1.0],
        name='A',
    )

    B_state = bg.State(
        chains=[shared_chain],
        energy_terms=[bg.energies.PLDDTEnergy(residues), bg.energies.SurfaceAreaEnergy(residues)],
        energy_terms_weights=[1.0, 1.0],
        name='B',
    )
    return bg.System([A_state, B_state])


@pytest.fixture
def huge_system() -> bg.State:
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=10_000)
    residues = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[bg.energies.PTMEnergy(), bg.energies.OverallPLDDTEnergy()],
        energy_term_weights=[1.0, 2.0],
        name='state_A',
    )
    return bg.System([state])


@pytest.fixture
def energies_system() -> bg.State:
    """System where each state has an energy term that tracks all residues in state"""
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)

    A_residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    A_state = bg.State(
        chains=[bg.Chain(A_residues)],
        energy_terms=[bg.energies.PLDDTEnergy(A_residues), bg.energies.SurfaceAreaEnergy(A_residues)],
        energy_terms_weights=[1.0, 1.0],
        name='A',
    )

    B_residues = [bg.Residue(name=aa, chain_ID='B', index=i, mutable=True) for i, aa in enumerate(sequence)]
    B_state = bg.State(
        chains=[bg.Chain(B_residues)],
        energy_terms=[bg.energies.PLDDTEnergy(B_residues), bg.energies.SurfaceAreaEnergy(B_residues)],
        energy_terms_weights=[1.0, 1.0],
        name='B',
    )
    return bg.System([A_state, B_state])


@pytest.fixture
def SA_minimizer() -> bg.minimizer.SimulatedAnnealing:
    """Returns a minimizer that skips folding and closes all files minimizer creates by end of test."""
    with patch('desprot.minimizer.inspect.stack') as mock_inspect_function:
        # normally points to file executed, but is unpredictable when run by pytest and so must be set manually
        mock_inspect_function.return_value = [[1, __file__]]
        minimizer = bg.minimizer.SimulatedAnnealing(
            folder=None,
            mutator=bg.mutation.Canonical(),
            initial_temperature=0.35,
            final_temperature=0.1,
            n_steps=6,
            experiment_name=str(np.random.randint(low=0, high=999_999)),
            structure_log_frequency=2,
        )
    yield minimizer
    shutil.rmtree(minimizer.log_path)


@pytest.fixture
def ST_minimizer() -> bg.minimizer.SimulatedTempering:
    """Returns a minimizer that skips folding and closes all files minimizer creates by end of test."""
    with patch('desprot.minimizer.inspect.stack') as mock_inspect_function:
        # normally points to file executed, but is unpredictable when run by pytest and so must be set manually
        mock_inspect_function.return_value = [[1, __file__]]
        minimizer = bg.minimizer.SimulatedTempering(
            folder=None,
            mutator=bg.mutation.Canonical(),
            high_temperature=0.1,
            low_temperature=0.01,
            n_steps_high=2,
            n_steps_low=1,
            n_cycles=2,
            experiment_name=str(np.random.randint(low=0, high=999_999)),
            structure_log_frequency=2,
        )
    yield minimizer
    shutil.rmtree(minimizer.log_path)


@pytest.fixture
def monomer(base_sequence):
    residues = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    return [bg.Chain(residues=residues)]


@pytest.fixture
def dimer(base_sequence):
    residues_A = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    residues_B = [bg.Residue(name=aa, chain_ID='C-B', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    return [bg.Chain(residues=residues_A), bg.Chain(residues=residues_B)]


@pytest.fixture
def trimer(base_sequence):
    residues_A = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    residues_B = [bg.Residue(name=aa, chain_ID='C-B', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    residues_C = [bg.Residue(name=aa, chain_ID='C-C', index=i, mutable=True) for i, aa in enumerate(base_sequence)]
    return [bg.Chain(residues=residues_A), bg.Chain(residues=residues_B), bg.Chain(residues=residues_C)]


@pytest.fixture
def nominal_mixed_system(trimer: list[bg.Chain]) -> bg.System:
    """system with 3 mutable 20 amino acid chains. These are shared between 2 states with easier energies."""
    state_1 = bg.State(
        chains=trimer[:2],
        energy_terms=[bg.energies.PTMEnergy()],
        energy_term_weights=[1.0],
        name='state_1',
    )

    state_2 = bg.State(
        chains=trimer[1:],
        energy_terms=[bg.energies.OverallPLDDTEnergy()],
        energy_term_weights=[1.0],
        name='state_2',
    )

    return bg.System([state_1, state_2])


@pytest.fixture
def temp_path() -> pl.Path:
    num = np.random.randint(low=0, high=999_999)  # ensures multiple folders can be created at the same time
    path = pl.Path(__file__).resolve().parent / f'{num} data'
    yield path
    shutil.rmtree(path)
