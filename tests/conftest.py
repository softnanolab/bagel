"""Module used to configure pytest behaviour."""

import os
import pytest
import shutil
import numpy as np
import pathlib as pl
from biotite.structure import AtomArray, Atom, array, concatenate
from biotite.structure.io import load_structure
import bagel as bg


def pytest_addoption(parser):
    """Globally adds flag to pytest command line call. Used to specify how to handle tests that require oracles."""
    parser.addoption(
        '--oracles',
        required=True,
        action='store',
        help='What do do with tests that require oracles. options: skip or local or modal',
        choices=('skip', 'local', 'modal'),
    )


"""
### PyTest Fixtures ###

These are imported implicitly in all test_*.py modules when pytest is called in the terminal.

This is confusing when just looking at said test_*.py modules, however explicitly importing these from a .py file leads
to bugs. The below fixture for example would create a new ESMFolder for each module, despite the session scope flag.
This was leading to test breaking exceptions.
"""

import modal
from boileroom import app


@pytest.fixture(scope='session')
def modal_app_context(request) -> modal.App:
    flag = request.config.getoption('--oracles')
    if flag == 'modal':
        modal_app_context = app.run()
        modal_app_context.__enter__()
        yield modal_app_context
        modal_app_context.__exit__(None, None, None)
    else:
        yield None


@pytest.fixture(scope='session')  # ensures only 1 Modal App is requested per process
def esmfold(request, modal_app_context) -> bg.oracles.folding.ESMFold:
    """
    Fixture that must be called in tests that require oracles.
    Behaviour based on  the --oracles flag of the origional pytest call.
    """
    flag = request.config.getoption('--oracles')
    if flag == 'skip':
        pytest.skip(reason='--oracles flag of the origional pytest call set to skip')
    elif flag == 'local':
        model = bg.oracles.folding.ESMFold(use_modal=False)
        yield model
        del model
    elif flag == 'modal':
        with modal.enable_output():
            model = bg.oracles.folding.ESMFold(use_modal=True, modal_app_context=modal_app_context)
            yield model
            del model
    else:
        raise ValueError(f'Unknown --oracles flag: {flag}')


@pytest.fixture(scope='session')
def esm2(request, modal_app_context) -> bg.oracles.embedding.ESM2:
    """Fixture that returns an ESM2 object."""
    flag = request.config.getoption('--oracles')
    if flag == 'skip':
        pytest.skip(reason='--oracles flag of the origional pytest call set to skip')
    elif flag == 'local':
        model = bg.oracles.embedding.ESM2(use_modal=False)
        yield model
        del model
    elif flag == 'modal':
        with modal.enable_output():
            model = bg.oracles.embedding.ESM2(use_modal=True, modal_app_context=modal_app_context)
            yield model
            del model
    else:
        raise ValueError(f'Unknown --oracles flag: {flag}')


@pytest.fixture
def fake_esm2(request, monkeypatch) -> bg.oracles.embedding.ESM2:
    """
    Fixture that returns an ESM2 object that doesn't load any model.
    Use this primarily for testing functions that require an Oracle input,
    but also mock the output of the Oracle.
    """

    # Create a dummy _load method
    def mock_load(self, config={}):
        pass

    # Patch the _load method
    monkeypatch.setattr(bg.oracles.embedding.ESM2, '_load', mock_load)

    # Now create the actual instance - _load will be patched
    return bg.oracles.embedding.ESM2(use_modal=False)


@pytest.fixture
def fake_esmfold(request, monkeypatch) -> bg.oracles.folding.ESMFold:
    """
    Fixture that returns an ESMFold object that doesn't load any model.
    Use this primarily for testing functions that require an Oracle input,
    but also mock the output of the Oracle.
    """

    # Create a dummy _load method
    def mock_load(self, config={}):
        pass

    # Patch the _load method
    monkeypatch.setattr(bg.oracles.folding.ESMFold, '_load', mock_load)

    # Now create the actual instance - _load will be patched
    return bg.oracles.folding.ESMFold(use_modal=False)


@pytest.fixture
def fake_state(fake_esmfold: bg.oracles.folding.ESMFold) -> bg.State:
    return bg.State(
        name='fake_state',
        chains=[bg.Chain(residues=[bg.Residue(name='C', chain_ID='A', index=i) for i in range(5)])],
        energy_terms=[],
    )


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
    return str(pl.Path(__file__).resolve().parent / 'structures' / 'example_protein.pdb')


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
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure_chains: list[bg.Chain],
    small_structure_residues: list[bg.Residue],
    small_structure: AtomArray,
) -> bg.State:
    energy_terms = [
        bg.energies.PTMEnergy(oracle=fake_esmfold, weight=1.0),
        bg.energies.SurfaceAreaEnergy(oracle=fake_esmfold, residues=small_structure_residues[1:], weight=1.0),
    ]
    state = bg.State(
        chains=small_structure_chains,
        energy_terms=energy_terms,
        name='small',
    )
    state._energy = -0.5
    folding_result = bg.oracles.folding.ESMFoldResult(
        input_chains=small_structure_chains,
        structure=small_structure,
        ptm=np.array([0.7])[None, :],
        pae=np.zeros((len(small_structure), len(small_structure)))[None, :, :],
        local_plddt=np.zeros(len(small_structure))[None, :],
    )
    state._energy_terms_value = {
        energy_terms[0].name: -0.7,
        energy_terms[1].name: 0.2,
    }
    state._oracles_result = bg.oracles.OraclesResultDict()
    state._oracles_result[state.oracles_list[0]] = folding_result
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
def simplest_dimer() -> AtomArray:
    # A 2-residues chain aligned along the x-axis plus an additional single chain residue
    # aligned along the 100 direction, form a isocele triangle with basis 1 and cross-distances
    # of sqrt(5)/2.
    atoms = [
        Atom(coord=[0, 0, 0], chain_id='A', atom_name='CA', res_name='GLY', res_id=0, element='C'),
        Atom(coord=[1, 0, 0], chain_id='A', atom_name='CA', res_name='GLY', res_id=1, element='C'),
        Atom(coord=[0.5, 1, 0], chain_id='B', atom_name='CA', res_name='GLY', res_id=0, element='C'),
    ]
    return array(atoms)


@pytest.fixture
def simplest_dimer_residues() -> list[bg.Residue]:
    residues = [
        bg.Residue(name='G', chain_ID='A', index=0),
        bg.Residue(name='G', chain_ID='A', index=1),
        bg.Residue(name='G', chain_ID='B', index=0),
    ]
    return residues


@pytest.fixture
def square_structure_residues() -> list[bg.Residue]:
    residues = [
        bg.Residue(name='G', chain_ID='E', index=0),
        bg.Residue(name='V', chain_ID='E', index=1, mutable=False),
        bg.Residue(name='V', chain_ID='E', index=2, mutable=False),
        bg.Residue(name='V', chain_ID='E', index=3),
    ]
    return residues


@pytest.fixture
def square_structure_chains(square_structure_residues: list[bg.Residue]) -> list[bg.Chain]:
    return [bg.Chain(residues=square_structure_residues)]


@pytest.fixture
def simplest_dimer_chains(simplest_dimer_residues: list[bg.Residue]) -> list[bg.Chain]:
    return [bg.Chain(residues=simplest_dimer_residues[:2]), bg.Chain(residues=simplest_dimer_residues[2:])]


@pytest.fixture
def simplest_dimer_state(
    fake_esmfold: bg.oracles.folding.ESMFold,
    simplest_dimer_chains: list[bg.Chain],
    simplest_dimer_residues: list[bg.Residue],
    simplest_dimer: AtomArray,
) -> bg.State:
    energy_terms = [
        bg.energies.PLDDTEnergy(
            oracle=fake_esmfold,
            residues=simplest_dimer_residues,
            weight=1.0,
        ),
        bg.energies.FlexEvoBindEnergy(
            oracle=fake_esmfold,
            residues=[simplest_dimer_residues[0:2], [simplest_dimer_residues[2]]],
            plddt_weighted=True,
            symmetrized=True,
            weight=1.0,
        ),
    ]
    state = bg.State(
        chains=simplest_dimer_chains,
        energy_terms=energy_terms,
        name='simplest_dimer',
    )
    folding_result = bg.oracles.folding.ESMFoldResult(
        input_chains=simplest_dimer_chains,
        structure=simplest_dimer,
        local_plddt=0.5 * np.ones(len(simplest_dimer))[None, :],
        ptm=np.array([0.4])[None, :],
        pae=np.zeros((len(simplest_dimer), len(simplest_dimer)))[None, :, :],
    )
    state._energy = 0.0
    state._oracles_result = bg.oracles.OraclesResultDict()
    state._oracles_result[state.oracles_list[0]] = folding_result
    return state


@pytest.fixture
def formolase_ordered_structure() -> AtomArray:
    pdb_path = os.path.join(os.path.dirname(__file__), 'structures', '4qq8_ordered.pdb')
    structure = load_structure(pdb_path)
    return structure


@pytest.fixture
def formolase_ordered_residues(formolase_ordered_structure: AtomArray) -> list[bg.Residue]:
    all_residues = []
    for chain_id in np.unique(formolase_ordered_structure.chain_id):
        chain_mask = formolase_ordered_structure.chain_id == chain_id
        sequence = bg.oracles.folding.utils.sequence_from_atomarray(formolase_ordered_structure[chain_mask])
        residues = [bg.Residue(name=aa, chain_ID=chain_id, index=i) for i, aa in enumerate(sequence)]
        all_residues.extend(residues)
    return all_residues


@pytest.fixture
def formolase_structure() -> AtomArray:
    pdb_path = os.path.join(os.path.dirname(__file__), 'structures', '4qq8_protein_only.pdb')
    structure = load_structure(pdb_path)
    return structure


@pytest.fixture
def mixed_structure_state(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_chains: list[bg.Chain],
    line_structure_chains: list[bg.Chain],
    square_structure: AtomArray,
    line_structure: AtomArray,
    line_structure_residues: list[bg.Residue],
    square_structure_residues: list[bg.Residue],
) -> bg.State:
    energy_terms = [
        bg.energies.PLDDTEnergy(
            oracle=fake_esmfold,
            residues=line_structure_residues + square_structure_residues,
            weight=1.0,
        ),
        bg.energies.PAEEnergy(
            oracle=fake_esmfold,
            residues=[line_structure_residues, square_structure_residues],
            inheritable=False,
            weight=1.0,
        ),
    ]
    state = bg.State(
        chains=line_structure_chains + square_structure_chains,
        energy_terms=energy_terms,
        name='mixed',
    )
    folding_result = bg.oracles.folding.ESMFoldResult(
        input_chains=line_structure_chains + square_structure_chains,
        structure=concatenate((line_structure, square_structure)),
        ptm=np.array([0.4])[None, :],
        pae=np.zeros((len(line_structure), len(line_structure)))[None, :, :],
        local_plddt=np.zeros(len(line_structure))[None, :],
    )
    state._energy = 0.1
    state._energy_terms_value = {
        energy_terms[0].name: -0.4,
        energy_terms[1].name: 0.5,
    }
    state._oracles_result = bg.oracles.OraclesResultDict()
    state._oracles_result[state.oracles_list[0]] = folding_result
    return state


@pytest.fixture
def mixed_system(small_structure_state: bg.State, mixed_structure_state: bg.State) -> bg.System:
    system = bg.System(
        states=[small_structure_state, mixed_structure_state],
        name='mixed_system',
    )
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


@pytest.fixture
def plm_only_state(esm2: bg.oracles.embedding.ESM2) -> bg.State:
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

    # This really should be taken from the ESM2 class
    n_features = 1280
    reference_embeddings = np.zeros((2, n_features))
    reference_embeddings[0, 0] = 1.0
    reference_embeddings[1, 0] = 1.0
    energy = bg.energies.EmbeddingsSimilarityEnergy(
        oracle=esm2,
        residues=residues[:2],
        reference_embeddings=reference_embeddings,
        weight=0.5,
    )
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[energy],
        name='state_A',
    )
    return state


@pytest.fixture
def simple_state(fake_esmfold: bg.oracles.folding.ESMFold) -> bg.State:
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)
    residues = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[
            bg.energies.PTMEnergy(oracle=fake_esmfold, weight=1.0),
            bg.energies.OverallPLDDTEnergy(oracle=fake_esmfold, weight=1.0),
        ],
        name='state_A',
    )
    state._structure = AtomArray(length=len(residues))
    state._energy_terms_value = {
        state.energy_terms[0].name: -1.0,
        state.energy_terms[1].name: -0.5,
    }
    return state


@pytest.fixture
def real_simple_state(simple_state: bg.State, esmfold: bg.oracles.folding.ESMFold) -> bg.State:
    state_copy = simple_state.__copy__()
    state_copy._energy_terms_value = {}
    for term in state_copy.energy_terms:
        term.oracle = esmfold
    return state_copy


@pytest.fixture
def shared_chain_system(fake_esmfold: bg.oracles.folding.ESMFold) -> bg.State:
    """System where each state references the same chain"""
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    shared_chain = bg.Chain(residues)

    A_state = bg.State(
        chains=[shared_chain],
        energy_terms=[
            bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues, weight=1.0),
            bg.energies.SurfaceAreaEnergy(oracle=fake_esmfold, residues=residues, weight=1.0),
        ],
        name='A',
    )

    B_state = bg.State(
        chains=[shared_chain],
        energy_terms=[
            bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues, weight=1.0),
            bg.energies.SurfaceAreaEnergy(oracle=fake_esmfold, residues=residues, weight=1.0),
        ],
        name='B',
    )
    return bg.System([A_state, B_state])


@pytest.fixture
def huge_system() -> bg.State:
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=10_000)
    residues = [bg.Residue(name=aa, chain_ID='C-A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[bg.energies.PTMEnergy(weight=1.0), bg.energies.OverallPLDDTEnergy(weight=2.0)],
        name='state_A',
    )
    return bg.System([state])


@pytest.fixture
def energies_system(fake_esmfold: bg.oracles.folding.ESMFold) -> bg.System:
    """System where each state has an energy term that tracks all residues in state"""
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=5)

    A_residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    A_state = bg.State(
        chains=[bg.Chain(A_residues)],
        energy_terms=[
            bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=A_residues, weight=1.0),
            bg.energies.SurfaceAreaEnergy(oracle=fake_esmfold, residues=A_residues, weight=1.0),
        ],
        name='A',
    )

    B_residues = [bg.Residue(name=aa, chain_ID='B', index=i, mutable=True) for i, aa in enumerate(sequence)]
    B_state = bg.State(
        chains=[bg.Chain(B_residues)],
        energy_terms=[
            bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=B_residues, weight=1.0),
            bg.energies.SurfaceAreaEnergy(oracle=fake_esmfold, residues=B_residues, weight=1.0),
        ],
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
        energy_terms=[bg.energies.PTMEnergy(weight=1.0)],
        name='state_1',
    )

    state_2 = bg.State(
        chains=trimer[1:],
        energy_terms=[bg.energies.OverallPLDDTEnergy(weight=1.0)],
        name='state_2',
    )

    return bg.System([state_1, state_2])


@pytest.fixture
def temp_path() -> pl.Path:
    num = np.random.randint(low=0, high=999_999)  # ensures multiple folders can be created at the same time
    path = pl.Path(__file__).resolve().parent / f'{num} data'
    yield path
    shutil.rmtree(path)
