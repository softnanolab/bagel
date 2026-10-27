import numpy as np
import pytest
import bagel as bg
from biotite.structure import AtomArray


PROTENIX_WEIGHTS_UNAVAILABLE = pytest.mark.skip(
    reason='Protenix cannot download its checkpoint on Modal (HTTP 403 from the weights host); re-enable once boileroom fixes it'
)


@pytest.mark.parametrize(
    ('oracle_fixture', 'result_class'),
    [
        ('esmfold', bg.oracles.folding.ESMFoldResult),
        ('esmfold2', bg.oracles.folding.ESMFold2Result),
        ('boltz2', bg.oracles.folding.Boltz2Result),
        ('chai1', bg.oracles.folding.Chai1Result),
        ('rf3', bg.oracles.folding.RF3Result),
        pytest.param('protenix', bg.oracles.folding.ProtenixResult, marks=PROTENIX_WEIGHTS_UNAVAILABLE),
        ('opendde', bg.oracles.folding.OpenDDEResult),
        ('alphafold2_multimer', bg.oracles.folding.AlphaFold2MultimerResult),
    ],
)
def test_folding_oracle_modal_smoke(
    request,
    oracle_fixture: str,
    result_class: type[bg.oracles.folding.FoldingResult],
    monomer: list[bg.Chain],
) -> None:
    oracle = request.getfixturevalue(oracle_fixture)
    result = oracle.fold(monomer)

    num_residues = sum(len(chain.residues) for chain in monomer)

    assert isinstance(result, result_class)
    assert isinstance(result.structure, AtomArray)
    assert len(result.structure) > 0
    assert isinstance(result.local_plddt, np.ndarray)
    assert result.local_plddt.shape == (1, num_residues)
    assert isinstance(result.pae, np.ndarray)
    assert result.pae.shape == (1, num_residues, num_residues)
    assert isinstance(result.ptm, np.ndarray)

    unique_residue_ids = np.unique(result.structure.res_id)
    assert np.array_equal(unique_residue_ids, np.arange(num_residues))
    assert np.array_equal(np.unique(result.structure.chain_id), [monomer[0].chain_ID])


@pytest.mark.parametrize(
    'oracle_fixture',
    ['rf3', pytest.param('protenix', marks=PROTENIX_WEIGHTS_UNAVAILABLE), 'opendde', 'alphafold2_multimer'],
)
def test_multi_sample_oracle_modal_dimer(request, oracle_fixture: str, dimer: list[bg.Chain]) -> None:
    """The new oracles fold a complex and return one structure with BAGEL's chain ids and residue numbering."""
    result = request.getfixturevalue(oracle_fixture).fold(dimer)
    num_residues = sum(len(chain.residues) for chain in dimer)

    assert result.local_plddt.shape == (1, num_residues)
    assert result.pae.shape == (1, num_residues, num_residues)
    assert result.ptm.shape == (1, 1)
    assert np.all((result.local_plddt >= 0) & (result.local_plddt <= 1))
    assert np.array_equal(np.unique(result.structure.chain_id), sorted(chain.chain_ID for chain in dimer))
    for chain in dimer:
        residue_ids = np.unique(result.structure.res_id[result.structure.chain_id == chain.chain_ID])
        assert np.array_equal(residue_ids, np.arange(len(chain.residues)))


@pytest.mark.parametrize(
    'oracle_class',
    [
        bg.oracles.folding.ESMFold2,
        bg.oracles.folding.RF3,
        pytest.param(bg.oracles.folding.Protenix, marks=PROTENIX_WEIGHTS_UNAVAILABLE),
        bg.oracles.folding.OpenDDE,
        bg.oracles.folding.AlphaFold2Multimer,
    ],
)
def test_oracle_modal_fixed_msa_for_target_chain(request, oracle_class, base_sequence: str) -> None:
    """A fixed (here single-row) MSA for the target chain is accepted; the other chain is folded alone."""
    if request.config.getoption('--oracles') != 'modal':
        pytest.skip(reason='needs --oracles modal')
    # distinct sequences: AlphaFold2 shares one MSA between equal chains, so an MSA on one copy would apply to both
    target = bg.Chain([bg.Residue(name=aa, chain_ID='T', index=i) for i, aa in enumerate(base_sequence)])
    binder = bg.Chain(
        [bg.Residue(name=aa, chain_ID='B', index=i, mutable=True) for i, aa in enumerate(base_sequence[::-1])]
    )
    dimer = [target, binder]
    oracle = oracle_class(backend='modal', msa={target.chain_ID: f'>query\n{target.sequence}\n'})

    result = oracle.fold(dimer)

    num_residues = sum(len(chain.residues) for chain in dimer)
    assert result.local_plddt.shape == (1, num_residues)
    assert result.ptm.shape == (1, 1)


@pytest.mark.parametrize(
    'oracle_class', [bg.oracles.folding.RF3, bg.oracles.folding.ESMFold2, bg.oracles.folding.OpenDDE]
)
def test_oracle_modal_n_samples_gives_ranked_structures_and_an_averaged_energy(
    request, oracle_class, monomer: list[bg.Chain]
) -> None:
    """Several samples per prediction: n structures, best first, and a State energy averaged over them."""
    if request.config.getoption('--oracles') != 'modal':
        pytest.skip(reason='needs --oracles modal')
    oracle = oracle_class(backend='modal', n_samples=3)
    num_residues = sum(len(chain.residues) for chain in monomer)

    samples = oracle.fold_samples(monomer)

    assert len(samples) == 3
    for sample in samples:
        assert sample.local_plddt.shape == (1, num_residues)
        assert sample.ptm.shape == (1, 1)
        assert np.array_equal(np.unique(sample.structure.res_id), np.arange(num_residues))
    coordinates = [sample.structure.coord for sample in samples]
    assert not np.allclose(coordinates[0], coordinates[1])  # different diffusion draws, not copies

    term = bg.energies.PLDDTEnergy(oracle=oracle, residues=None, weight=1.0)
    state = bg.State('s', monomer, [term])
    state.energy
    state_per_sample = []
    for sample in state._sample_results[oracle]:
        results = bg.oracles.OraclesResultDict()
        results[oracle] = sample
        state_per_sample.append(term.compute(results)[0])
    assert len(state_per_sample) == 3
    assert state.energy_term_values[term.name] == pytest.approx(np.mean(state_per_sample))
    assert state.energy_term_stds[term.name] == pytest.approx(np.std(state_per_sample, ddof=1))
