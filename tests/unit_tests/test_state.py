import bagel as bg
import numpy as np
from biotite.structure import AtomArray
import pytest


def test_state_remove_residue_from_all_energy_terms_removes_correct_residue(mixed_structure_state: bg.State) -> None:
    # Remove a mutable residue in chain E (index 3 is mutable; index 2 is immutable in this fixture)
    mixed_structure_state.chains[2].remove_residue(index=3)
    mixed_structure_state.remove_residue_from_all_energy_terms(chain_ID='E', residue_index=3)

    chain_ids, res_ids = mixed_structure_state.energy_terms[0].residue_groups[0]
    # started at ['C', 'D', 'D', 'E', 'E', 'E', 'E'], [0, 0, 1, 0, 1, 2, 3], remove index 3 at chain E
    # removing index 3 E residue leaves the remaining E indices [0, 1, 2]
    assert np.all(chain_ids == ['C', 'D', 'D', 'E', 'E', 'E']) and np.all(res_ids == [0, 0, 1, 0, 1, 2])

    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[0]
    # started at ['C', 'D', 'D'], [0, 0, 1], should remain the same
    assert np.all(chain_ids == ['C', 'D', 'D']) and np.all(res_ids == [0, 0, 1])

    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[1]
    # started at ['E', 'E', 'E', 'E'], [0, 1, 2, 3], remove index 3 at chain E
    # removing index 3 E residue leaves the remaining E indices [0, 1, 2]
    assert np.all(chain_ids == ['E', 'E', 'E']) and np.all(res_ids == [0, 1, 2])


def test_state_add_residue_to_all_energy_terms_adds_residue_to_residue_group(mixed_structure_state: bg.State) -> None:
    mixed_structure_state.chains[1].add_residue(amino_acid='A', index=1)
    mixed_structure_state.add_residue_to_all_energy_terms(chain_ID='D', residue_index=1)

    # started at ['C', 'D', 'D', 'E', 'E', 'E', 'E'], [0, 0, 1, 0, 1, 2, 3]
    chain_ids, res_ids = mixed_structure_state.energy_terms[0].residue_groups[0]  # inheritable energy
    assert np.all(chain_ids == ['C', 'D', 'D', 'E', 'E', 'E', 'E', 'D'])  # extra D added at end
    assert np.all(
        res_ids == [0, 0, 2, 0, 1, 2, 3, 1]
    )  # index of original residue shifted up and added new residue at end

    # started at ['C', 'D', 'D'], [0, 0, 1] and shift of last index in chain D, but no residue added since non inheritable
    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[0]  # non inheritable energy
    assert np.all(chain_ids == ['C', 'D', 'D'])  # extra D not added
    assert np.all(res_ids == [0, 0, 2])  # index of original residue shifted up

    # started at ['E', 'E', 'E', 'E'], [0, 1, 2, 3] and remains the same because chain D is not part of this energy term
    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[1]  # non inheritable energy
    assert np.all(chain_ids == ['E', 'E', 'E', 'E'])  # extra D not added
    assert np.all(res_ids == [0, 1, 2, 3])  # index of original residue shifted up


def test_state_get_energy(fake_esmfold: bg.oracles.folding.ESMFold, monkeypatch) -> None:
    """Test the get_energy method of State class."""

    # Mock the predict method of ESMFold
    def mock_predict(self, chains):
        # Create a mock structure with minimal required data
        mock_structure = AtomArray(0)  # Empty structure
        return bg.oracles.folding.ESMFoldResult(
            input_chains=chains,
            structure=mock_structure,
            ptm=np.array([0.7]),  # Mock pTM value
            pae=np.zeros((0, 0)),  # Empty PAE matrix
            local_plddt=np.array([]),  # Empty local plDDT scores
        )

    monkeypatch.setattr(bg.oracles.folding.ESMFold, 'predict', mock_predict)

    # Create a mock Chain
    chain = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])

    # Create mock energy terms with different weights
    class MockEnergyTerm(bg.energies.EnergyTerm):
        def compute(self, oracles_result):
            # Return both unweighted and weighted energies
            return 1.0, 1.0 * self.weight

    # Test 1: First call - should calculate everything from scratch
    energy_term1 = MockEnergyTerm(
        name='MockEnergyTerm1',
        oracle=fake_esmfold,
        weight=2.0,
        inheritable=True,
    )
    energy_term2 = MockEnergyTerm(
        name='MockEnergyTerm2',
        oracle=fake_esmfold,
        weight=3.0,
        inheritable=True,
    )

    # Create state with mock components
    state = bg.State(name='test_state', chains=[chain], energy_terms=[energy_term1, energy_term2])

    energy = state.get_energy()
    assert energy == 5.0  # 1.0 * 2.0 + 1.0 * 3.0
    assert state._energy == 5.0
    assert state._energy_terms_value == {'MockEnergyTerm1': 1.0, 'MockEnergyTerm2': 1.0}
    assert fake_esmfold in state._oracles_result

    # Test 2: Second call - should use cached values
    original_oracles_result = state._oracles_result.copy()
    energy = state.get_energy()
    assert energy == 5.0
    assert state._oracles_result == original_oracles_result  # Should not recalculate oracle results

    # Test 3: Test with empty energy terms
    empty_state = bg.State(name='empty_state', chains=[chain], energy_terms=[])
    assert empty_state.get_energy() == 0.0

    # Test 4: Test that duplicate energy term names raise AssertionError
    duplicate_term1 = MockEnergyTerm(
        name='MockEnergyTerm',
        oracle=fake_esmfold,
        weight=2.0,
        inheritable=True,
    )
    duplicate_term2 = MockEnergyTerm(
        name='MockEnergyTerm',
        oracle=fake_esmfold,
        weight=3.0,
        inheritable=True,
    )

    duplicate_state = bg.State(
        name='duplicate_test_state', chains=[chain], energy_terms=[duplicate_term1, duplicate_term2]
    )

    with pytest.raises(AssertionError, match='Energy term names must be unique'):
        duplicate_state.get_energy()

    # Test 5: Test with duplicate PTM energy term
    ptm_duplicate_term1 = bg.energies.PTMEnergy(
        oracle=fake_esmfold,
        weight=2.0,
    )
    ptm_duplicate_term2 = bg.energies.PTMEnergy(
        oracle=fake_esmfold,
        weight=3.0,
    )

    ptm_duplicate_state = bg.State(
        name='ptm_duplicate_test_state', chains=[chain], energy_terms=[ptm_duplicate_term1, ptm_duplicate_term2]
    )

    with pytest.raises(AssertionError, match='Energy term names must be unique'):
        ptm_duplicate_state.get_energy()

    # Test 6: Test with named PTM energy term
    ptm_term1 = bg.energies.PTMEnergy(
        name='1',
        oracle=fake_esmfold,
        weight=2.0,
    )
    ptm_term2 = bg.energies.PTMEnergy(
        name='2',
        oracle=fake_esmfold,
        weight=3.0,
    )

    ptm_state = bg.State(name='ptm_test_state', chains=[chain], energy_terms=[ptm_term1, ptm_term2])
    ptm_energy = float(ptm_state.get_energy())  # Why is the energy output as an array?
    assert np.round(ptm_energy, 1) == -3.5  # -0.7 * 2.0 + -0.7 * 3.0

    # Test 7: Test with multiple oracles
    class MockOracleA(bg.oracles.Oracle):
        result_class = str  # Define result class for this mock oracle

        def predict(self, chains):
            return 'ResultA'

    class MockOracleB(bg.oracles.Oracle):
        result_class = str  # Define result class for this mock oracle

        def predict(self, chains):
            return 'ResultB'

    class MultiOracleEnergyTerm(bg.energies.EnergyTerm):
        def compute(self, oracles_result):
            # Just return 1.0 for both weighted and unweighted
            return 1.0, 1.0

    oracle_a = MockOracleA()
    oracle_b = MockOracleB()

    term_a = MultiOracleEnergyTerm(
        name='MultiOracleTermA',
        oracle=oracle_a,  # Single oracle per term
        weight=1.0,
        inheritable=True,
    )
    term_b = MultiOracleEnergyTerm(
        name='MultiOracleTermB',
        oracle=oracle_b,  # Single oracle per term
        weight=2.0,
        inheritable=True,
    )

    multi_oracle_state = bg.State(name='multi_oracle_state', chains=[chain], energy_terms=[term_a, term_b])

    energy = multi_oracle_state.get_energy()
    assert energy == 2.0  # 1.0 + 2.0
    assert len(multi_oracle_state._oracles_result) == 2  # Should have results from both oracles
    assert oracle_a in multi_oracle_state._oracles_result
    assert oracle_b in multi_oracle_state._oracles_result
