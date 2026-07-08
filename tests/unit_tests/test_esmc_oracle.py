"""Unit tests for ESMC and ESMFold2 oracles."""

import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, array

import bagel as bg
from bagel.oracles.base import OraclesResultDict
from bagel.oracles.embedding.esmc import ESMC, ESMCResult
from bagel.oracles.folding.esmfold2 import ESMFold2, ESMFold2Result


class TestESMCResult:
    """Tests for the ESMCResult dataclass."""

    def test_esmc_result_stores_embeddings(self):
        residues = [bg.Residue(name='A', chain_ID='A', index=0)]
        chains = [bg.Chain(residues)]
        embeddings = np.random.randn(5, 128).astype(np.float64)
        result = ESMCResult(input_chains=chains, embeddings=embeddings)
        assert result.embeddings.shape == (5, 128)
        assert result.input_chains == chains

    def test_esmc_result_in_oracles_dict(self):
        residues = [bg.Residue(name='A', chain_ID='A', index=0)]
        chains = [bg.Chain(residues)]
        embeddings = np.random.randn(1, 128).astype(np.float64)

        # Use a mock oracle that has result_class = ESMCResult
        oracle = ESMC.__new__(ESMC)
        oracle.result_class = ESMCResult
        result = ESMCResult(input_chains=chains, embeddings=embeddings)

        oracles_result = OraclesResultDict()
        oracles_result[oracle] = result
        assert oracles_result.get_embeddings(oracle).shape == (1, 128)


class TestESMFold2Result:
    """Tests for the ESMFold2Result dataclass."""

    def test_esmfold2_result_validates_plddt_range(self):
        residues = [bg.Residue(name='A', chain_ID='A', index=0)]
        chains = [bg.Chain(residues)]
        mock_structure = array(
            [Atom(coord=[0.0, 0.0, 0.0], chain_id='A', res_id=0, res_name='ALA', atom_name='CA', element='C')]
        )

        # Valid pLDDT values (0 to 1)
        result = ESMFold2Result(
            input_chains=chains,
            structure=mock_structure,
            local_plddt=np.array([[0.5]]),
            ptm=np.array([[0.7]]),
            pae=np.zeros((1, 1, 1)),
        )
        assert np.allclose(result.local_plddt, 0.5)

    def test_esmfold2_result_rejects_invalid_plddt(self):
        residues = [bg.Residue(name='A', chain_ID='A', index=0)]
        chains = [bg.Chain(residues)]
        mock_structure = array(
            [Atom(coord=[0.0, 0.0, 0.0], chain_id='A', res_id=0, res_name='ALA', atom_name='CA', element='C')]
        )

        with pytest.raises(ValueError, match='local_plddt'):
            ESMFold2Result(
                input_chains=chains,
                structure=mock_structure,
                local_plddt=np.array([[1.5]]),  # out of range
                ptm=np.array([[0.7]]),
                pae=np.zeros((1, 1, 1)),
            )

    def test_esmfold2_result_in_oracles_dict(self):
        residues = [bg.Residue(name='A', chain_ID='A', index=0)]
        chains = [bg.Chain(residues)]
        mock_structure = array(
            [Atom(coord=[0.0, 0.0, 0.0], chain_id='A', res_id=0, res_name='ALA', atom_name='CA', element='C')]
        )

        oracle = ESMFold2.__new__(ESMFold2)
        oracle.result_class = ESMFold2Result
        result = ESMFold2Result(
            input_chains=chains,
            structure=mock_structure,
            local_plddt=np.array([[0.5]]),
            ptm=np.array([[0.7]]),
            pae=np.zeros((1, 1, 1)),
        )

        oracles_result = OraclesResultDict()
        oracles_result[oracle] = result
        returned_structure = oracles_result.get_structure(oracle)
        assert len(returned_structure) == 1


class TestESMCOracle:
    """Tests for the ESMC oracle class."""

    def test_esmc_result_class(self):
        assert ESMC.result_class is ESMCResult

    def test_esmc_pre_process_monomer(self, fake_esmc):
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        result = fake_esmc._pre_process(chains)
        assert result == ['AAA']

    def test_esmc_pre_process_multimer(self, fake_esmc):
        chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
        chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
        result = fake_esmc._pre_process([chain_a, chain_b])
        assert result == ['AAA:GG']


class TestESMFold2Oracle:
    """Tests for the ESMFold2 oracle class."""

    def test_esmfold2_result_class(self):
        assert ESMFold2.result_class is ESMFold2Result

    def test_esmfold2_pre_process_monomer(self, fake_esmfold2):
        chains = [bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])]
        result = fake_esmfold2._pre_process(chains)
        assert result == ['AAA']

    def test_esmfold2_pre_process_multimer(self, fake_esmfold2):
        chain_a = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
        chain_b = bg.Chain([bg.Residue(name='G', chain_ID='B', index=i) for i in range(2)])
        result = fake_esmfold2._pre_process([chain_a, chain_b])
        assert result == ['AAA:GG']
