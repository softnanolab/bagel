import pytest
from bagel.oracles.base import OraclesResultDict, Oracle, OracleResult
from bagel.chain import Chain, Residue


class DummyOracle(Oracle):
    result_class = OracleResult

    def predict(self, chains):
        pass


class DummyResult(OracleResult):
    input_chains: list[Chain]

    def save_attributes(self, filepath):
        pass


def test_get_input_chains_returns_correct_chains():
    # Create dummy chains
    residues = [Residue(name='A', chain_ID='X', index=0), Residue(name='A', chain_ID='X', index=1)]
    chain1 = Chain(residues=residues[:1])
    chain2 = Chain(residues=residues[1:])
    input_chains = [chain1, chain2]

    # Create dummy oracle and result
    oracle = DummyOracle()
    result = DummyResult(input_chains=input_chains)

    # Insert into OraclesResultDict
    oracles_result = OraclesResultDict()
    oracles_result[oracle] = result

    # Assert get_input_chains returns the correct chains
    returned_chains = oracles_result.get_input_chains(oracle)
    assert returned_chains == input_chains


def test_get_input_chains_missing_oracle_raises():
    oracles_result = OraclesResultDict()
    oracle = DummyOracle()
    with pytest.raises(KeyError):
        _ = oracles_result.get_input_chains(oracle)
