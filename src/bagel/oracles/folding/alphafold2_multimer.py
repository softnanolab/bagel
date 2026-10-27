"""AlphaFold2-Multimer structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class AlphaFold2MultimerResult(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from AlphaFold2-Multimer."""


class AlphaFold2Multimer(RankedSamplesFoldingOracle):
    """Predict protein structures with AlphaFold2-Multimer (ColabFold) through BoilerRoom.

    BoilerRoom runs ``num_models`` models (5 by default) and ranks them; BAGEL uses the top-ranked one. No MSA is
    searched unless you pass one: ``use_msa_server`` defaults to ``False`` (a ColabFold server search at every step would
    dominate the run time). Pass ``msa={chain_ID: A3M text or file}`` for chains that are never mutated; chains without
    an entry are folded from their sequence alone. Chains with identical sequences share one MSA in AlphaFold2, so they must
    be given the same MSA (an entry on only one of them is refused).
    """

    result_class: type[AlphaFold2MultimerResult] = AlphaFold2MultimerResult
    model_name = 'AlphaFold2Multimer'
    supports_user_msa = True
    identical_chains_share_msa = True

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.alphafold.alphafold2_multimer import AlphaFold2Multimer as AlphaFold2MultimerBoiler

        model_config = dict(config or {})
        model_config.setdefault('use_msa_server', False)
        self.model = AlphaFold2MultimerBoiler(backend=self.backend, device=self.device, config=model_config)
