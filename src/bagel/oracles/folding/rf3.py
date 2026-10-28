"""RF3 (RoseTTAFold 3) structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class RF3Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from RF3 (RoseTTAFold 3)."""


class RF3(RankedSamplesFoldingOracle):
    """Predict protein structures with RF3 (RoseTTAFold 3) through BoilerRoom.

    BoilerRoom returns ``diffusion_batch_size`` samples (5 by default); BAGEL uses the top-ranked one. RF3 never searches
    for an MSA: pass ``msa={chain_ID: A3M text or file}`` for chains that are never mutated, otherwise every chain is
    folded from its sequence alone. ``config={"optimization": "exact"}`` runs the optimization-kit kernels (A100/H100/H200
    only; refused on other GPUs).
    """

    result_class: type[RF3Result] = RF3Result
    model_name = 'RF3'
    supports_user_msa = True

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.rf3.rf3 import RF3 as RF3Boiler

        model_config = dict(config or {})
        self._require_samples(model_config, 'diffusion_batch_size', default=5)
        self.model = RF3Boiler(backend=self.backend, device=self.device, config=model_config)
