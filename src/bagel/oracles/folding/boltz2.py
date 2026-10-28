"""Boltz-2 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class Boltz2Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Boltz-2."""


class Boltz2(RankedSamplesFoldingOracle):
    """Predict protein structures with Boltz-2 through BoilerRoom.

    ``n_samples`` diffusion samples are generated per prediction (``diffusion_samples``) and ranked by pTM.
    """

    rank_by_ptm = True

    result_class: type[Boltz2Result] = Boltz2Result
    model_name = 'Boltz2'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.boltz.boltz2 import Boltz2 as Boltz2Boiler

        model_config = dict(config or {})
        self._require_samples(model_config, 'diffusion_samples', default=1)
        # the model gets exactly the config the user gave unless more than one sample is needed
        passed = model_config if (config is not None or self.n_samples > 1) else None
        self.model = Boltz2Boiler(backend=self.backend, device=self.device, config=passed)
