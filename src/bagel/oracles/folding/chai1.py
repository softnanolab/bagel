"""Chai-1 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingOracle, ConfidenceFoldingResult


class Chai1Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Chai-1."""


class Chai1(ConfidenceFoldingOracle):
    """Predict protein structures with Chai-1 through BoilerRoom."""

    result_class: type[Chai1Result] = Chai1Result
    model_name = 'Chai1'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.chai.chai1 import Chai1 as Chai1Boiler

        model_config = dict(config or {})
        num_samples = model_config.setdefault('num_diffn_samples', 1)
        if num_samples != 1:
            raise ValueError('BAGEL supports exactly one Chai-1 diffusion sample; set num_diffn_samples=1')
        self.model = Chai1Boiler(backend=self.backend, device=self.device, config=model_config)
