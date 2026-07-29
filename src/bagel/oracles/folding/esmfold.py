"""ESMFold structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingOracle, ConfidenceFoldingResult


class ESMFoldResult(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from ESMFold."""


class ESMFold(ConfidenceFoldingOracle):
    """Predict protein structures with ESMFold through BoilerRoom."""

    result_class: type[ESMFoldResult] = ESMFoldResult
    model_name = 'ESMFold'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.esm.esmfold import ESMFold as ESMFoldBoiler

        self.model = ESMFoldBoiler(backend=self.backend, device=self.device, config=config)
