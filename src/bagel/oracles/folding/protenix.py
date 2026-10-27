"""Protenix structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class ProtenixResult(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Protenix."""


class Protenix(RankedSamplesFoldingOracle):
    """Predict protein structures with Protenix through BoilerRoom.

    BoilerRoom returns several samples (``sample=5`` by default); BAGEL uses the top-ranked one, always from a single seed (several seeds are refused). No MSA is searched
    unless you pass one: ``use_msa`` defaults to ``False`` (a fresh MSA-server search at every step would dominate the run
    time) and to ``True`` when ``msa={chain_ID: A3M text or file}`` is given, for chains that are never mutated; chains
    without an entry are folded from their sequence alone. ``config={"optimization": "exact"}`` runs the optimization-kit
    kernels (A100/H100/H200 only; refused on other GPUs).
    """

    result_class: type[ProtenixResult] = ProtenixResult
    model_name = 'Protenix'
    supports_user_msa = True
    explicit_empty_msa = True

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.protenix.protenix import Protenix as ProtenixBoiler

        model_config = dict(config or {})
        if len(str(model_config.get('seeds', '101')).split(',')) != 1:
            raise ValueError(
                f'BAGEL uses exactly one seed per fold (samples are the diffusion draws), got seeds={model_config["seeds"]!r}'
            )
        model_config.setdefault('use_msa', bool(self.msas))
        self.model = ProtenixBoiler(backend=self.backend, device=self.device, config=model_config)
