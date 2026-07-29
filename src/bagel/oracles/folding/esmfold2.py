"""ESMFold2 structure-prediction oracle."""

from typing import Any

from boileroom.models.esmfold2.types import ProteinInput, StructurePredictionInput
from ...chain import Chain
from .base import ConfidenceFoldingOracle, ConfidenceFoldingResult


class ESMFold2Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from ESMFold2."""


class ESMFold2(ConfidenceFoldingOracle):
    """Oracle that uses ESMFold2 to predict protein structures from sequence.

    ESMFold2 is a diffusion-based all-atom structure prediction model from
    EvolutionaryScale. The boileroom wrapper runs it **locally** on the selected
    backend (Modal GPU or Apptainer): model weights are loaded from Hugging Face
    (``biohub/ESMFold2``) and inference runs in-process. No inference API or API
    token is involved.

    Multimers are handled **natively** as separate chains — ESMFold2 does not use
    a glycine linker or a positional-id skip. ``_pre_process`` builds a structured
    boileroom fold input (one protein entity per chain) so ESMFold2 receives
    bagel's own ``chain_ID``s directly, rather than the auto-assigned ``A``/``B``/
    ``C`` labels produced by the ``':'``-delimited string shortcut.

    Parameters
    ----------
    backend : str
        BoilerRoom backend, normally ``"modal"`` or ``"apptainer"``.
    device : str | None
        Optional device passed to BoilerRoom.
    config : dict
        Configuration options forwarded to the boileroom ESMFold2 wrapper.
        Supported keys include:
        - model_name: ESMFold2 weights to load (e.g. "biohub/ESMFold2")
        - num_sampling_steps: Diffusion sampling steps
        - num_loops: Refinement loops
    """

    result_class: type[ESMFold2Result] = ESMFold2Result
    model_name = 'ESMFold2'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        # Imported here (not at module scope) so the module stays importable
        # without the boileroom ESMFold2 wrapper present, and so mocked oracles in
        # tests can patch out _load entirely.
        from boileroom.models.esmfold2.esmfold2 import ESMFold2 as ESMFold2Boiler

        self.model = ESMFold2Boiler(backend=self.backend, device=self.device, config=config)

    def _pre_process(self, chains: list[Chain]) -> StructurePredictionInput:
        """Build a structured ESMFold2 fold input that preserves each ``chain_ID``.

        Returns one typed complex with a protein entity per chain keyed by the
        chain's own ``chain_ID``. Using structured input (instead of a ``':'``-joined string)
        makes ESMFold2 fold the chains as a single complex while receiving bagel's
        chain IDs directly.
        """
        return StructurePredictionInput(
            sequences=[ProteinInput(id=chain.chain_ID, sequence=chain.sequence) for chain in chains]
        )
