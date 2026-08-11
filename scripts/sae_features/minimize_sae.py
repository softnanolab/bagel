"""Energy minimisation over ESM-C sparse-autoencoder (SAE) features.

This script runs a **canonical Monte Carlo** simulation at **low temperature** in
which mutations are accepted/rejected by a :class:`~bagel.energies.SAEnergy` term.
SAEnergy is a linear function of a user-chosen set of SAE features of the design
(with default coefficients all equal to ``1``), so the low-temperature MC walk
greedily drives the sequence toward the concepts those features encode.

By default ``SAEnergy`` negates the (L1-normalized) feature combination, so:

* a **positive** coefficient (the default) pushes the corresponding feature
  *up* (design toward that concept), while
* a **negative** coefficient pushes the feature *down* (design away from it).

(Pass ``maximize=False`` to flip this, i.e. minimize the selected features.)

The heavy pieces (the SAE oracle and the MC loop) are factored into small helper
functions so the wiring can be unit-tested with a fake oracle, without Modal or a
GPU. See ``tests/unit_tests/test_sae_minimize_script.py``.

Run it (real ESM-C SAE model, via Modal by default)::

    python scripts/sae_features/minimize_sae.py

Set ``BAGEL_BACKEND=apptainer`` for local containerised execution.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import bagel as bg

# A short example scaffold to optimise. Replace with your own target.
DEFAULT_SEQUENCE = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQATHVDQWDWEWAGIKATEAFLPDYPDLDA'

# Example set of SAE feature indices to steer (illustrative; replace with the
# features whose concepts you want to design for/against). With default
# coefficients (all 1) and the default sign, the MC walk *increases* the summed
# activation of these features across the protein.
DEFAULT_FEATURE_INDICES = [12, 128, 4096, 8000]


def build_sae_oracle(backend: str, config: dict[str, Any] | None = None) -> bg.oracles.SAE:
    """Construct the SAE feature oracle for the given backend.

    With no ``config`` this uses BoilerRoom's default SAE (ESMC-6B / layer 60 via
    the Biohub Forge API, which needs ``ESM_API_KEY`` set or ``forge_token`` in the
    config). Pass ``{'feature_source': 'local', 'esmc_model_name': 'esmc_600m',
    'sae_repo_id': ..., 'sae_layer': ...}`` to run a 300M/600M SAE locally.
    """
    return bg.oracles.SAE(backend=backend, config=config or {})


def build_system(
    oracle: bg.oracles.embedding.EmbeddingOracle,
    sequence: str = DEFAULT_SEQUENCE,
    feature_indices: Sequence[int] = tuple(DEFAULT_FEATURE_INDICES),
    coefficients: Sequence[float] | None = None,
    weight: float = 1.0,
    chain_id: str = 'A',
) -> bg.System:
    """Assemble a single-state :class:`~bagel.System` driven by an SAEnergy term.

    Parameters
    ----------
    oracle : EmbeddingOracle
        The SAE oracle (real, or a fake for testing).
    sequence : str
        Starting amino-acid sequence; all residues are mutable.
    feature_indices : Sequence[int]
        SAE features summed by the energy term.
    coefficients : Sequence[float] | None
        Linear coefficients aligned with ``feature_indices`` (default all ones).
    weight : float
        Overall weight of the SAEnergy term.
    chain_id : str
        Chain identifier for the designed chain.
    """
    residues = [bg.Residue(name=aa, chain_ID=chain_id, index=i, mutable=True) for i, aa in enumerate(sequence)]
    chain = bg.Chain(residues=residues)

    energy_terms = [
        bg.energies.SAEnergy(
            oracle=oracle,
            feature_indices=list(feature_indices),
            coefficients=list(coefficients) if coefficients is not None else None,
            weight=weight,
        )
    ]

    state = bg.State(name='sae_state', chains=[chain], energy_terms=energy_terms)
    return bg.System(states=[state])


def build_minimizer(
    temperature: float = 0.01,
    n_steps: int = 2000,
    n_mutations: int = 1,
    log_interval: int = 20,
) -> bg.minimizer.MonteCarloMinimizer:
    """Build a low-temperature canonical Monte Carlo minimizer.

    Parameters
    ----------
    temperature : float
        MC temperature. Low values (default ``0.01``) make the walk close to
        greedy energy minimisation while still allowing occasional uphill moves.
    n_steps : int
        Number of MC steps.
    n_mutations : int
        Mutations proposed per step (canonical: substitutions only, fixed length).
    log_interval : int
        Logging cadence.
    """
    return bg.minimizer.MonteCarloMinimizer(
        mutator=bg.mutation.Canonical(n_mutations=n_mutations),
        temperature=temperature,
        n_steps=n_steps,
        callbacks=[bg.callbacks.DefaultLogger(log_interval=log_interval)],
    )


def run(
    sequence: str = DEFAULT_SEQUENCE,
    feature_indices: Sequence[int] = tuple(DEFAULT_FEATURE_INDICES),
    coefficients: Sequence[float] | None = None,
    weight: float = 1.0,
    temperature: float = 0.01,
    n_steps: int = 2000,
    backend: str | None = None,
) -> bg.System:
    """Run the SAE-feature energy minimisation and return the final system."""
    backend = backend or os.getenv('BAGEL_BACKEND', 'modal')
    print(f'Backend: {backend}')
    oracle = build_sae_oracle(backend)
    system = build_system(
        oracle=oracle,
        sequence=sequence,
        feature_indices=feature_indices,
        coefficients=coefficients,
        weight=weight,
    )
    minimizer = build_minimizer(temperature=temperature, n_steps=n_steps)
    return minimizer.minimize_system(system=system)


def main() -> None:
    """Entry point that runs the design loop against the real ESM-C SAE model."""
    import modal

    with modal.enable_output():
        run()


if __name__ == '__main__':
    main()
