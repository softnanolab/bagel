"""
Example: Designing a protein binder using Protenix as the folding oracle.

This script demonstrates how to use Protenix instead of ESMFold for
structure prediction in BAGEL's design loop.

Key differences from the ESMFold-based workflow:
  1. No glycine linker needed -- Protenix handles multi-chain natively
  2. No positional encoding skip needed
  3. Protenix provides ipTM scores for interface quality assessment
  4. Protenix uses a diffusion-based generative model (AlphaFold 3 architecture)
"""

import random
import os
from typing import Any

import bagel as bg


def run_protenix_binder() -> Any:
    #use_modal = True if os.getenv('USE_MODAL', 'True').lower() in ('true', '1', 'yes') else False
    use_modal = True
    print(f'Whether to use modal: {use_modal}')

    # Define the target protein
    # This sequence comes from a PDB of the interleukin-8 protein
    # >1IL8_1|Chains A, B|INTERLEUKIN-8|Homo sapiens (9606)
    target_sequence = 'SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS'

    # Define immutable residues for the target
    residues_target = [
        bg.Residue(name=aa, chain_ID='Tar', index=i, mutable=False)
        for i, aa in enumerate(target_sequence)
    ]

    # Define binding hotspot (residues 10-20)
    residues_hotspot = [residues_target[i] for i in range(10, 20)]
    target_chain = bg.Chain(residues=residues_target)

    # Create a random binder sequence
    binder_length = 10
    binder_sequence = ''.join([
        random.choice(list(bg.constants.aa_dict.keys()))
        for _ in range(binder_length)
    ])

    residues_binder = [
        bg.Residue(name=aa, chain_ID='Bin', index=i, mutable=True)
        for i, aa in enumerate(binder_sequence)
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # =========================================================================
    # Define the FoldingOracle using Protenix
    # =========================================================================
    # NOTE: Unlike ESMFold, Protenix does NOT need glycine linkers or
    # positional encoding skips for multi-chain complexes.
    # Each chain is handled as a separate entity natively.
    protenix_config = {
        'model_name': 'protenix_mini_default_v0.5.0',
        'n_sample': 1,       # Number of structure samples per prediction
        'n_cycle': 10,       # Number of pairformer recycling cycles
        'n_step': 200,       # Number of diffusion steps
        'use_msa': False,    # Disable MSA for speed in design loops
        'seed': 101,
        'dtype': 'bf16',
    }
    protenix = bg.oracles.ProtenixOracle(use_modal=use_modal, config=protenix_config)

    # =========================================================================
    # Define energy terms
    # =========================================================================
    # All energy terms work identically with ProtenixOracle as with ESMFold,
    # since ProtenixFoldResult provides the same fields (local_plddt, ptm, pae).
    energy_terms = [
        # Global confidence
        bg.energies.PTMEnergy(
            oracle=protenix,
            weight=1.0,
        ),
        # Per-residue confidence
        bg.energies.OverallPLDDTEnergy(
            oracle=protenix,
            weight=1.0,
        ),
        # Hydrophobic core packing
        bg.energies.HydrophobicEnergy(
            oracle=protenix,
            weight=5.0,
        ),
        # Interface quality: PAE between hotspot and binder
        bg.energies.PAEEnergy(
            oracle=protenix,
            residues=[residues_hotspot, residues_binder],
            weight=5.0,
        ),
        # Physical proximity of binder to hotspot
        bg.energies.SeparationEnergy(
            oracle=protenix,
            residues=[residues_hotspot, residues_binder],
            weight=1.0,
        ),
    ]

    # Define the state with both chains
    state = bg.State(
        name='protenix_binder_design',
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
    )

    # Define system and minimizer
    initial_system = bg.System(states=[state])

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),
        high_temperature=2,
        low_temperature=0.1,
        n_cycles=2,
        n_steps_low=2,
        n_steps_high=2,
        callbacks=[
            bg.callbacks.DefaultLogger(log_interval=1),
            bg.callbacks.FoldingLogger(folding_oracle=protenix, log_interval=50),
        ],
    )

    best_system = minimizer.minimize_system(system=initial_system)

    return best_system


if __name__ == '__main__':
    run_protenix_binder()
