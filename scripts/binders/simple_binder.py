import random
import bagel as bg
import os
from typing import Any

def run_simple_binder() -> Any:
    # Get the value of an environment variable
    use_modal = True if os.getenv('USE_MODAL', 'True').lower() in ('true', '1', 'yes') else False

    # Check
    print(f'Whether to use modal: {use_modal}')

    # Define the target protein
    # This sequence comes from a PDB of the interleukin-8 protein
    # >1IL8_1|Chains A, B|INTERLEUKIN-8|Homo sapiens (9606)
    target_sequence = 'SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS'

    # Define the mutability of the residues, all immutable in this case since this is the target sequence
    mutability = [False for _ in range(len(target_sequence))]

    # Define a chain providing a list of residues
    residues_target = [
        bg.Residue(name=aa, chain_ID='Maxi', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]

    # Define residues in the hotspot where you want to bind. Here we choose those between residues 10-20
    residues_hotspot = [residues_target[i] for i in range(10, 20)]
    target_chain = bg.Chain(residues=residues_target)

    # For the binder, start with a random sequence of 10 residues
    binder_length = 10
    binder_sequence = ''.join([random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length)])

    # Define the mutability of the residues, all mutable in this case since this is the design sequence
    mutability = [True for _ in range(len(target_sequence))]

    # Define the chain
    residues_binder = [
        bg.Residue(name=aa, chain_ID='Stef', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(binder_sequence, mutability))
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # Define the FoldingOracle
    # See https://openreview.net/forum?id=g8S53BmXE6 for linker parameter tuning
    config = {
        'glycine_linker': 'GGGG',
        'position_ids_skip': 1024,
    }
    esmfold = bg.oracles.ESMFold(use_modal=use_modal, config=config)

    # Define the energy terms to be applied to the chain. apply them to residues, and specify the weight
    energy_terms = [
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            weight=5.0,
        ),
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[residues_hotspot, residues_binder],
            weight=5.0,
        ),
        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=[residues_hotspot, residues_binder],
            weight=1.0,
        ),
    ]

    # Define the state
    state = bg.State(
        name='state_A',
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
    )

    # Define the system
    initial_system = bg.System(states=[state])

    # Define the minimizer
    # Simulated tempering does n_steps_low at a low temperature (enhancing local minimization),
    # and n_steps_high at a high temperature (exploring the space)
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1), # cannot add/remove residues, only substitutes for different amino acid types
        high_temperature=2,
        low_temperature=0.1,
        n_cycles=10,
        n_steps_low=100,
        n_steps_high=20,
        log_frequency=50,
    )

    best_system = minimizer.minimize_system(system=initial_system)

    return best_system

if __name__ == '__main__':
    run_simple_binder()
