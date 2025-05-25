import random
import bagel as bg
import os
from typing import Any

def run_simple_binder() -> Any:
    # Get the value of an environment variable
    use_modal = True if os.getenv('USE_MODAL', 'True').lower() in ('true', '1', 'yes') else False

    # Check
    print(f'Whether to use modal: {use_modal}')

    # PART 1: Define the target protein

    # First define a random sequence of amino acids selecting randomly from the 20 amino acids
    # This sequence for the target is imported from the PDB. It is the sequence of the interleukin-8 protein
    # The sequence is the following:
    # >1IL8_1|Chains A, B|INTERLEUKIN-8|Homo sapiens (9606)

    target_sequence = 'SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS'
    target_sequence = target_sequence[:20]

    # Now define the mutability of the residues, all immutable in this case since this is the target sequence
    mutability = [False for _ in range(len(target_sequence))]

    # Now define the chain
    residues_target = [
        bg.Residue(name=aa, chain_ID='Maxi', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]

    # Now define residues in the hotspot where you want to bind. Here we choose those between residues 40-60
    residues_hotspot = [residues_target[i] for i in range(10, 20)]
    target_chain = bg.Chain(residues=residues_target)

    # For the binder, start with a random sequence of amino acids selecting randomly from the 30 amino acids
    binder_length = 10
    binder_sequence = ''.join([random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length)])

    # Now define the mutability of the residues, all mutable in this case since this is the design sequence
    mutability = [True for _ in range(len(target_sequence))]

    # Now define the chain
    residues_binder = [
        bg.Residue(name=aa, chain_ID='Stef', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(binder_sequence, mutability))
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # Now define the folding algorithm
    config = {
        'output_pdb': True,
        'output_cif': False,
        'glycine_linker': 'GGGG',
        'position_ids_skip': 100,
    }
    esmfold = bg.oracles.ESMFold(use_modal=use_modal, config=config)

    # Now define the energy terms to be applied to the chain. apply them to residues, and specify the weight
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
    ]

    # Now define the state
    state = bg.State(
        name='state_A',
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
    )

    # Now define the system
    initial_system = bg.System(states=[state])

    # Now define the minimizer
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),
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
