import random
import bagel as bg
import os
from typing import Any

def run_selective_binder() -> Any:
    # Get the value of an environment variable
    use_modal = True if os.getenv('USE_MODAL', 'True').lower() in ('true', '1', 'yes') else False

    # Check
    print(f'Whether to use modal: {use_modal}')

    ##################
    # Define the target and non-target protein (the one the binder should avoid)
    ##################

    # Define the target protein
    # These sequences comes from the PDB, see details below
    # >1AAY_3|Chain C[auth A]|PROTEIN (ZIF268 ZINC FINGER PEPTIDE)|Mus musculus (10090)
    # MERPYACPVESCDRRFSRSDELTRHIRIHTGQKPFQCRICMRNFSRSDHLTTHIRTHTGEKPFACDICGRKFARSDERKRHTKIHLRQKD
    target_sequence = 'MERPYACPVESCDRRFSRSDELTRHIRIHTGQKPFQCRICMRNFSRSDHLTTHIRTHTGEKPFACDICGRKFARSDERKRHTKIHLRQKD'
    # >1ZR9_1|Chain A|Zinc finger protein 593|Homo sapiens (9606)
    # GHHHHHHLEKAKRRRPDLDEIHRELRPQGSARPQPDPNAEFDPDLPGGGLHRCLACARYFIDSTNLKTHFRSKDHKKRLKQLSVEPYSQEEAERAAGMGSYVPPRRLAVPTEVSTEVPEMDTST
    non_target_sequence ='GHHHHHHLEKAKRRRPDLDEIHRELRPQGSARPQPDPNAEFDPDLPGGGLHRCLACARYFIDSTNLKTHFRSKDHKKRLKQLSVEPYSQEEAERAAGMGSYVPPRRLAVPTEVSTEVPEMDTST'

    # Define the mutability of the residues, all immutable in this case since these are the potential targets
    mutability_target = [False for _ in range(len(target_sequence))]
    mutability_non_target = [False for _ in range(len(non_target_sequence))]

    # Define a chain providing a list of residues
    residues_target = [
        bg.Residue(name=aa, chain_ID='Maxi', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability_target))
    ]
    residues_non_target = [
        bg.Residue(name=aa, chain_ID='Maxi', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(non_target_sequence, mutability_non_target))
    ]
    # Build the chains from the residues
    target_chain = bg.Chain(residues=residues_target)
    non_target_chain = bg.Chain(residues=residues_non_target)

    # Define residues in the hotspot where you want to bind.
    # Here we choose the last ten of the target sequence
    residues_hotspot = [residues_target[i] for i in range( len( target_sequence ) - 10, len( target_sequence ) )]

    ##################
    # Define the binder
    ##################
    # For the binder, start with a random sequence of 15 residues
    binder_length = 15
    binder_sequence = ''.join([random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length)])

    # Define the mutability of the residues, all mutable in this case since this is the design sequence
    mutability_binder = [True for _ in range(binder_length)]

    # Define the chain
    residues_binder = [
        bg.Residue(name=aa, chain_ID='bind', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(binder_sequence, mutability_binder))
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    ##################
    # Define the FoldingOracle
    # See https://openreview.net/forum?id=g8S53BmXE6 for linker parameter tuning
    config = {
        'glycine_linker': 'G'*25, # 25 Glycine residues as a linker
        'position_ids_skip': 1024,
    }
    esmfold = bg.oracles.ESMFold(use_modal=use_modal, config=config)

    ################
    # Define the energy terms that define the two states of the system
    # State A: binder with the target protein
    # State B: binder with the non-target protein
    ################

    # Define the energy terms to be applied in state_A
    energy_terms_target = [
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_target + residues_binder,
            weight=1.0,
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

    # Define the state where the binder is designed to bind to the target
    state_bound = bg.State(
        name='state_A',
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms_target,
    )

    # Define the energy terms that define state_B
    energy_terms_non_target = [
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_target + residues_binder,
            weight=1.0,
        ),
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[residues_non_target, residues_binder],
            weight= -5.0, # Use a negative weight to discourage binding to the non-target
        ),
        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=[residues_non_target, residues_binder],
            weight=-1.0, # Use a negative weight to discourage binding to the non-target
        ),
    ]

    # Define the state
    state_unbound = bg.State(
        name='state_B',
        chains=[binder_chain, non_target_chain],
        energy_terms=energy_terms_non_target,
    )

    # Define the system: this is a collection of states. The minimizer will explore the space of these states.
    # The minimizer will try to minimize the total energy of the entire SYSTEM. If the energy terms chosen are
    # correct, the minimization will find a binder that binds to the target protein
    # and does not bind to the non-target protein.
    initial_system = bg.System(states=[state_bound, state_unbound])

    # Define the minimizer
    # Simulated tempering does n_steps_low at a low temperature (enhancing local minimization),
    # and n_steps_high at a high temperature (exploring the space)
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1), # cannot add/remove residues, only substitutes for different amino acid types
        high_temperature=1.0,
        low_temperature=0.1,
        n_cycles=100,
        n_steps_low=200,
        n_steps_high=50,
        log_frequency=50,
    )

    best_system = minimizer.minimize_system(system=initial_system)

    return best_system

if __name__ == '__main__':
    run_selective_binder()
