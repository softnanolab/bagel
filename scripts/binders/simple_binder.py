import random
import bricklane as bl
import os
import modal

with modal.enable_output():
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
        bl.Residue(name=aa, chain_ID='Maxi', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]
    # Now define residues in the hotspot where you want to bind. Here we choose those between residues 40-60
    residues_hotspot = [residues_target[i] for i in range(10, 20)]
    target_chain = bl.Chain(residues=residues_target)

    # For the binder, start with a random sequence of amino acids selecting randomly from the 30 amino acids
    binder_length = 10
    binder_sequence = ''.join([random.choice(list(bl.constants.aa_dict.keys())) for _ in range(binder_length)])
    # Now define the mutability of the residues, all mutable in this case since this is the design sequence
    mutability = [True for _ in range(len(target_sequence))]
    # Now define the chain
    residues_binder = [
        bl.Residue(name=aa, chain_ID='Stef', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(binder_sequence, mutability))
    ]
    binder_chain = bl.Chain(residues=residues_binder)

    # Now define the energy terms to be applied to the chain. In this example, all terms apply to all residues
    energy_terms = [
        bl.energies.PTMEnergy(),
        bl.energies.OverallPLDDTEnergy(),
        bl.energies.HydrophobicEnergy(),
        bl.energies.AlignmentErrorEnergy(
            group_1_residues=residues_hotspot,
            group_2_residues=residues_binder,
        ),
    ]

    # Now define the energy term weights
    energy_terms_weights = [1.0, 1.0, 5.0, 5.0]

    # Now define the state
    state = bl.State(
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
        energy_terms_weights=energy_terms_weights,
        name='state_A',
    )

    # Now define the system
    initial_system = bl.System(states=[state])

    # Now define the folding algorithm, run locally not on modal
    config = {
        'output_pdb': True,
        'output_cif': False,
        'glycine_linker': 'GGGG',
        'position_ids_skip': 100,
    }
    esmfold = bl.folding.ESMFolder(
        use_modal=use_modal, config=config
    )  # Looks like it is calling modal regardless of the flag, why?

    # Now define the minimizer
    minimizer = bl.minimizer.SimulatedTempering(
        all_folding_algorithms={'EsmFold': esmfold},
        mutation_protocol=bl.mutation.Canonical(),
        algorithm_name='EsmFold',
        max_mutations_per_step=1,
        high_temperature=2,
        low_temperature=0.1,
        n_steps=10,
        n_steps_low=100,
        n_steps_high=20,
        preserve_detailed_balance=False,
        log_frequency=50,
    )

    best_system = minimizer.minimize_system(system=initial_system)
