import numpy as np
import bagel as bg


sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=190)
residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

state = bg.State(
    chains=[bg.Chain(residues)],
    energy_terms=[
        bg.energies.PTMEnergy(),
        bg.energies.OverallPLDDTEnergy(),
        bg.energies.HydrophobicEnergy(surface_only=True),
        bg.energies.RingSymmetryEnergy([residues[i*50 : (i*50)+40] for i in range(4)]),
        # 5 residues on either side of each symmetry group unconstrained to allow for chain flexibility
        bg.energies.SecondaryStructureEnergy([residues[(i*50)+5 : (i*50)+35] for i in range(4)], 'beta-sheet'),
    ],
    energy_terms_weights=[1.0, 1.0, 3.0, 1.0, 1.0],
    name='state_A',
)

minimizer = bg.minimizer.SimulatedAnnealing(
    all_folding_algorithms={'EsmFold': bg.folding.ESMFolder(use_modal=True)},
    mutation_protocol=bg.mutation.Canonical(),
    max_mutations_per_step=1,
    initial_temperature=0.2,
    final_temperature=0.05,
    n_steps=2_000,
    log_frequency=1,
    experiment_name='annealing_scaffold_with_joiners',
)

minimizer.minimize_system(system=bg.System([state]))
