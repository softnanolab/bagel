import numpy as np
import bricklane as bl


sequence = np.random.choice(list(bl.constants.aa_dict.keys()), size=190)
residues = [bl.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

state = bl.State(
    chains=[bl.Chain(residues)],
    energy_terms=[
        bl.energies.PTMEnergy(),
        bl.energies.OverallPLDDTEnergy(),
        bl.energies.HydrophobicEnergy(surface_only=True),
        bl.energies.RingSymmetryEnergy([residues[i*50 : (i*50)+40] for i in range(4)]),
        # 5 residues on either side of each symmetry group unconstrained to allow for chain flexibility
        bl.energies.SecondaryStructureEnergy([residues[(i*50)+5 : (i*50)+35] for i in range(4)], 'beta-sheet'),
    ],
    energy_terms_weights=[1.0, 1.0, 3.0, 1.0, 1.0],
    name='state_A',
    verbose=True,
)

minimizer = bl.minimizer.SimulatedAnnealing(
    all_folding_algorithms={'EsmFold': bl.folding.ESMFolder(use_modal=True)},
    mutation_protocol=bl.mutation.Canonical(),
    max_mutations_per_step=1,
    initial_temperature=0.2,
    final_temperature=0.05,
    n_steps=2_000,
    log_frequency=1,
    experiment_name='annealing_scaffold_with_joiners',
)

minimizer.minimize_system(system=bl.System([state]))
