import numpy as np
import desprot as dp
import desprot.energies as dpe


sequence = np.random.choice(list(dp.constants.aa_dict.keys()), size=190)
residues = [dp.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

state = dp.State(
    chains=[dp.Chain(residues)],
    energy_terms=[
        dpe.PTMEnergy(),
        dpe.OverallPLDDTEnergy(),
        dpe.HydrophobicEnergy(surface_only=True),
        dpe.RingSymmetryEnergy([residues[i*50 : (i*50)+40] for i in range(4)]),
        # 5 residues on either side of each symmetry group unconstrained to allow for chain flexibility
        dpe.SecondaryStructureEnergy([residues[(i*50)+5 : (i*50)+35] for i in range(4)], 'beta-sheet'),
    ],
    energy_terms_weights=[1.0, 1.0, 3.0, 1.0, 1.0],
    state_ID='state_A',
    verbose=True,
)

minimizer = dp.minimizer.SimulatedAnnealing(
    all_folding_algorithms={'EsmFold': dp.folding.ESMFolder(use_modal=True)},
    mutation_protocol=dp.mutation.Canonical(),
    max_mutations_per_step=1,
    initial_temperature=0.2,
    final_temperature=0.05,
    n_steps=2_000,
    log_frequency=1,
    experiment_name='annealing_scaffold_with_joiners',
)

minimizer.minimize_system(system=dp.System([state]))
