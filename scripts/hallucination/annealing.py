import numpy as np
import bagel as bg

sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=100)
residues = [bg.Residue(name=aa, chain_ID="A", index=i, mutable=True) for i, aa in enumerate(sequence)]

esmfold = bg.folding.ESMFold(use_modal=True)

state = bg.State(
    chains = [bg.Chain(residues)],
    energy_terms = [bg.energies.PTMEnergy(
        oracle=esmfold,
        weight=1.0,
    ), bg.energies.OverallPLDDTEnergy(
        oracle=esmfold,
        weight=1.0,
    ), bg.energies.HydrophobicEnergy(
        oracle=esmfold,
        weight=1.0,
    )],
    name = "state_A",
    verbose = True
)

minimizer = bg.minimizer.SimulatedAnnealing(
    mutator=bg.mutation.Canonical(),
    initial_temperature = 0.2,
    final_temperature = 0.02,
    n_steps = 2_000,
    log_frequency = 5,
    experiment_name = 'annealing_hallucination',
)

minimizer.minimize_system(bg.System([state]))
