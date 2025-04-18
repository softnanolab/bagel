import numpy as np
import desprot as dp
import desprot.energies as dpe


sequence = np.random.choice(list(dp.constants.aa_dict.keys()), size=100)
residues = [dp.Residue(name=aa, chain_ID="A", index=i, mutable=True) for i, aa in enumerate(sequence)]

state = dp.State(
    chains = [dp.Chain(residues)],
    energy_terms = [dpe.PTMEnergy(), dpe.OverallPLDDTEnergy(), dpe.HydrophobicEnergy()],
    energy_terms_weights = [1.0, 1.0, 1.0],
    state_ID = "state_A",
    verbose = True
)

minimizer = dp.minimizer.SimulatedAnnealing(
    folder=dp.folding.ESMFolder(use_modal=True),
    mutator=dp.mutation.Canonical(),
    initial_temperature = 0.2,
    final_temperature = 0.02,
    n_steps = 2_000,
    log_frequency = 5,
    experiment_name = 'annealing_hallucination',
)

minimizer.minimize_system(dp.System([state]))
