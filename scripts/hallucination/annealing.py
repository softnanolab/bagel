import numpy as np
import bricklane as bl

sequence = np.random.choice(list(bl.constants.aa_dict.keys()), size=100)
residues = [bl.Residue(name=aa, chain_ID="A", index=i, mutable=True) for i, aa in enumerate(sequence)]

state = bl.State(
    chains = [bl.Chain(residues)],
    energy_terms = [bl.energies.PTMEnergy(), bl.energies.OverallPLDDTEnergy(), bl.energies.HydrophobicEnergy()],
    energy_terms_weights = [1.0, 1.0, 1.0],
    state_ID = "state_A",
    verbose = True
)

minimizer = bl.minimizer.SimulatedAnnealing(
    folder=bl.folding.ESMFolder(use_modal=True),
    mutator=bl.mutation.Canonical(),
    initial_temperature = 0.2,
    final_temperature = 0.02,
    n_steps = 2_000,
    log_frequency = 5,
    experiment_name = 'annealing_hallucination',
)

minimizer.minimize_system(bl.System([state]))
