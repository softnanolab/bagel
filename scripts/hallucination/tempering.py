import numpy as np
import bricklane as bl


sequence = np.random.choice(list(bl.constants.aa_dict.keys()), size=50)
residues = [bl.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

state = bl.State(
    chains=[bl.Chain(residues)],
    energy_terms=[bl.energies.PTMEnergy(), bl.energies.OverallPLDDTEnergy(), bl.energies.HydrophobicEnergy()],
    energy_terms_weights=[1.0, 1.0, 5.0],
    name='state_A',
    verbose=True,
)

minimizer = bl.minimizer.SimulatedTempering(
    folder=bl.folding.ESMFolder(use_modal=True),
    mutator=bl.mutation.Canonical(),
    high_temperature=1,
    low_temperature=0.1,
    n_steps_high=50,
    n_steps_low=200,
    n_cycles=40,
    log_frequency=10,
    experiment_name='tempering_hallucination',
)

minimizer.minimize_system(bl.System([state]))
