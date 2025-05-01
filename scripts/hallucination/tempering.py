import numpy as np
import bagel as bg


sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=50)
residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

state = bg.State(
    chains=[bg.Chain(residues)],
    energy_terms=[bg.energies.PTMEnergy(), bg.energies.OverallPLDDTEnergy(), bg.energies.HydrophobicEnergy()],
    energy_terms_weights=[1.0, 1.0, 5.0],
    name='state_A',
)

minimizer = bg.minimizer.SimulatedTempering(
    folder=bg.folding.ESMFolder(use_modal=True),
    mutator=bg.mutation.Canonical(),
    high_temperature=1,
    low_temperature=0.1,
    n_steps_high=50,
    n_steps_low=200,
    n_cycles=40,
    log_frequency=10,
    experiment_name='tempering_hallucination',
)

minimizer.minimize_system(bg.System([state]))
