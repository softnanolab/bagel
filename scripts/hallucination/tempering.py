import numpy as np
import desprot as dp
import desprot.energies as dpe


sequence = np.random.choice(list(dp.constants.aa_dict.keys()), size=50)
residues = [dp.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]

state = dp.State(
    chains=[dp.Chain(residues)],
    energy_terms=[dpe.PTMEnergy(), dpe.OverallPLDDTEnergy(), dpe.HydrophobicEnergy()],
    energy_terms_weights=[1.0, 1.0, 5.0],
    state_ID='state_A',
    verbose=True,
)

minimizer = dp.minimizer.SimulatedTempering(
    folder=dp.folding.ESMFolder(use_modal=True),
    mutator=dp.mutation.Canonical(),
    high_temperature=1,
    low_temperature=0.1,
    n_steps_high=50,
    n_steps_low=200,
    n_cycles=40,
    log_frequency=10,
    experiment_name='tempering_hallucination',
)

minimizer.minimize_system(dp.System([state]))
