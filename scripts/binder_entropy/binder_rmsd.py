import random
import bagel as bg
import os
from typing import Any


def run_binder_rmsd() -> Any:
    """
    Design a binder that is already pre-organised in its bound shape.

    Binding a flexible binder costs conformational entropy, because the binder is frozen into a single shape.
    A binder that folds alone into (nearly) the same structure it has in the complex pays much less. To reward this,
    the design uses two states that share the same binder chain:

    - 'complex':  binder + target, co-folded. Holds the usual binding terms, plus BinderRMSDEnergy.
    - 'isolated': binder alone. It is the *reference state* of BinderRMSDEnergy, which measures the RMSD between
                  the binder in the two states.

    Because the binder Chain object is shared, every mutation of the binder is seen by both states.
    """
    # Get the backend from an environment variable (default: modal)
    backend = os.getenv('BAGEL_BACKEND', 'modal')
    print(f'Backend: {backend}')

    # Define the target protein, the interleukin-8 sequence used in scripts/binders/simple_binder.py
    target_sequence = 'SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS'
    residues_target = [
        bg.Residue(name=aa, chain_ID='Maxi', index=i, mutable=False) for i, aa in enumerate(target_sequence)
    ]
    target_chain = bg.Chain(residues=residues_target)

    # Hotspot on the target where we want to bind: residues 10-20
    residues_hotspot = [residues_target[i] for i in range(10, 20)]

    # The binder starts from a random sequence, and all its residues are mutable
    binder_length = 10
    binder_sequence = ''.join(random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length))
    residues_binder = [
        bg.Residue(name=aa, chain_ID='Stef', index=i, mutable=True) for i, aa in enumerate(binder_sequence)
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # Define the FoldingOracle. One oracle is enough here: the same one folds the complex and the isolated binder.
    # See https://openreview.net/forum?id=g8S53BmXE6 for linker parameter tuning
    config = {
        'glycine_linker': 'GGGG',
        'position_ids_skip': 1024,
    }
    esmfold = bg.oracles.ESMFold(backend=backend, config=config)

    # Reference state: the binder alone. It needs at least one energy term, since a state without any cannot be
    # evaluated. A confident isolated fold matters here: if the model has no idea what the isolated binder looks like,
    # the RMSD to it is meaningless. Note this state also adds its own energy to the total, like any other state.
    isolated_state = bg.State(
        name='isolated',
        chains=[binder_chain],
        energy_terms=[
            bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=1.0, name='isolated'),
        ],
    )

    # The complex state: the usual binder design terms, plus the RMSD between the bound and the isolated binder.
    energy_terms = [
        bg.energies.PTMEnergy(oracle=esmfold, weight=1.0),
        bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=1.0),
        bg.energies.HydrophobicEnergy(oracle=esmfold, weight=5.0),
        bg.energies.PAEEnergy(oracle=esmfold, residues=[residues_hotspot, residues_binder], weight=5.0),
        bg.energies.SeparationEnergy(oracle=esmfold, residues=[residues_hotspot, residues_binder], weight=1.0),
        bg.energies.BinderRMSDEnergy(
            oracle=esmfold,
            residues=residues_binder,  # residues to compare, all must also be in the reference state
            reference_state=isolated_state,  # where the isolated structure (and its pLDDT) come from
            plddt_scaled=True,  # weight residue i in the mean by pLDDT_i ** plddt_exponent, from the isolated fold
            plddt_exponent=2.0,  # must be > 0
            weight=1.0,
        ),
    ]
    complex_state = bg.State(
        name='complex',
        chains=[binder_chain, target_chain],
        energy_terms=energy_terms,
    )

    # Both states must be in the system. The system is deep-copied at every step, and this is what keeps the
    # reference_state of the energy term pointing to the copy of the isolated state, not to a stale one.
    initial_system = bg.System(states=[complex_state, isolated_state])

    # Simulated tempering does n_steps_low at a low temperature (enhancing local minimization),
    # and n_steps_high at a high temperature (exploring the space)
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),  # cannot add/remove residues, only substitutes amino acids
        high_temperature=2,
        low_temperature=0.1,
        n_cycles=10,
        n_steps_low=100,
        n_steps_high=20,
        callbacks=[
            bg.callbacks.DefaultLogger(log_interval=1),
            bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50),
        ],
    )

    best_system = minimizer.minimize_system(system=initial_system)

    return best_system


if __name__ == '__main__':
    run_binder_rmsd()
