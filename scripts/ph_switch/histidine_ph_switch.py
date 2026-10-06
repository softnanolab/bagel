"""
De novo design of an acid-labile protein: folded near pH 7.4, unfolded at pH ~6.

PHYSICAL IDEA
-------------
Histidine is the only canonical side chain whose pKa (~6.0-6.5 free in solution)
sits between pH 6 and pH 7.4. The switch is built out of one thermodynamic fact:

  - At pH 7.4 the imidazole is neutral. A neutral imidazole is a perfectly
    acceptable core residue: it is roughly as apolar as tyrosine and packs well.
  - At pH 6 it protonates to the imidazolium cation. Burying a *charged* group in
    a low-dielectric core with no counter-ion and no hydrogen-bond acceptor costs
    a large desolvation penalty. That penalty is paid only by the folded state,
    because in the unfolded state the same cation is fully solvated.
  - Each buried histidine therefore subtracts from the folding free energy as the
    pH drops. Several of them, mutually repelling, subtract a lot.

So the design objective is: a well-folded, compact monomer whose core contains
several histidines that are (a) deeply buried and (b) NOT hydrogen bonded to a
carboxylate. Point (b) matters more than it looks: a His-Asp or His-Glu dyad
raises the histidine pKa and stabilises the protonated form, which would make the
protein *more* stable at low pH. That is the opposite switch.

IMPORTANT LIMITATION, STATED UP FRONT
-------------------------------------
None of BAGEL's folding oracles (ESMFold, ESMFold2, Chai1, Boltz2) take pH as an
input. There is no way to ask them to "fold this at pH 6". The pH dependence here
is therefore *encoded by construction* (buried, unpaired histidines) rather than
predicted. Nothing in this script estimates the transition midpoint, so a design
that satisfies every term might switch at pH 4.5 or at pH 7.0 rather than at 6.

A deliberate omission: an earlier version of this script carried a
`ProtonationMimicEnergy` term that folded a histidine-to-arginine copy of the
sequence and penalised it for folding well, as surrogate negative design against
the charged state. It was removed, because ESMFold has no electrostatics in it.
Asking whether the arginine variant folds worse is not an electrostatics
question; it asks whether that sequence looks less like a natural folded protein
to a language model. The two overlap loosely and the score cannot tell you which
one you measured. It was also largely redundant with the burial and carboxylate
terms, it doubled the oracle cost per Monte Carlo step, and it created a
perverse incentive: one cheap way to make a variant fold badly is to make the
parent barely fold at all.

That comparison survives as a post-hoc ranking signal in `filter_designs.py`,
where it costs one extra fold per finished candidate rather than one per step.
For an actual predicted midpoint, run PROPKA or a Poisson-Boltzmann calculation
on the designed structures. Those compute the desolvation and charge-charge
terms this script only gestures at. Everything still needs wet-lab confirmation
by pH-dependent circular dichroism or tryptophan fluorescence.

Run with:
    python scripts/ph_switch/histidine_ph_switch.py
Set BAGEL_BACKEND=apptainer to run the oracle locally instead of on Modal.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import bagel as bg
from bagel.energies import BuriedHistidineEnergy, HistidineCarboxylateContactEnergy


# ----------------------------------------------------------------------------
# The design run.
# ----------------------------------------------------------------------------
def build_system(length: int = 100, seed: int | None = None) -> tuple[bg.System, Any]:
    """Assemble the single-state system for the acid-labile design."""
    rng = np.random.default_rng(seed)
    backend = os.getenv('BAGEL_BACKEND', 'modal')
    print(f'Backend: {backend}')

    # Start from a random sequence. Cysteine is excluded by the default mutation
    # bias; a disulfide would clamp the fold shut and blunt the switch.
    alphabet = [aa for aa in bg.constants.aa_dict.keys() if aa != 'C']
    sequence = rng.choice(alphabet, size=length)
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    chain = bg.Chain(residues)

    esmfold = bg.oracles.ESMFold(backend=backend)

    energy_terms = [
        # --- Fold quality: without these the switch terms design a disordered
        # peptide that is "unfolded at pH 6" only because it is unfolded always.
        bg.energies.PTMEnergy(oracle=esmfold, weight=2.0),
        bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=2.0),
        bg.energies.GlobularEnergy(oracle=esmfold, weight=0.5),
        # --- Keep a real hydrophobic core and a polar surface. The histidines
        # are a perturbation on an otherwise normal globular protein, not a
        # replacement for one.
        bg.energies.HydrophobicEnergy(oracle=esmfold, mode='surface', weight=2.0),
        # NOTE: this term and BuriedHistidineEnergy below pull against each other.
        # Histidine's Kyte-Doolittle index is -3.2, so every buried histidine makes
        # the burial-weighted hydropathy worse. That tension is deliberate -- it is
        # what stops the core becoming all histidine -- but it means the ratio of
        # these two weights is the single most important knob in this script. Sweep
        # it (0.1/8.0 through 1.5/2.0) before trusting any one run.
        bg.energies.HydropathyEnergy(oracle=esmfold, mode='core', weight=-0.5),
        # --- The switch itself.
        BuriedHistidineEnergy(oracle=esmfold, target_count=6, sasa_cutoff=0.15, weight=4.0),
        # --- Stop the optimiser building pKa-raising His/Asp dyads, which would
        # invert the switch. Weighted above the per-histidine burial reward.
        HistidineCarboxylateContactEnergy(oracle=esmfold, distance_cutoff=4.5, max_contacts=4, weight=3.0),
        # No surrogate term for the protonated state: see the module docstring.
        # The charge-mimic comparison lives in filter_designs.py instead, where
        # it runs once per finished candidate rather than once per MC step.
    ]

    state = bg.State(name='folded_neutral_pH', chains=[chain], energy_terms=energy_terms)
    return bg.System([state]), esmfold


def main() -> Any:
    system, esmfold = build_system(length=100, seed=0)

    # Enrich histidine in the proposal distribution. Uniform sampling proposes H
    # about 5% of the time, so reaching six *buried* histidines by chance is slow.
    # This biases proposals only -- acceptance is still decided by the energy, so
    # it changes mixing speed, not the target distribution's optimum.
    bias = dict(bg.constants.mutation_bias_no_cystein)
    bias['H'] = 4.0 * bias['H']
    total = sum(bias.values())
    bias = {aa: p / total for aa, p in bias.items()}

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1, mutation_bias=bias),
        high_temperature=1.0,
        low_temperature=0.1,
        n_cycles=100,
        n_steps_low=200,
        n_steps_high=50,
        experiment_name='ph6_histidine_switch',
        callbacks=[
            bg.callbacks.DefaultLogger(log_interval=1),
            bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50),
        ],
    )

    best_system = minimizer.minimize_system(system=system)
    print('Best sequence:', best_system.states[0].chains[0].sequence)
    return best_system


if __name__ == '__main__':
    main()
