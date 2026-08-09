# BAGEL script patterns

Copy the closest pattern and adapt it. Each is distilled from a real script in `scripts/`;
the cited file is the ground-truth version to re-read if anything looks off.

## Table of contents
1. Binder design (single state) — from `scripts/car/lnk_cd20_monomer.py`, `scripts/binders/simple_binder.py`
2. Selective / negative design (multi-state) — from `scripts/technical-report/selective_zinc_finger.py`
3. De novo symmetric scaffold — from `scripts/scaffolds/annealing.py`
4. Embedding-based enzyme miniaturization — from `scripts/mini-enzymes/petase.py`
5. Custom temperature schedule and chained optimizers

---

## 1. Binder design (single state)

Design a mutable binder chain against an immutable target, rewarding a confident fold and a
confident, close interface at a chosen epitope. `SimulatedTempering` alternates exploration
and refinement.

```python
from __future__ import annotations
import random
import bagel as bg
import fire

TARGET_SEQUENCE = 'SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANTEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS'
EPITOPE = range(10, 20)          # 0-based residue indices on the target to bind
AMINO_ACIDS_NO_CYS = [aa for aa in bg.constants.aa_dict if aa != 'C']


def main(backend: str = 'modal', binder_length: int = 20,
         n_cycles: int = 100, n_steps_low: int = 200, n_steps_high: int = 50) -> bg.System:
    # 1-2. Target chain (immutable) and its epitope group.
    residues_target = [bg.Residue(name=aa, chain_ID='TGT', index=i, mutable=False)
                       for i, aa in enumerate(TARGET_SEQUENCE)]
    target_chain = bg.Chain(residues=residues_target)
    residues_epitope = [residues_target[i] for i in EPITOPE]

    # Binder chain: random, fully mutable.
    binder_seq = ''.join(random.choice(AMINO_ACIDS_NO_CYS) for _ in range(binder_length))
    residues_binder = [bg.Residue(name=aa, chain_ID='BIND', index=i, mutable=True)
                       for i, aa in enumerate(binder_seq)]
    binder_chain = bg.Chain(residues=residues_binder)

    # 3. Folding oracle. Multi-chain state -> glycine linker + position_ids_skip.
    esmfold = bg.oracles.ESMFold(backend=backend,
                                 config={'glycine_linker': 'G' * 25, 'position_ids_skip': 1024})

    # 4. Energy terms: global confidence, per-group pLDDT, and the interface (PAE + separation).
    energy_terms = [
        bg.energies.PTMEnergy(oracle=esmfold, weight=0.2),
        bg.energies.PLDDTEnergy(oracle=esmfold, residues=residues_epitope, weight=1.0, name='epitope'),
        bg.energies.PLDDTEnergy(oracle=esmfold, residues=residues_binder, weight=1.0, name='binder'),
        bg.energies.PAEEnergy(oracle=esmfold, residues=[residues_epitope, residues_binder],
                              weight=2.0, name='interface'),          # two groups
        bg.energies.SeparationEnergy(oracle=esmfold, residues=[residues_epitope, residues_binder],
                                     weight=1.0, name='interface'),   # two groups
        bg.energies.HydrophobicEnergy(oracle=esmfold, residues=residues_binder, weight=1.0, name='binder'),
    ]

    # 5. State -> System.
    state = bg.State(name='bound', chains=[target_chain, binder_chain], energy_terms=energy_terms)
    system = bg.System(states=[state])

    # 6-7. Tempering minimizer.
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),
        high_temperature=1.0, low_temperature=0.1,
        n_cycles=n_cycles, n_steps_low=n_steps_low, n_steps_high=n_steps_high,
        preserve_best_system_every_n_steps=n_steps_low + n_steps_high,
        callbacks=[bg.callbacks.DefaultLogger(log_interval=1),
                   bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50)],
    )
    return minimizer.minimize_system(system=system)


if __name__ == '__main__':
    fire.Fire(main)
```

---

## 2. Selective / negative design (multi-state)

Bind the target **and avoid** an off-target. Two states share the **same mutable binder
chain**; the off-target state uses **negative weights** on its interface terms. The
minimizer lowers total system energy, so it favors a binder that engages the target and
disengages the off-target.

```python
# ... build residues_target, residues_non_target (both immutable), residues_binder (mutable),
#     residues_hotspot on the target, and one esmfold oracle as above ...

state_bound = bg.State(
    name='target', chains=[binder_chain, target_chain],
    energy_terms=[
        bg.energies.PTMEnergy(oracle=esmfold, weight=1.0),
        bg.energies.PLDDTEnergy(oracle=esmfold, residues=residues_target + residues_binder, weight=1.0),
        bg.energies.PAEEnergy(oracle=esmfold, residues=[residues_hotspot, residues_binder], weight=5.0),
        bg.energies.SeparationEnergy(oracle=esmfold, residues=[residues_hotspot, residues_binder], weight=1.0),
    ],
)

state_avoid = bg.State(
    name='off_target', chains=[binder_chain, non_target_chain],
    energy_terms=[
        bg.energies.PTMEnergy(oracle=esmfold, weight=1.0),
        bg.energies.PLDDTEnergy(oracle=esmfold, residues=residues_non_target + residues_binder, weight=1.0),
        bg.energies.PAEEnergy(oracle=esmfold, residues=[residues_non_target, residues_binder], weight=-5.0),   # discourage
        bg.energies.SeparationEnergy(oracle=esmfold, residues=[residues_non_target, residues_binder], weight=-1.0),
    ],
)

system = bg.System(states=[state_bound, state_avoid])
# ... same SimulatedTempering minimizer as pattern 1 ...
```

---

## 3. De novo symmetric scaffold

No target — generate a fully mutable chain with a target topology. Uses `SimulatedAnnealing`
and shape terms. Note `groups=` (not `residues=`) on the symmetry and secondary-structure
terms.

```python
import numpy as np
import bagel as bg

sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=190)
residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
esmfold = bg.oracles.ESMFold(backend='modal')

state = bg.State(
    name='scaffold',
    chains=[bg.Chain(residues)],
    energy_terms=[
        bg.energies.PTMEnergy(oracle=esmfold, weight=1.0),
        bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=1.0),
        bg.energies.HydrophobicEnergy(oracle=esmfold, weight=3.0, mode='surface'),
        bg.energies.RingSymmetryEnergy(oracle=esmfold, weight=1.0,
                                       groups=[residues[i*50 : i*50+40] for i in range(4)]),
        bg.energies.SecondaryStructureEnergy(oracle=esmfold, weight=1.0, structure_type='beta-sheet',
                                             groups=[residues[i*50+5 : i*50+35] for i in range(4)]),
    ],
)

minimizer = bg.minimizer.SimulatedAnnealing(
    mutator=bg.mutation.Canonical(),
    initial_temperature=0.2, final_temperature=0.05, n_steps=2000,
    experiment_name='symmetric_scaffold',
    callbacks=[bg.callbacks.DefaultLogger(log_interval=1),
               bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50)],
)
minimizer.minimize_system(system=bg.System([state]))
```

---

## 4. Embedding-based enzyme miniaturization

Keep a protein "like" a reference in embedding space while conserving catalytic residues and
letting the length shrink. Uses an **embedding oracle**, `EmbeddingsSimilarityEnergy` +
`ChemicalPotentialEnergy`, a `GrandCanonical` mutator (variable length), and a fixed-low-T
`MonteCarloMinimizer`.

```python
import numpy as np
import bagel as bg
import fire

def main(backend: str = 'modal', temperature: float = 1e-4, n_steps: int = 100000) -> bg.System:
    full_sequence = bg.utils.get_sequence_from_pdb_id(pdb_id='5xjh', sequence_index=0)
    conserved_idx = {86, 159, 184, 205, 213, 217, 236}   # 0-based catalytic/functional residues
    mutability = [i not in conserved_idx for i in range(len(full_sequence))]

    residues = [bg.Residue(name=aa, chain_ID='AAA', index=i, mutable=mut)
                for i, (aa, mut) in enumerate(zip(full_sequence, mutability))]
    chain = bg.Chain(residues=residues)
    conserved_residues = [r for r in residues if not r.mutable]

    # Capture the reference embeddings of the conserved residues BEFORE optimizing.
    esm2 = bg.oracles.ESM2(backend=backend, config={'model_name': 'esm2_t33_650M_UR50D'})
    result = esm2.embed(chains=[chain])
    reference_embeddings = result.embeddings[~np.array(mutability)]

    state = bg.State(
        name='mini',
        chains=[chain],
        energy_terms=[
            bg.energies.EmbeddingsSimilarityEnergy(oracle=esm2, residues=conserved_residues,
                                                   reference_embeddings=reference_embeddings, weight=1.0),
            bg.energies.ChemicalPotentialEnergy(oracle=esm2, weight=1e-3, name='chemical_potential'),
        ],
    )

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=bg.mutation.GrandCanonical(
            n_mutations=1,
            move_probabilities={'substitution': 0.70, 'addition': 0.15, 'removal': 0.15}),
        temperature=temperature, n_steps=n_steps,
    )
    return minimizer.minimize_system(system=bg.System(states=[state]))


if __name__ == '__main__':
    fire.Fire(main)
```

---

## 5. Custom temperature schedule and chained optimizers

The two mechanisms the user asked for, shown together: shape any schedule as an array, then
chain a broad MC exploration phase into an annealing refinement phase.

```python
import numpy as np
import bagel as bg

# ... build `initial_system` and `esmfold` as in pattern 1 ...

# Phase 1 — custom schedule: hold hot to explore, then exponentially cool.
schedule = np.concatenate([np.full(500, 2.0), 2.0 * np.exp(-np.linspace(0, 4, 1500))])
explored = bg.minimizer.MonteCarloMinimizer(
    mutator=bg.mutation.Canonical(n_mutations=2),
    temperature=schedule,           # length must equal n_steps
    n_steps=len(schedule),
    experiment_name='phase1_explore',
    callbacks=[bg.callbacks.DefaultLogger(log_interval=10)],
).minimize_system(system=initial_system)

# Phase 2 — refine from phase 1's best design with simulated annealing.
refined = bg.minimizer.SimulatedAnnealing(
    mutator=bg.mutation.Canonical(n_mutations=1),
    initial_temperature=0.5, final_temperature=0.02, n_steps=2000,
    experiment_name='phase2_anneal',
    callbacks=[bg.callbacks.DefaultLogger(log_interval=10),
               bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50)],
).minimize_system(system=explored)     # <-- feeds phase 1's best system forward
```

Distinct `experiment_name`s keep each phase's logs in their own folder. You can chain any
number of phases this way — e.g. tempering to find a basin, then a short low-T MC polish.
