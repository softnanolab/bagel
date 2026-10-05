# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness. Generated: 2025-01-XX.
# -----------------------------------------------------------------------------
"""
Generic binder (60-100 residues) vs TNF-alpha monomer non-interface region.

Design a multi-valent binder that:
  1. Binds to the non-interface region of TNF-alpha monomers (epitope)
  2. Contains buried histidines for pH-dependent unfolding
  3. Has no His-Asp/Glu pairs that would raise histidine pKa
  4. Works against both human and mouse TNF-alpha

The epitope is pre-computed by identify_epitope.py: residues NOT within 9Å
of other chains in the TNF trimer.

Energy terms:
  - Fold quality (PTM, pLDDT, compactness)
  - Buried histidines in the binder core (saturating at 6)
  - Prevents His-carboxylate pairs that would raise pKa
  - ipSAE binding energy (minimize PAE between binder and epitope)

Optimizer: Simulated Tempering (high_T=0.2, low_T=0.02)

Usage:
  python binder_generic_monomer_epitope.py --backend modal --seed 0
"""

from __future__ import annotations

import os
import sys
import numpy as np
import bagel as bg

# Import custom utilities
sys.path.insert(0, '.')
from utils import ipSAEEnergy, load_epitope_residues

# TNF-alpha sequences
TNF_HUMAN = 'VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL'
TNF_MOUSE = 'LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL'

def make_states(binder_chain, esmfold):
    """Create states for human and mouse TNF targeting."""
    states = []
    
    for species, tnf_seq in [('human', TNF_HUMAN), ('mouse', TNF_MOUSE)]:
        # Load epitope residues (non-interface)
        epitope_file = f'tnf_{species}_epitope.txt'
        try:
            with open(epitope_file) as f:
                epitope_indices = list(map(int, f.read().split()))
        except FileNotFoundError:
            print(f'ERROR: {epitope_file} not found. Run identify_epitope.py first.')
            sys.exit(1)
        
        # Create TNF target (immutable)
        tnf_residues = [bg.Residue(name=aa, chain_ID='T', index=i, mutable=False)
                       for i, aa in enumerate(tnf_seq)]
        tnf_chain = bg.Chain(residues=tnf_residues)
        
        # Epitope residues (for ipSAE)
        epitope_residues = [tnf_residues[i] for i in epitope_indices]
        
        # Import custom energy terms from the pH-switch script
        from ph_switch_histidine import (
            BuriedHistidineEnergy,
            HistidineCarboxylateContactEnergy,
        )
        
        # Energy terms for this species
        energy_terms = [
            # Binder fold quality
            bg.energies.PTMEnergy(oracle=esmfold, weight=2.0, name=f'{species}_ptm'),
            bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=2.0, name=f'{species}_plddt'),
            bg.energies.GlobularEnergy(oracle=esmfold, weight=0.5, name=f'{species}_globular'),
            
            # Buried histidines in the binder for pH switch
            BuriedHistidineEnergy(
                oracle=esmfold,
                target_count=6,
                sasa_cutoff=0.15,
                residues=[r for r in binder_chain.residues],
                weight=4.0,
                name=f'{species}_buried_his',
            ),
            
            # Prevent His-Asp/Glu pairs that would raise pKa
            HistidineCarboxylateContactEnergy(
                oracle=esmfold,
                distance_cutoff=4.5,
                max_contacts=4,
                weight=3.0,
                name=f'{species}_no_dyad',
            ),
            
            # Binding via ipSAE (minimize PAE to epitope)
            ipSAEEnergy(
                oracle=esmfold,
                residues=[binder_chain.residues, epitope_residues],
                weight=-4.0,  # Negative: lower PAE (higher ipSAE) is better
                name=f'{species}_binding',
            ),
        ]
        
        state = bg.State(
            name=f'binder_vs_tnf_{species}_epitope',
            chains=[binder_chain, tnf_chain],
            energy_terms=energy_terms,
        )
        states.append(state)
    
    return states


def main(backend='modal', seed=0, n_steps=1000):
    """Run the binder design."""
    print(f'Backend: {backend}, Seed: {seed}, Steps: {n_steps}')
    np.random.seed(seed)
    
    # Create binder: random length 60-100, all mutable
    binder_length = np.random.randint(60, 101)
    binder_seq = np.random.choice(list(bg.constants.aa_dict.keys()), size=binder_length)
    binder_residues = [bg.Residue(name=aa, chain_ID='B', index=i, mutable=True)
                      for i, aa in enumerate(binder_seq)]
    binder_chain = bg.Chain(residues=binder_residues)
    print(f'Binder length: {binder_length}')
    
    # Oracle
    esmfold = bg.oracles.ESMFold(backend=backend)
    
    # Create states for both human and mouse
    states = make_states(binder_chain, esmfold)
    
    # Multi-state system: optimize against both human and mouse simultaneously
    system = bg.System(states)
    
    # Simulated tempering optimizer
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),
        high_temperature=0.2,
        low_temperature=0.02,
        n_cycles=50,
        n_steps_low=20,
        n_steps_high=5,
        experiment_name=f'tnf_binder_generic_monomer_{seed}',
        callbacks=[
            bg.callbacks.DefaultLogger(log_interval=5),
            bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=20),
        ],
    )
    
    best = minimizer.minimize_system(system=system)
    print('\nBest binder sequence:')
    print(best.states[0].chains[0].sequence)
    return best


if __name__ == '__main__':
    import fire
    fire.Fire(main)
