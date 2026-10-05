# Generated with assistance from Claude (bagel-script-builder). Review before use.
"""Generic binder vs full TNF-alpha homo-trimer."""

from __future__ import annotations
import os, sys, numpy as np, bagel as bg
sys.path.insert(0, '.')
from utils import ipSAEEnergy

TNF_HUMAN = 'VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL'
TNF_MOUSE = 'LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL'

def make_states(binder_chain, esmfold):
    from ph_switch_histidine import BuriedHistidineEnergy, HistidineCarboxylateContactEnergy
    states = []
    for species, tnf_seq in [('human', TNF_HUMAN), ('mouse', TNF_MOUSE)]:
        # Build homo-trimer
        tnf_chains = []
        for i in range(3):
            residues = [bg.Residue(name=aa, chain_ID=f'T{i}', index=j, mutable=False)
                       for j, aa in enumerate(tnf_seq)]
            tnf_chains.append(bg.Chain(residues=residues))
        
        # All TNF residues are targets for binding
        all_tnf_residues = [r for chain in tnf_chains for r in chain.residues]
        
        energy_terms = [
            bg.energies.PTMEnergy(oracle=esmfold, weight=2.0, name=f'{species}_ptm'),
            bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=2.0, name=f'{species}_plddt'),
            bg.energies.GlobularEnergy(oracle=esmfold, weight=0.5, name=f'{species}_globular'),
            BuriedHistidineEnergy(oracle=esmfold, target_count=6, residues=binder_chain.residues,
                                weight=4.0, name=f'{species}_buried_his'),
            HistidineCarboxylateContactEnergy(oracle=esmfold, weight=3.0, name=f'{species}_no_dyad'),
            ipSAEEnergy(oracle=esmfold, residues=[binder_chain.residues, all_tnf_residues],
                       weight=-4.0, name=f'{species}_binding'),
        ]
        
        state = bg.State(name=f'binder_vs_tnf_{species}_trimer',
                        chains=[binder_chain] + tnf_chains, energy_terms=energy_terms)
        states.append(state)
    return states

def main(backend='modal', seed=0):
    np.random.seed(seed)
    binder_length = np.random.randint(60, 101)
    binder_seq = np.random.choice(list(bg.constants.aa_dict.keys()), size=binder_length)
    binder_residues = [bg.Residue(name=aa, chain_ID='B', index=i, mutable=True)
                      for i, aa in enumerate(binder_seq)]
    binder_chain = bg.Chain(residues=binder_residues)
    
    esmfold = bg.oracles.ESMFold(backend=backend)
    states = make_states(binder_chain, esmfold)
    system = bg.System(states)
    
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),
        high_temperature=0.2, low_temperature=0.02,
        n_cycles=50, n_steps_low=20, n_steps_high=5,
        experiment_name=f'tnf_binder_generic_trimer_{seed}',
        callbacks=[bg.callbacks.DefaultLogger(log_interval=5),
                  bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=20)],
    )
    best = minimizer.minimize_system(system=system)
    print('Best binder:', best.states[0].chains[0].sequence)
    return best

if __name__ == '__main__':
    import fire
    fire.Fire(main)
