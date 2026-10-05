# Generated with assistance from Claude (bagel-script-builder). Review before use.
"""DARPin binder (124 residues, 12 mutable) vs TNF-alpha monomer epitope."""

from __future__ import annotations
import os, sys, numpy as np, bagel as bg
sys.path.insert(0, '.')
from utils import ipSAEEnergy, load_epitope_residues, make_darpin_scaffold

TNF_HUMAN = 'VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL'
TNF_MOUSE = 'LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL'

def make_states(darpin_chain, esmfold):
    from ph_switch_histidine import BuriedHistidineEnergy, HistidineCarboxylateContactEnergy
    states = []
    mutable_residues = [r for r in darpin_chain.residues if r.mutable]
    
    for species, tnf_seq in [('human', TNF_HUMAN), ('mouse', TNF_MOUSE)]:
        epitope_file = f'tnf_{species}_epitope.txt'
        try:
            with open(epitope_file) as f:
                epitope_indices = list(map(int, f.read().split()))
        except FileNotFoundError:
            print(f'ERROR: {epitope_file} not found. Run identify_epitope.py first.')
            sys.exit(1)
        
        tnf_residues = [bg.Residue(name=aa, chain_ID='T', index=i, mutable=False)
                       for i, aa in enumerate(tnf_seq)]
        tnf_chain = bg.Chain(residues=tnf_residues)
        epitope_residues = [tnf_residues[i] for i in epitope_indices]
        
        energy_terms = [
            bg.energies.PTMEnergy(oracle=esmfold, weight=2.0, name=f'{species}_ptm'),
            bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=2.0, name=f'{species}_plddt'),
            bg.energies.GlobularEnergy(oracle=esmfold, weight=0.5, name=f'{species}_globular'),
            BuriedHistidineEnergy(oracle=esmfold, target_count=4, residues=mutable_residues,
                                weight=4.0, name=f'{species}_buried_his'),
            HistidineCarboxylateContactEnergy(oracle=esmfold, weight=3.0, name=f'{species}_no_dyad'),
            ipSAEEnergy(oracle=esmfold, residues=[mutable_residues, epitope_residues],
                       weight=-4.0, name=f'{species}_binding'),
        ]
        
        state = bg.State(name=f'darpin_vs_tnf_{species}_epitope',
                        chains=[darpin_chain, tnf_chain], energy_terms=energy_terms)
        states.append(state)
    return states

def main(backend='modal', seed=0):
    np.random.seed(seed)
    darpin_chain = bg.Chain(residues=make_darpin_scaffold(length=124, n_variable=12))
    
    esmfold = bg.oracles.ESMFold(backend=backend)
    states = make_states(darpin_chain, esmfold)
    system = bg.System(states)
    
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1),
        high_temperature=0.2, low_temperature=0.02,
        n_cycles=50, n_steps_low=20, n_steps_high=5,
        experiment_name=f'tnf_darpin_monomer_{seed}',
        callbacks=[bg.callbacks.DefaultLogger(log_interval=5),
                  bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=20)],
    )
    best = minimizer.minimize_system(system=system)
    print('Best DARPin:', best.states[0].chains[0].sequence)
    return best

if __name__ == '__main__':
    import fire
    fire.Fire(main)
