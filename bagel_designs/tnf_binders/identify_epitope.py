# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness. Generated: 2025-01-XX.
# -----------------------------------------------------------------------------
"""
Helper script: identify TNF-alpha interface residues for epitope masking.

Folds the TNF-alpha homo-trimer (both human and mouse) and identifies residues
at the interface (within 9 Angstrom of another chain). These are excluded from
the epitope in the monomer-epitope binder design scripts.

Outputs:
  - tnf_human_epitope.txt: residue indices (0-indexed) NOT at the interface
  - tnf_mouse_epitope.txt: residue indices (0-indexed) NOT at the interface
"""

from __future__ import annotations

import os
import numpy as np
import bagel as bg
from biotite.structure import CellList

def identify_interface_residues(structure, distance_cutoff=9.0):
    """
    Identify residues within `distance_cutoff` of atoms from other chains.
    Returns a set of residue indices (0-indexed) that are at the interface.
    """
    interface_res = set()
    chains = np.unique(structure.chain_id)
    
    if len(chains) < 2:
        return interface_res
    
    # Build a cell list for fast distance lookups
    cell_list = CellList(structure.coord, cell_size=distance_cutoff)
    
    for chain in chains:
        chain_mask = structure.chain_id == chain
        other_chains_mask = ~chain_mask
        
        if not np.any(other_chains_mask):
            continue
        
        # Find atoms in this chain close to atoms in other chains
        chain_atoms = np.where(chain_mask)[0]
        for atom_idx in chain_atoms:
            neighbors = cell_list.get_neighbors(atom_idx)
            neighbors = neighbors[other_chains_mask[neighbors]]
            
            if len(neighbors) > 0:
                # This residue is close to another chain
                res_idx = structure.res_id[atom_idx]
                interface_res.add(res_idx)
    
    return interface_res

def main(backend='modal'):
    print(f'Backend: {backend}')
    
    # TNF-alpha sequences
    tnf_human = 'VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL'
    tnf_mouse = 'LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL'
    
    oracle = bg.oracles.ESMFold(backend=backend)
    
    for species, sequence in [('human', tnf_human), ('mouse', tnf_mouse)]:
        print(f'\nProcessing {species} TNF-alpha...')
        
        # Build the homo-trimer (3 copies of the same chain)
        trimers = []
        for i in range(3):
            residues = [bg.Residue(name=aa, chain_ID=f'T{i}', index=j, mutable=False)
                       for j, aa in enumerate(sequence)]
            trimers.append(bg.Chain(residues=residues))
        
        # Fold the trimer
        result = oracle.predict(chains=trimers)
        structure = result.structure
        
        print(f'  Trimer structure: {len(structure)} atoms')
        
        # Identify interface residues
        interface = identify_interface_residues(structure, distance_cutoff=9.0)
        print(f'  Interface residues (within 9Å): {len(interface)} out of {len(sequence)}')
        
        # Epitope = non-interface residues
        all_residues = set(range(len(sequence)))
        epitope = sorted(all_residues - interface)
        
        print(f'  Epitope residues (non-interface): {len(epitope)}')
        
        # Write to file
        outfile = f'tnf_{species}_epitope.txt'
        with open(outfile, 'w') as f:
            f.write(' '.join(map(str, epitope)))
        print(f'  Wrote {outfile}')
    
    print('\nDone. Use the epitope files in the monomer-epitope binder scripts.')

if __name__ == '__main__':
    import fire
    fire.Fire(main)
