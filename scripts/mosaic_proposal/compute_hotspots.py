"""
Define a hotspot patch on each benchmark target from its PDB structure, by a fixed rule (nothing is taken from the literature):
  1. relative solvent accessibility of every residue of the isolated target chain (biotite SASA / theoretical maximum of the residue type)
  2. anchor = the exposed (rel. SASA >= 0.25) hydrophobic residue (AVILMFWY) with the largest exposed hydrophobic area within 10 A (CA-CA)
  3. hotspot = the 8 exposed residues (rel. SASA >= 0.20) closest to the anchor (CA-CA), the anchor included
Residue positions are 0-based indices into the target sequence used in the benchmark.
"""

import json
import pathlib as pl
import sys

import numpy as np
from biotite.sequence import ProteinSequence
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

sys.path.insert(0, str(pl.Path(__file__).parent))
from common import TARGETS

SPEC = {'bcma': ('4ZFO', 'F'), 'ubq': ('1UBQ', 'A'), 'gb1': ('1PGA', 'A')}
MAX_SASA = {'TRP': 285, 'ARG': 274, 'TYR': 263, 'PHE': 240, 'LYS': 236, 'GLN': 225, 'HIS': 224, 'MET': 224, 'GLU': 223, 'LEU': 201, 'ILE': 197,
            'ASN': 195, 'ASP': 193, 'VAL': 174, 'THR': 172, 'CYS': 167, 'PRO': 159, 'SER': 155, 'ALA': 129, 'GLY': 104}
HYDROPHOBIC = set('AVILMFWY')
out = {}
for target, (pdb_id, chain) in SPEC.items():
    atoms = pdb.PDBFile.read(str(pl.Path(__file__).parent / 'structures' / f'{pdb_id}.pdb')).get_structure(model=1)
    atoms = atoms[(atoms.chain_id == chain) & struc.filter_amino_acids(atoms)]
    atoms = atoms[atoms.element != 'H']
    sasa_atoms = struc.sasa(atoms, vdw_radii='ProtOr', point_number=200)
    res_ids, res_names = struc.get_residues(atoms)
    sasa_res = struc.apply_residue_wise(atoms, sasa_atoms, np.nansum)
    letters = ''.join(ProteinSequence.convert_letter_3to1(n) for n in res_names)
    seq = TARGETS[target]
    start = seq.find(letters)
    assert start >= 0 and len(letters) <= len(seq), f'{target}: observed residues are not a contiguous stretch of the benchmark sequence'
    idx = start + np.arange(len(letters))  # index of each observed residue in the benchmark sequence
    rel = np.array([s / MAX_SASA[n] for s, n in zip(sasa_res, res_names)])
    ca = np.array([atoms[(atoms.res_id == r) & (atoms.atom_name == 'CA')].coord[0] for r in res_ids])
    dist = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    exposed = rel >= 0.25
    hyd_exposed = np.array([l in HYDROPHOBIC for l in letters]) & exposed
    patch = [(sasa_res * hyd_exposed)[dist[i] <= 10.0].sum() if hyd_exposed[i] else -1 for i in range(len(letters))]
    anchor = int(np.argmax(patch))
    cand = np.where(rel >= 0.20)[0]
    order = cand[np.argsort(dist[anchor, cand])][:8]
    hot = sorted(int(idx[k]) for k in order)
    out[target] = {'pdb': pdb_id, 'chain': chain, 'observed_start_index': int(start), 'n_observed': len(letters), 'anchor_index': int(idx[anchor]), 'hotspot': hot,
                   'hotspot_residues': [f'{seq[i]}{i + 1}' for i in hot]}
    print(f'{target}: {pdb_id} chain {chain}, {len(letters)} of {len(seq)} residues observed; anchor {seq[idx[anchor]]}{idx[anchor] + 1}; hotspot (sequence numbering from 1): {out[target]["hotspot_residues"]}; '
          f'mean rel. SASA of the hotspot {rel[order].mean():.2f}')
pl.Path(__file__).with_name('hotspots.json').write_text(json.dumps(out, indent=1))
