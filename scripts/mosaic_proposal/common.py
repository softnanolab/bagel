"""Shared setup of the 1TNF-monomer binder benchmark (see README.md)."""

import os

import numpy as np

import bagel as bg

# >1TNF_1|Chains A, B, C|TUMOR NECROSIS FACTOR-ALPHA (RCSB FASTA). Only one chain: the target is the monomer.
TNF_MONOMER = (
    'VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEG'
    'AEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLLFAESGQVYFGIIAL'
)
# >4ZFO chain A (UniProt Q02223): human BCMA (TNFRSF17) extracellular domain, 54 residues, from the J22.9-xi Fab complex.
BCMA = 'MLQMAGQCSQNEYFDSLLHACIPCQLRCSSNTPPLTCQRYCNASVTNSVKGTNA'
# >1UBQ chain A: human ubiquitin (76); >1PGA chain A: protein G B1 domain (56)
UBQ = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
GB1 = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
TARGETS = {'tnf': TNF_MONOMER, 'bcma': BCMA, 'ubq': UBQ, 'gb1': GB1}
TARGET = TARGETS[os.environ.get('BENCH_TARGET', 'tnf')]  # set BENCH_TARGET=bcma for the BCMA benchmark
def _load_hotspots() -> dict[str, list[int]]:
    import json
    import pathlib

    f = pathlib.Path(__file__).with_name('hotspots.json')
    return {k: v['hotspot'] for k, v in json.loads(f.read_text()).items()} if f.exists() else {}


HOTSPOTS = _load_hotspots()  # 0-based positions in the target sequence, from compute_hotspots.py
BINDER_LENGTH = 60
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def make_oracle() -> bg.oracles.folding.ESMFold2:
    """ESMFold2 with the unoptimised ('vanilla') boileroom implementation."""
    # BENCH_GPU selects the GPU of bagel's ESMFold2 container (default: boileroom's default, a T4), e.g. BENCH_GPU=A100-80GB
    return bg.oracles.folding.ESMFold2(backend='modal', device=os.environ.get('BENCH_GPU') or None, config={'optimization': 'vanilla'})


def initial_binder(seed: int) -> str:
    """Random binder (no cysteine, as in the default mutation bias); identical for every arm of a given seed."""
    rng = np.random.RandomState(1000 + seed)
    return ''.join(rng.choice(list(AMINO_ACIDS.replace('C', '')), size=BINDER_LENGTH))


def build_system(oracle: bg.oracles.folding.ESMFold2, binder_sequence: str, target_name: str | None = None) -> bg.System:
    """Binder (all mutable) against the fixed TNF monomer. Every term has a mosaic counterpart and uses all residues
    (pLDDT over the whole complex; PAE and ipSAE between the whole binder and the whole target)."""
    binder = bg.Chain([bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate(binder_sequence)])
    target_sequence = TARGETS[target_name] if target_name else TARGET
    target = bg.Chain([bg.Residue(a, 'T', i, mutable=False) for i, a in enumerate(target_sequence)])
    if os.environ.get('BENCH_TERMS', 'all') == 'hotspot':
        # same four terms, but PAE and ipSAE only between the binder and a hotspot patch of the target (positions from compute_hotspots.py)
        hotspot = [target.residues[i] for i in HOTSPOTS[target_name or os.environ.get('BENCH_TARGET', 'tnf')]]
        terms = [
            bg.energies.PTMEnergy(oracle=oracle, weight=1.0),
            bg.energies.OverallPLDDTEnergy(oracle=oracle, weight=1.0),
            bg.energies.PAEEnergy(oracle=oracle, residues=[hotspot, binder.residues], weight=2.0, name='hotspot'),
            bg.energies.ipSAEEnergy(oracle=oracle, residues=[hotspot, binder.residues], weight=2.0, direction='min', name='hotspot'),
        ]
    else:
        terms = [
            bg.energies.PTMEnergy(oracle=oracle, weight=1.0),
            bg.energies.OverallPLDDTEnergy(oracle=oracle, weight=1.0),  # all residues, binder and target
            bg.energies.PAEEnergy(oracle=oracle, residues=[binder.residues, target.residues], weight=2.0),
            bg.energies.ipSAEEnergy(oracle=oracle, residues=[binder.residues, target.residues], weight=2.0, direction='min'),
        ]
    state = bg.State(name='binder_vs_TNF', chains=[binder, target], energy_terms=terms)
    return bg.System(states=[state])
