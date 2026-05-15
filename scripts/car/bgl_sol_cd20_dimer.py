"""
BAGEL-CAR BGL/SOL CD20-Dimer Binder Design

Designs an 80-residue de novo binder that bridges both monomers of the CD20
homodimer, contacting the CD20 epitope. This is the BAGEL dimer objective used
for the BGL and SOL strategies described in the BAGEL-CAR manuscript.

Original design: Nucleate UK London, Bits to Binders competition.
"""

from __future__ import annotations

import random

import bagel as bg
import fire

# ---------------------------------------------------------------------------
# CD20 extracellular domain sequence (169 residues).
# ---------------------------------------------------------------------------
CD20_SEQUENCE = (
    'FMRESKTLGAVQIMNGLFHIALGGLLMIPAGIYAPICVTVWYPLWGGIMYIISGSLLAATE'
    'KNSRKCLVKGKMIMNSLSLFAAISGMILSIMDILNIKISHFLKMESLNFIRAHTPYINIYN'
    'CEPANPSEKNSPSTQYCYSIQSLFLGILSVMLIFAFFQELVIAGIVE'
)

# Epitope groups in PDB-style 1-based residue numbering.
CD20A_EPITOPE_RESIDUES = range(114, 133)  # His114-Asn132
CD20B_EPITOPE_RESIDUES = range(114, 132)  # His114-Lys131
# The one-residue CD20B truncation is retained for provenance.
POSITION_IDS_SKIP = 512
GLYCINE_LINKER = 'G' * 25
AMINO_ACIDS_NO_CYSTEINE = [aa for aa in bg.constants.aa_dict if aa != 'C']


def residues_by_pdb_number(residues: list[bg.Residue], pdb_numbers: range) -> list[bg.Residue]:
    return [residues[pdb_number - 1] for pdb_number in pdb_numbers]


def main(
    use_modal: bool = True,
    binder_sequence: str | None = None,
    n_cycles: int = 2000,
    n_steps_low: int = 200,
    n_steps_high: int = 200,
    n_mutations: int = 2,
    log_frequency: int = 200,
    log_path: str | None = None,
) -> bg.System:
    for name, value in (
        ('n_cycles', n_cycles),
        ('n_steps_low', n_steps_low),
        ('n_steps_high', n_steps_high),
        ('n_mutations', n_mutations),
        ('log_frequency', log_frequency),
    ):
        if value < 1:
            raise ValueError(f'{name} must be a positive integer')

    # -- Target chains (CD20 homodimer, both immutable) ---------------------
    residues_monomer_a = [
        bg.Residue(name=aa, chain_ID='C20A', index=i, mutable=False) for i, aa in enumerate(CD20_SEQUENCE)
    ]
    monomer_a = bg.Chain(residues=residues_monomer_a)

    residues_monomer_b = [
        bg.Residue(name=aa, chain_ID='C20B', index=i, mutable=False) for i, aa in enumerate(CD20_SEQUENCE)
    ]
    monomer_b = bg.Chain(residues=residues_monomer_b)

    epitope_a = residues_by_pdb_number(residues_monomer_a, CD20A_EPITOPE_RESIDUES)
    epitope_b = residues_by_pdb_number(residues_monomer_b, CD20B_EPITOPE_RESIDUES)
    epitope_combined = epitope_a + epitope_b

    # -- Binder chain (80 residues, fully mutable) --------------------------
    binder_length = 80
    if binder_sequence is None:
        binder_sequence = ''.join(random.choice(AMINO_ACIDS_NO_CYSTEINE) for _ in range(binder_length))
    else:
        if len(binder_sequence) != binder_length:
            raise ValueError('Binder sequence must be 80 residues')
        if 'C' in binder_sequence:
            raise ValueError('Historical initialization excluded cysteine')
    residues_binder = [
        bg.Residue(name=aa, chain_ID='BIND', index=i, mutable=True) for i, aa in enumerate(binder_sequence)
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # -- Oracle -------------------------------------------------------------
    esmfold = bg.oracles.ESMFold(
        use_modal=use_modal,
        config={
            'glycine_linker': GLYCINE_LINKER,
            'position_ids_skip': POSITION_IDS_SKIP,
        },
    )

    # -- Energy terms -------------------------------------------------------
    energy_terms = [
        # Global confidence
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=0.2,
        ),
        # Per-chain folding quality
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_monomer_a,
            weight=1.0,
            name='cd20a',
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_monomer_b,
            weight=1.0,
            name='cd20b',
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_binder,
            weight=1.0,
            name='binder',
        ),
        # Dimer interface confidence
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[residues_monomer_a, residues_monomer_b],
            weight=2.0,
            name='cd20a_cd20b',
        ),
        # Binder-epitope binding confidence (both monomers)
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[epitope_a, residues_binder],
            weight=2.0,
            name='epitope_a_binder',
        ),
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[epitope_b, residues_binder],
            weight=2.0,
            name='epitope_b_binder',
        ),
        # Original implementation used a term that averaged all
        # cross-group backbone atom-pair distances; current SeparationEnergy
        # uses the distance between group centroids instead.
        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=[epitope_combined, residues_binder],
            weight=1.0,
            name='epitopes_binder',
        ),
        # Binder hydrophobicity penalty
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            residues=residues_binder,
            weight=1.0,
            name='binder',
        ),
    ]

    # -- State & System -----------------------------------------------------
    state = bg.State(
        name='cd20_dimer',
        chains=[monomer_a, monomer_b, binder_chain],
        energy_terms=energy_terms,
    )
    system = bg.System(states=[state])

    # -- Minimizer ----------------------------------------------------------
    # Original implementation made up to 2 mutations (choosing randomly
    # between 1 or 2 mutations at any MC step) and did not exclude self-substitutions.
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=n_mutations, exclude_self=False),
        high_temperature=1.0,
        low_temperature=0.1,
        n_cycles=n_cycles,
        n_steps_low=n_steps_low,
        n_steps_high=n_steps_high,
        preserve_best_system_every_n_steps=n_steps_low + n_steps_high,
        log_path=log_path,
        log_frequency=log_frequency,
    )

    best_system = minimizer.minimize_system(system=system)
    return best_system


if __name__ == '__main__':
    fire.Fire(main)
