"""
BAGEL-CAR LNK Monomer Design

Designs a 20-residue peptide binder against a single CD20 monomer epitope
(His114-Asn132 of the extracellular domain). This is the first stage of the
LNK pipeline described in the BAGEL-CAR manuscript: BAGEL-derived monomer
seeds are later connected via RFdiffusion linker design and SolubleMPNN
sequence optimisation.

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

# Epitope group in PDB-style 1-based residue numbering.
EPITOPE_RESIDUES = range(114, 133)  # His114-Asn132
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
        if not isinstance(value, int):
            raise TypeError(f'{name} must be an integer')
        if value < 1:
            raise ValueError(f'{name} must be a positive integer')

    # -- Target chain (CD20 monomer, fully immutable) -----------------------
    residues_target = [
        bg.Residue(name=aa, chain_ID='CD20', index=i, mutable=False) for i, aa in enumerate(CD20_SEQUENCE)
    ]
    target_chain = bg.Chain(residues=residues_target)

    residues_epitope = residues_by_pdb_number(residues_target, EPITOPE_RESIDUES)
    epitope_residue_indices = {residue.index for residue in residues_epitope}
    residues_non_epitope = [r for r in residues_target if r.index not in epitope_residue_indices]

    # -- Binder chain (20 residues, fully mutable) --------------------------
    binder_length = 20
    allowed_residues = set(AMINO_ACIDS_NO_CYSTEINE)
    if binder_sequence is None:
        binder_sequence = ''.join(random.choice(AMINO_ACIDS_NO_CYSTEINE) for _ in range(binder_length))
    else:
        if len(binder_sequence) != binder_length:
            raise ValueError(f'binder_sequence must be {binder_length} residues, got {len(binder_sequence)}')
        invalid = set(binder_sequence) - allowed_residues
        if invalid:
            raise ValueError(
                f'binder_sequence contains invalid residues {sorted(invalid)!r}; '
                f'allowed (uppercase, cysteine excluded): {sorted(allowed_residues)!r}'
            )
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
        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=0.2,
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_non_epitope,
            weight=0.2,
            name='cd20_non_epitope',
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_epitope,
            weight=1.0,
            name='cd20_epitope',
        ),
        bg.energies.PLDDTEnergy(
            oracle=esmfold,
            residues=residues_binder,
            weight=1.0,
            name='binder',
        ),
        bg.energies.PAEEnergy(
            oracle=esmfold,
            residues=[residues_epitope, residues_binder],
            weight=2.0,
            name='epitope_binder',
        ),
        # Original implementation used a term that averaged all
        # cross-group backbone atom-pair distances; current SeparationEnergy
        # uses the distance between group centroids instead.
        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=[residues_epitope, residues_binder],
            weight=1.0,
            name='epitope_binder',
        ),
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            residues=residues_binder,
            weight=1.0,
            name='binder',
        ),
    ]

    # -- State & System -----------------------------------------------------
    state = bg.State(
        name='cd20_monomer',
        chains=[target_chain, binder_chain],
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
