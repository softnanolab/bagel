import bagel as bg
import copy
import numpy as np
from biotite.database.rcsb import fetch
from bagel.utils import get_atomarray_in_residue_range, get_sequence_from_pdb_id, get_reconciled_sequence, sequence_from_atomarray
import pathlib as pl
import fire

from utils import validate_residue_identity, load_best_sequence_and_mask


def main(use_modal: bool = False, experiment_name: str | None = None, temperature: float = 1e-4, n_steps: int = 100000):
    """
    PLM-based mini-enzyme optimization for Subtilisin Carlsberg protease (PDB: 1sbc).

    Subtilisin Carlsberg is a serine protease. Catalytic residues: D32, H64, N155, S221.

    Args:
        use_modal: Whether to use Modal for GPU inference
        experiment_name: Name for the experiment run
        temperature: MC sampling temperature
        n_steps: Number of MC steps
    """
    print(f'Whether to use modal: {use_modal}')

    log_path = pl.Path("./logs") / 'mini-enzymes' / 'protease'

    offset = 1

    full_sequence = get_sequence_from_pdb_id(pdb_id="1sbc", sequence_index=0)

    chain_atoms = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1sbc", format="pdb"))
    chain_atoms = get_atomarray_in_residue_range(chain_atoms, start=offset, end=275)

    # Catalytic residues: D32, H64, N155, S221
    crucial_residues_indices = [32, 64, 155, 221]

    crucial_mask = np.isin(chain_atoms.res_id, crucial_residues_indices)
    crucial_atoms = copy.deepcopy(chain_atoms[crucial_mask])

    crucial_residues_indices_0based = [(i - offset) for i in crucial_residues_indices]

    full_sequence, _ = get_reconciled_sequence(chain_atoms, full_sequence)

    loaded_sequence, loaded_conserved_indices = load_best_sequence_and_mask(
        log_path=log_path,
        experiment_name=experiment_name,
        state_name='state_A'
    )

    if loaded_sequence is not None and loaded_conserved_indices is not None:
        print("Restarting from previous optimization")
        full_sequence = loaded_sequence
        conserved_residues_indices = loaded_conserved_indices
    else:
        print("Starting fresh optimization")
        conserved_residues_indices = []
        buffer = 4
        for res_id in crucial_residues_indices:
            conserved_residues_indices.extend(range(res_id - buffer, res_id + buffer))
        conserved_residues_indices = list(set(conserved_residues_indices))
        conserved_residues_indices = [(i - offset) for i in conserved_residues_indices]

    conserved_residues_indices_1based = [i + offset for i in conserved_residues_indices]
    conserved_mask = np.isin(chain_atoms.res_id, conserved_residues_indices_1based)
    conserved_atoms = copy.deepcopy(chain_atoms[conserved_mask])

    mutability = [False if i in conserved_residues_indices else True for i in range(len(full_sequence))]

    all_residues = [
        bg.Residue(name=aa, chain_ID='AAA', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(full_sequence, mutability))
    ]

    expected_crucial_residues = ["Asp32", "His64", "Asn155", "Ser221"]
    crucial_residues = [all_residues[i] for i in crucial_residues_indices_0based]
    validate_residue_identity(residues=crucial_residues, expected_residues=expected_crucial_residues)

    crucial_atoms_residues = list(sequence_from_atomarray(crucial_atoms))
    validate_residue_identity(residues=crucial_residues, expected_residues=crucial_atoms_residues)

    conserved_residues = [res for res in all_residues if res.mutable is False]
    print(f"Number of conserved residues: {len(conserved_residues)}")

    chain = bg.Chain(residues=all_residues)

    esm2 = bg.oracles.ESM2(use_modal=use_modal, config={'model_name': 'esm2_t33_650M_UR50D'})
    result = esm2.embed(chains=[chain])
    immutable = ~np.array(mutability)
    reference_embeddings = result.embeddings[immutable]

    energy_terms = [
        bg.energies.EmbeddingsSimilarityEnergy(
            oracle=esm2,
            weight=1.0,
            residues=conserved_residues,
            reference_embeddings=reference_embeddings),
        bg.energies.ChemicalPotentialEnergy(
            oracle=esm2,
            weight=1e-3,
            name='chemical_potential_energy',
        )]

    state = bg.State(
        name='state_A',
        chains=[chain],
        energy_terms=energy_terms,
    )

    initial_system = bg.System(states=[state])

    my_mutator = bg.mutation.GrandCanonical(
        n_mutations=1,
        move_probabilities={
            'substitution': 0.70,
            'addition': 0.15,
            'removal': 0.15,
        }
    )

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=my_mutator,
        temperature=temperature,
        n_steps=n_steps,
        log_frequency=20,
        experiment_name=str(experiment_name) if experiment_name is not None else None,
        log_path=log_path
    )

    final_system = minimizer.minimize_system(system=initial_system)
    return final_system


if __name__ == '__main__':
    fire.Fire(main)
