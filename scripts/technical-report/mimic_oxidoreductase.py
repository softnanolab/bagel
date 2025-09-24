import bagel as bg
import os
import numpy as np
from typing import Any
import modal

# This script implements the method described in Rajendran et al 2025 to mimic enzymes
# using the ESM2 embeddings. It uses the Metropolis minimizer to generate a diverse set of structures
# that maintain the environment of the conserved residues, while mutating the rest of the residues.

def run_generate_mimic() -> Any:

    # Get the value of an environment variable
    use_modal = False

    # Check
    print(f'Whether to use modal: {use_modal}')

    # Define the target enzyme we want to mimic
    # Here it is (just to make a random example):
    # >1A2J_1|Chain A|DISULFIDE BOND FORMATION PROTEIN|Escherichia coli (562)
    # AQYEDGKQYTTLEKPVAGAPQVLEFFSFFCPHCYQFEEVLHISDNVKKKLPEGVKMTKYHVNFMGGDLGKDLTQAWAVAMALGVEDKVTVPLFEGVQKTQTIRSASDIRDVFINAGIKGEEYDAAWNSFVVKSLVAQQEKAAADVQLRGVPAMFVNGKYQLNPQGMDTSNMDVFVQQYADTVKYLSEKK
    target_sequence = 'AQYEDGKQYTTLEKPVAGAPQVLEFFSFFCPHCYQFEEVLHISDNVKKKLPEGVKMTKYHVNFMGGDLGKDLTQAWAVAMALGVEDKVTVPLFEGVQKTQTIRSASDIRDVFINAGIKGEEYDAAWNSFVVKSLVAQQEKAAADVQLRGVPAMFVNGKYQLNPQGMDTSNMDVFVQQYADTVKYLSEKK'

    # Non-mutable should be CYS-30 and CYS-33, 1-indexed, so 0-index it
    immutable_list = [30 - 1, 33 - 1]
    mutable_list = [i for i in range(len(target_sequence)) if i not in immutable_list]
    mutability = [True if i in mutable_list else False for i in range(len(target_sequence))]
    print(f"Length of target sequence: {len(target_sequence)}")
    print(f"Mutability: {mutability}")
    print(f"Immutable list: {immutable_list}")
    print(f"Mutable list: {mutable_list}")

    # Define a chain providing a list of residues
    all_residues = [
        bg.Residue(name=aa, chain_ID='zing', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]
    # For use later, define the conserved residues
    # These are the residues that we want to keep constant and whose environment we want to maintain
    # In this case, these are the residues that are not mutable
    conserved_residues = [res for res in all_residues if res.mutable is False]
    print( f"Number of conserved residues: {len(conserved_residues)}" )

    # Define the chain object
    chain = bg.Chain(residues=all_residues)

    # Extract embeddings for the residues that you want to remain constant and whose environment
    # you want to maintain
    # First define the EmbeddingOracle
    esm2 = bg.oracles.ESM2( use_modal=use_modal, config={'model_name': 'esm2_t33_650M_UR50D'})
    # Embed the chain to get the embeddings for the residues
    result = esm2.embed( chains=[chain] )
    # Extract only the embeddings for the immutable residues
    immutable = ~np.array( mutability )  # get the immutable residues (those that are not mutable)
    reference_embeddings = result.embeddings[immutable]  # get the embeddings for the reference (immutable) residues

    # Define the energy terms to be applied to the chain. Apply them to residues, and specify the weight
    # Following Rajendran et al 2025, we use the EmbeddingsSimilarityEnergy of the conserved residues
    energy_terms = [
        bg.energies.EmbeddingsSimilarityEnergy(
            oracle=esm2,
            weight=1.0,
            residues=conserved_residues,  # apply the energy term to the conserved residues
            reference_embeddings=reference_embeddings,)]

    # Define the state
    state = bg.State(
        name='state_A',
        chains=[chain],
        energy_terms=energy_terms,
    )

    # Define the system - it's a collection of states but here there is only one.
    initial_system = bg.System(states=[state])

    # Define the minimizer
    # In this case, we want to do MC sampling to generate a diverse set of structures,
    # see Rajendran et al 2025. So we do standard MC with fixed temperature of 1e-4.
    # We are not really minimizing the system, but rather generating a diverse set of structures
    # at a fixed temperature in this way.
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=bg.mutation.Canonical(n_mutations=1), # cannot add/remove residues, only substitutes for different amino acid types
        temperature=1e-4,  # fixed temperature for the MC sampling
        n_steps=10000,  # number of steps to run the minimizer
        log_frequency=1,
    )

    final_system = minimizer.minimize_system(system=initial_system)

    return final_system

if __name__ == '__main__':
    run_generate_mimic()
