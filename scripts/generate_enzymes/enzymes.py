import bagel as bg
import os
import numpy as np
from typing import Any

def run_generate_mimic() -> Any:

    # Get the value of an environment variable
    use_modal = True if os.getenv('USE_MODAL', 'True').lower() in ('true', '1', 'yes') else False

    # Check
    print(f'Whether to use modal: {use_modal}')

    # Define the target protein we want to mimic
    # Here it is:
    # 7NJC_1|Chain A|Zinc finger (CCCH type) motif-containing protein|Toxoplasma gondii
    # (strain ATCC 50611 / Me49) (508771)
    # sequence: GDPFGHVASPQSTKRFFIIKSNRMSNIYTSIQHGVWATSKGNSRKLSNAFTSTDHVLLLFSANESGGFQGFGRMMSLPDPQLFPGIWGPVQLRLGSNFRVMWLKQCKIEFEELGKVTNPWNDDLPLRKSRDGTEVPPALGSLLCTWMSQRPSEDLLAGTGIDPATR
    target_sequence = 'GDPFGHVASPQSTKRFFIIKSNRMSNIYTSIQHGVWATSKGNSRKLSNAFTSTDHVLLLFSANESGGFQGFGRMMSLPDPQLFPGIWGPVQLRLGSNFRVMWLKQCKIEFEELGKVTNPWNDDLPLRKSRDGTEVPPALGSLLCTWMSQRPSEDLLAGTGIDPATR'

    # Define the mutability of the residues. Here, mutable residues are those NOT binding to RNA,
    # as observed in the PDB structure, all others are immutable.
    mutable_list = list( range(16) + range(66, 120) + range(131,157) )
    mutability = np.array( [ True if i in mutable_list else False for i in range(len(target_sequence)) ] )

    # Extract embeddings for the residues that you want to remain constant and whose environment
    # you want to maintain
    # First define the EmbeddingOracle
    esm2 = bg.oracles.ESM2( use_modal=use_modal, config={ 'model_name': 'esm2_t33_650M_UR50D', } )

    result = esm2.embed( sequence=[target_sequence] )
    reference_embeddings = result.embeddings[mutability]  # get the embeddings for the reference (immutable) residues

    # Define a chain providing a list of residues
    all_residues = [
        bg.Residue(name=aa, chain_ID='zinger', index=i, mutable=mut)
        for i, (aa, mut) in enumerate(zip(target_sequence, mutability))
    ]

    conserved_residues = [res for res in all_residues if res.mutable is False]

    chain = bg.Chain(residues=all_residues)

    # Define the energy terms to be applied to the chain. apply them to residues, and specify the weight
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

    # Define the system
    initial_system = bg.System(states=[state])

    # Define the minimizer
    # In this case, we want to do MC sampling to generate a diverse set of structures,
    # see Rajendran et al 2025.
    # So we do standard MC with fixed temperature of 0.001
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=bg.mutation.Canonical(n_mutations=1), # cannot add/remove residues, only substitutes for different amino acid types
        temperature=0.001,  # fixed temperature for the MC sampling
        n_steps=10000,  # number of steps to run the minimizer
        log_frequency=50,
    )

    final_system = minimizer.minimize_system(system=initial_system)

    return final_system

if __name__ == '__main__':
    run_generate_mimic()
