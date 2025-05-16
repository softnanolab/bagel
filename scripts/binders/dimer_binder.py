import random
import bagel as bg
import os
import modal
from typing import Any, Optional, List, Dict
from pathlib import Path
import re

import modalfold
modalfold.utils.MODEL_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def parse_hotspot_string(hotspot_str: str, chain_residues: Dict[str, List[bg.Residue]]) -> List[bg.Residue]:
    """
    Parse a hotspot string and return the corresponding residues.

    Format examples:
    - "A1-10,B2,B4,B6" - residues 1-10 on chain A, and residues 2, 4, 6 on chain B
    - "A5-15" - only residues 5-15 on chain A

    Args:
        hotspot_str: String describing the hotspot residues
        chain_residues: Dictionary mapping chain IDs to lists of residues

    Returns:
        List of residues corresponding to the hotspot
    """
    hotspot_residues = []

    # Split by comma to get individual segments
    segments = hotspot_str.split(',')

    for segment in segments:
        # Use regex to parse the segment
        match = re.match(r'([A-Z])(\d+)(?:-(\d+))?', segment)
        if not match:
            raise ValueError(f"Invalid hotspot segment: {segment}")

        chain_id, start, end = match.groups()

        # Check if chain exists
        if chain_id not in chain_residues:
            raise ValueError(f"Chain {chain_id} not found in the system")

        # Convert to integers
        start_idx = int(start)

        # If it's a range (A1-10), get all residues in the range
        if end is not None:
            end_idx = int(end)
            for i in range(start_idx, end_idx + 1):
                # Find residue with matching index
                for residue in chain_residues[chain_id]:
                    if residue.index == i:
                        hotspot_residues.append(residue)
                        break
        # If it's a single residue (B2), get just that residue
        else:
            for residue in chain_residues[chain_id]:
                if residue.index == start_idx:
                    hotspot_residues.append(residue)
                    break

    return hotspot_residues

def design_dimer_binder(
    binder_length: int,
    target_sequence_1: str,
    target_sequence_2: str,
    hotspot: str = "A10-20",  # Default hotspot is residues 10-20 on chain A
    linker: str = "G" * 5,
    use_modal: bool = False,
    experiment_name: str = 'dimer_binder',
    log_path: str | Path = Path('data'),
) -> bg.System:
    """
    Design a protein binder for a dimer target.

    This function creates a target dimer protein, designs a binder peptide,
    and optimizes it using simulated tempering.

    Args:
        binder_length: Length of the binder peptide to design
        target_sequence_1: Amino acid sequence for the first chain of the target
        target_sequence_2: Amino acid sequence for the second chain of the target
        hotspot: String describing the hotspot residues (e.g., "A1-10,B2,B4,B6")
        use_modal: Whether to use modal for folding
        experiment_name: Name for the experiment
        log_path: Path to save logs

    Returns:
        The optimized system with the designed binder
    """
    # Step 1: Determine whether to use modal
    if use_modal is None:
        use_modal = os.getenv('USE_MODAL', 'True').lower() in ('true', '1', 'yes')
        print(f'Whether to use modal: {use_modal}')

    # Step 2: Create target chains
    # Create first chain (all immutable)
    residues_target_1 = [
        bg.Residue(name=aa, chain_ID='A', index=i+1, mutable=False)
        for i, aa in enumerate(target_sequence_1)
    ]
    target_chain_1 = bg.Chain(residues=residues_target_1)

    # Create second chain (all immutable)
    residues_target_2 = [
        bg.Residue(name=aa, chain_ID='B', index=i+1, mutable=False)
        for i, aa in enumerate(target_sequence_2)
    ]
    target_chain_2 = bg.Chain(residues=residues_target_2)

    # Step 3: Parse hotspot string to get hotspot residues
    target_chains = { 'A': residues_target_1, 'B': residues_target_2 }
    residues_hotspot = parse_hotspot_string(hotspot, target_chains)

    if not residues_hotspot:
        raise ValueError(f"No hotspot residues found for specification: {hotspot}")

    print(f"Using {len(residues_hotspot)} hotspot residues")

    # Step 4: Create binder chain with random sequence
    binder_sequence = ''.join([random.choice(list(bg.constants.aa_dict.keys())) for _ in range(binder_length)])
    residues_binder = [
        bg.Residue(name=aa, chain_ID='C', index=i, mutable=True)
        for i, aa in enumerate(binder_sequence)
    ]
    binder_chain = bg.Chain(residues=residues_binder)

    # Step 5: Define energy terms and weights
    energy_terms = [
        bg.energies.PTMEnergy(),
        bg.energies.OverallPLDDTEnergy(),
        bg.energies.HydrophobicEnergy(),
        bg.energies.PAEEnergy(
            group_1_residues=residues_hotspot,
            group_2_residues=residues_binder,
        ),
    ]
    energy_terms_weights = [1.0, 1.0, 5.0, 5.0]

    # Step 6: Create state and system
    state = bg.State(
        chains=[target_chain_1, target_chain_2, binder_chain],
        energy_terms=energy_terms,
        energy_terms_weights=energy_terms_weights,
        name='state_A',
    )
    initial_system = bg.System(states=[state])

    # Step 7: Configure folding
    config = {
        'output_pdb': False,
        'output_cif': False,
        'glycine_linker': linker,
        'position_ids_skip': 512,
    }
    os.environ["HF_MODEL_DIR"] = os.path.expanduser("~/.cache/huggingface/hub")
    folder = bg.folding.ESMFolder(use_modal=use_modal, config=config)

    # Step 8: Configure and run minimization
    minimizer = bg.minimizer.SimulatedTempering(
        folder=folder,
        mutator=bg.mutation.Canonical(n_mutations=1),
        high_temperature=1.0,
        low_temperature=0.5,
        n_steps_high=50,
        n_steps_low=50,
        n_cycles=2,
        preserve_best_system=True,
        log_frequency=5,
        experiment_name=experiment_name,
        log_path=log_path,
    )

    # Run minimization and return best system
    best_system = minimizer.minimize_system(system=initial_system)
    return best_system

if __name__ == "__main__":
    with modal.enable_output():
        best_system = design_dimer_binder(
            binder_length=10,
            target_sequence_1='TEIIVKLSDGRELCLDPKENWVQRVVEKFLKRAENS',
            target_sequence_2='SAKELRCQCIKTYSKPFHPKFIKELRVIESGPHCANT',
            hotspot="A1-10,B2,B4,B6",  # Example hotspot specification
        )
        print(f"Best system energy: {best_system.get_energy()}")
