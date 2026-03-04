"""
Helper functions for mini-enzyme optimization scripts.
"""
import re
import pathlib as pl
from typing import List, Tuple, Optional

import bagel as bg
from bagel.utils import aa_dict_3to1, aa_dict


def _extract_aa_code(residue_str: str) -> str:
    """Extract one-letter amino acid code from strings like 'Asp610', 'Arg-64', 'S160', or 'D'."""
    match = re.match(r'^([A-Za-z]{1,3})([-]?\d+)?$', residue_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse residue string: '{residue_str}'")

    aa = match.group(1).upper()
    if len(aa) == 1:
        return aa
    elif len(aa) == 3:
        return aa_dict_3to1[aa]
    else:
        raise ValueError(f"Invalid amino acid code: '{aa}'")


def validate_residue_identity(
    residues: List[bg.Residue],
    expected_residues: List[str]
) -> None:
    """
    Validate that residues match expected amino acid identities in order.

    Args:
        residues: List of Residue objects to validate
        expected_residues: List like ["Asp610", "Phe667"] - residue ID is ignored

    Raises:
        AssertionError: If lengths differ or identities don't match
    """
    if len(residues) != len(expected_residues):
        raise AssertionError(
            f"Length mismatch: {len(residues)} residues vs {len(expected_residues)} expected"
        )

    mismatches = []
    for i, (res, exp) in enumerate(zip(residues, expected_residues)):
        expected_1letter = _extract_aa_code(exp)
        if res.name != expected_1letter:
            got_3letter = aa_dict.get(res.name.upper(), res.name)
            exp_3letter = aa_dict.get(expected_1letter.upper(), expected_1letter)
            mismatches.append(
                f"Position {i}: expected {exp_3letter} (from '{exp}'), got {got_3letter} (res.name='{res.name}')"
            )

    if mismatches:
        raise AssertionError("Residue identity mismatches:\n  " + "\n  ".join(mismatches))


def load_best_sequence_and_mask(
    log_path: pl.Path,
    experiment_name: str,
    state_name: str,
) -> Tuple[Optional[str], Optional[List[int]]]:
    """
    Load the best sequence and mask from a previous optimization.

    Returns:
        Tuple of (sequence, conserved_residues_indices) or (None, None) if not found.
    """
    best_folder = log_path / str(experiment_name) / "best"
    fasta_path = best_folder / f"{state_name}.fasta"
    mask_path = best_folder / f"{state_name}.mask.fasta"

    if not fasta_path.exists() or not mask_path.exists():
        return None, None

    try:
        sequence = None
        with open(fasta_path, 'r') as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if line and not line.startswith('>'):
                    sequence = line.split(':')[0] if ':' in line else line
                    break

        if sequence is None:
            return None, None

        mask = None
        with open(mask_path, 'r') as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if line and not line.startswith('>'):
                    mask = line.split(':')[0] if ':' in line else line
                    break

        if mask is None or len(mask) != len(sequence):
            return None, None

        conserved_residues_indices = [i for i, char in enumerate(mask) if char == 'I']

        print(f"Loaded best sequence and mask from {best_folder}")
        print(f"Sequence length: {len(sequence)}, Conserved residues: {len(conserved_residues_indices)}")

        return sequence, conserved_residues_indices

    except Exception as e:
        print(f"Warning: Error reading restart files: {e}")
        return None, None
