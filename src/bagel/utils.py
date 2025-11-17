import re
import pathlib
import os
import tempfile
import logging
import numpy as np
from typing import Optional

from biotite.structure import AtomArray
from .constants import aa_dict

import biotite.database.rcsb as rcsb
import biotite.sequence.io.fasta as fasta
import biotite.sequence as seq

aa_dict_3to1 = {v: k for k, v in aa_dict.items()}

logger = logging.getLogger(__name__)


def get_version_from_pyproject(pyproject_path: Optional[str] = None) -> Optional[str]:
    """Parse the version string from pyproject.toml."""
    if pyproject_path is None:
        path = pathlib.Path(__file__).parent.parent.parent / 'pyproject.toml'
    else:
        path = pathlib.Path(pyproject_path)
    version = None
    if path.exists():
        with open(path) as f:
            for line in f:
                m = re.match(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', line)
                if m:
                    version = m.group(1)
                    break
    return version


def get_atomarray_in_residue_range(
    atoms: AtomArray,
    start: Optional[int] = None,
    end: Optional[int] = None,
    chain: Optional[str] = None,
) -> AtomArray:
    """
    Extract a range of residues, optionally restricted to a chain, from an AtomArray.

    Parameters
    ----------
    atoms: AtomArray
        AtomArray containing all atoms.
    start: int or None, default=None
        Starting residue index (inclusive). If None, uses the minimum `res_id` in `atoms`.
    end: int or None, default=None
        Ending residue index (inclusive). If None, uses the maximum `res_id` in `atoms`.
    chain: str or None, default=None
        Chain identifier (e.g., 'A', 'B'). If None, all chains are considered.

    Returns
    -------
    AtomArray
        The subset of atoms within the requested residue index range and chain.
    """
    slice_atoms = atoms[atoms.chain_id == chain] if chain is not None else atoms

    if start is None:
        start = int(slice_atoms.res_id.min())
    if end is None:
        end = int(slice_atoms.res_id.max())

    sel = (slice_atoms.res_id >= start) & (slice_atoms.res_id <= end)
    return slice_atoms[sel]


def sequence_from_atomarray(atoms: AtomArray) -> str:
    """
    Extract the amino acid sequence in 1-letter code from an AtomArray.

    Parameters
    ----------
    atoms: AtomArray
        AtomArray containing all atoms.

    Returns
    -------
    str
        Amino acid sequence in 1-letter convention, derived from residues with 'CA' atoms.
    Notes
    -----
    Any non-protein atoms (e.g., water, ions, ligands) are removed before extracting the sequence.
    If such atoms are present, they are ignored in the sequence extraction.
    """
    protein_mask = np.isin(atoms.res_name, list(aa_dict_3to1.keys()))
    protein_atoms = atoms[protein_mask]
    ca_mask = protein_atoms.atom_name == 'CA'
    return ''.join([aa_dict_3to1[res_name] for res_name in protein_atoms[ca_mask].res_name])


def get_sequence_from_fasta(pdb_id: str, sequence_index: int = 0) -> str:
    """Download sequence from the FASTA file using the pdb ID. This sequence is
    complete unlike that extracted from the PDB AtomArray object that only contains
    residues for which coordinates have been determined.
    Input:
    pdb_id : str, PDB id
    sequence_index: int = 0, Index of the chain to extract (in case of multimers)
    Output:
    sequence : str, sequence of the protein
    """
    # 1) Download the FASTA file from the RCSB for a given PDB ID
    fasta_file_path = rcsb.fetch(pdb_id, format='fasta', target_path=None, overwrite=True)
    # 2) Read the FASTA file into a Biotite FastaFile object
    fasta_file = fasta.FastaFile.read(fasta_file_path)

    sequences = fasta.get_sequences(fasta_file)
    output_sequences = []
    for seq_obj in sequences.values():
        output_sequences.append(str(seq_obj))

    assert sequence_index < len(output_sequences), (
        f'Requested sequence index {sequence_index} but only {len(output_sequences)} sequences found.'
    )

    return output_sequences[sequence_index]


def resolve_and_set_model_dir() -> pathlib.Path:
    """
    Resolve and set the MODEL_DIR environment variable to a user-writable cache
    location following XDG conventions.

    Precedence:
    1) Respect existing MODEL_DIR if set.
    2) Use XDG_CACHE_HOME if defined; otherwise default to ~/.cache.
    3) Append "bagel/models" and create the directory if it does not exist.

    Returns
    -------
    pathlib.Path
        Absolute path to the resolved model directory.
    """
    try:
        if os.environ.get('MODEL_DIR'):
            resolved = pathlib.Path(os.environ['MODEL_DIR']).expanduser().resolve()
        else:
            xdg_cache_home = os.getenv('XDG_CACHE_HOME')
            if xdg_cache_home:
                base_cache_dir = pathlib.Path(xdg_cache_home).expanduser().resolve()
            else:
                base_cache_dir = pathlib.Path.home() / '.cache'
            resolved = (base_cache_dir / 'bagel' / 'models').resolve()

        resolved.mkdir(parents=True, exist_ok=True)
        os.environ['MODEL_DIR'] = str(resolved)
        return resolved
    except (OSError, PermissionError) as exc:
        logger.warning(f'Falling back to a temporary model cache due to filesystem error: {str(exc)}')
    except Exception as exc:  # noqa: BLE001
        logger.warning(f'Falling back to a temporary model cache due to unexpected error: {str(exc)}')

    # Fallback to a user-writable temporary directory; ensure directory exists
    fallback_base = pathlib.Path(tempfile.gettempdir()) / 'bagel' / 'models'
    fallback_base.mkdir(parents=True, exist_ok=True)
    os.environ['MODEL_DIR'] = str(fallback_base)
    return fallback_base


def get_reconciled_sequence(atoms: AtomArray, fasta_sequence: str) -> None:
    """
    Take a sequence from an AtomArray object. If there are missing residues in the AtomArray,
    use the provided fasta_sequence to fill in the gaps.
    Note that if the AtomArray and FASTA sequence have mismatches, AtomArray sequence is preferred
    but a warning is raised.

    Parameters
    ----------
    atoms: AtomArray
        AtomArray containing all atoms.
    faata_sequence: str | None
        Amino acid sequence in 1-letter convention. If None, no reconciliation is performed
        and a random residue is chosen for missing residues.

    Returns
    -------
    list of str
        Reconciled amino acid sequence in 1-letter convention.
    bool
        Whether any residues were added to fill in gaps.

    Raises
    ------
    Warning
        If there is a mismatch between the sequence derived from `atoms` and the provided `sequence`.
    """
    offset = 1  # Because residue IDs in PDB files start from 1

    all_res_ids_from_chain = atoms.res_id.tolist()
    # Remove duplicates while preserving order
    seen = set()
    all_res_ids_from_chain = [x for x in all_res_ids_from_chain if not (x in seen or seen.add(x))]

    max_res_id = max(all_res_ids_from_chain)

    reconciled_sequence = []

    added = False

    for res_id in range(1, max_res_id + 1):
        try:
            aa_pdb = atoms.res_name[atoms.res_id == res_id][0]
            aa_pdb = aa_dict_3to1[aa_pdb]
            reconciled_sequence.append(aa_pdb)
        except IndexError:
            # Missing residue in the AtomArray
            if fasta_sequence is None:
                # If no fasta sequence is provided, choose a random amino acid
                aa_pdb = np.random.choice(list(aa_dict.keys()))
                reconciled_sequence.append(aa_pdb)
            else:
                aa_seq = fasta_sequence[res_id - offset]
                reconciled_sequence.append(aa_seq)
            added = True
            continue

        if fasta_sequence is not None:
            try:
                # This is because the FASTA sequence can be shorter than the PDB sequence because of discrepancies
                aa_seq = fasta_sequence[res_id - offset]
                if aa_pdb != aa_seq:
                    print(
                        f'Non-critical WARNING: Mismatch between PDB and sequence at residue {res_id}: PDB = {aa_pdb} vs FASTA = {aa_seq}'
                    )
            except IndexError:
                pass

    return ''.join(reconciled_sequence), added
