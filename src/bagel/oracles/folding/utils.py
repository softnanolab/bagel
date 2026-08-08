import os
import logging
from typing import Any
from io import StringIO

import pandas as pd  # This is necessary because its "unique" method does not sort elements and leaves them as they are
import numpy as np
import numpy.typing as npt
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from bagel.constants import atom_order, aa_dict
from bagel.chain import Chain

logger = logging.getLogger(__name__)

aa_dict_3to1 = {v: k for k, v in aa_dict.items()}


def sequence_from_atomarray(atoms: AtomArray) -> str:
    return ''.join(aa_dict_3to1[aa] for aa in atoms[atoms.atom_name == 'CA'].res_name)


def pdb_file_to_atomarray(pdb_path: str | StringIO) -> AtomArray:
    if not isinstance(pdb_path, (str, StringIO)):
        raise TypeError('pdb_path must be a string or StringIO')
    if isinstance(pdb_path, str):
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f'pdb_path does not exist: {pdb_path}')
    return PDBFile.read(pdb_path).get_structure(model=1)


def pdb_string_to_atomarray(pdb_string: str) -> AtomArray:
    if not isinstance(pdb_string, str):
        raise TypeError('pdb_string must be a string')
    return pdb_file_to_atomarray(StringIO(pdb_string))


def reindex_chains(atomarray: AtomArray, custom_chain_ids: list[str]) -> AtomArray:
    """
    Reindex the chains of an AtomArray based on a chain index map.
    This is necessary because ESMFold uses a different chain indexing than the
    flexible one use in desprot.
    """
    if not isinstance(atomarray, AtomArray):
        raise TypeError('atomarray must be an AtomArray')
    model_chain_ids = pd.unique(atomarray.chain_id)
    if len(model_chain_ids) != len(custom_chain_ids):
        raise ValueError('number of independent chains in atomarray and custom_chain_ids must be the same')

    atoms = atomarray.copy()
    original_chain_ids = atoms.chain_id.copy()
    for model_chain_id, custom_chain_id in zip(model_chain_ids, custom_chain_ids):
        atoms.chain_id[original_chain_ids == model_chain_id] = custom_chain_id
    return atoms


def prepare_single_structure(
    atom_arrays: list[AtomArray] | None,
    chains: list[Chain],
    model_name: str,
) -> AtomArray:
    """Validate and reindex the one-structure output supported by BAGEL."""
    if atom_arrays is None or len(atom_arrays) != 1:
        count = 0 if atom_arrays is None else len(atom_arrays)
        raise ValueError(f'{model_name} output must contain exactly one atom_array; got {count}')
    atoms = reindex_chains(atom_arrays[0], [chain.chain_ID for chain in chains])
    return reindex_residues(atoms, chains)


def _single_sample(values: Any, field_name: str, model_name: str) -> npt.NDArray[Any]:
    """Extract one non-empty sample from a BoilerRoom output field."""
    if values is None:
        raise ValueError(f'{model_name} output does not contain {field_name} (requested via include_fields)')
    if isinstance(values, (list, tuple)):
        if len(values) != 1:
            raise ValueError(f'{model_name} {field_name} must contain exactly one sample; got {len(values)}')
        sample = values[0]
    else:
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != 1:
            raise ValueError(f'{model_name} {field_name} must contain exactly one sample; got shape {array.shape}')
        sample = array[0]
    if sample is None:
        raise ValueError(f'{model_name} output does not contain {field_name} (requested via include_fields)')
    sample_array = np.asarray(sample)
    if sample_array.size == 0:
        raise ValueError(f'{model_name} output does not contain {field_name} (requested via include_fields)')
    return sample_array


def single_sample_vector(values: Any, field_name: str, model_name: str) -> npt.NDArray[np.float64]:
    """Return one per-residue vector with BAGEL's leading sample dimension."""
    sample = _single_sample(values, field_name, model_name)
    if sample.ndim != 1:
        raise ValueError(f'{model_name} {field_name} expected a vector; got shape {sample.shape}')
    return sample[None, :]


def single_sample_matrix(values: Any, field_name: str, model_name: str) -> npt.NDArray[np.float64]:
    """Return one pairwise matrix with BAGEL's leading sample dimension."""
    sample = _single_sample(values, field_name, model_name)
    if sample.ndim != 2:
        raise ValueError(f'{model_name} {field_name} expected a matrix; got shape {sample.shape}')
    return sample[None, :, :]


def single_sample_scalar(values: Any, field_name: str, model_name: str) -> npt.NDArray[np.float64]:
    """Return one scalar with BAGEL's two-dimensional result shape."""
    sample = _single_sample(values, field_name, model_name)
    if sample.size != 1:
        raise ValueError(f'{model_name} {field_name} expected a scalar; got shape {sample.shape}')
    return np.asarray(sample, dtype=np.float64).reshape(1, 1)


def reindex_residues(atoms: AtomArray, chains: list[Chain]) -> AtomArray:
    """
    Reindex output residues to match the input Chain residue indices.
    """
    atoms = atoms.copy()
    original_res_ids = atoms.res_id.copy()
    for chain in chains:
        chain_mask = atoms.chain_id == chain.chain_ID
        output_residue_ids = pd.unique(original_res_ids[chain_mask])
        input_residue_ids = [residue.index for residue in chain.residues]
        if len(output_residue_ids) != len(input_residue_ids):
            raise ValueError(f'number of residues in output chain {chain.chain_ID} does not match input chain')
        id_conversion = dict(zip(output_residue_ids, input_residue_ids))
        for old_id, new_id in id_conversion.items():
            atoms.res_id[chain_mask & (original_res_ids == old_id)] = new_id
    return atoms


def get_unique_residues(atom_array: AtomArray) -> list[tuple[int, str]]:
    residues: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for i in range(len(atom_array)):
        res_key = (int(atom_array.res_id[i]), str(atom_array.chain_id[i]))
        if res_key not in seen:
            seen.add(res_key)
            residues.append(res_key)
    return residues


### Reordering atoms to match ESMFold output ###
def reorder_atoms_in_template(atom_array: AtomArray) -> AtomArray:
    reordered_indices = []
    for res_id, chain_id in get_unique_residues(atom_array):
        indices = np.where((atom_array.res_id == res_id) & (atom_array.chain_id == chain_id))[0]
        atoms = atom_array[indices]

        # Skip non–amino-acid residues (e.g., HOH, ligands)
        res_name = atoms.res_name[0]
        if res_name not in aa_dict_3to1:
            logger.warning(f'Skipping non-amino-acid residue {res_name} (res_id={res_id}, chain_id={chain_id}).')
            continue

        # Filter and report atoms not in atom_order
        protein_mask = np.array([name in atom_order for name in atoms.atom_name])
        for name in atoms.atom_name[~protein_mask]:
            logger.warning(
                f"Removed non-protein atom '{name}' from residue {atoms.res_name[0]} "
                f'(res_id={res_id}, chain_id={chain_id}).'
            )

        # Reorder protein atoms
        protein_indices = indices[protein_mask]
        sort_keys = [atom_order[name] for name in atoms.atom_name[protein_mask]]
        sorted_indices = protein_indices[np.argsort(sort_keys)]
        reordered_indices.extend(sorted_indices)

    return atom_array[reordered_indices]


def validate_array_range(
    array: npt.NDArray[np.float64], field_name: str, min_val: float = 0, max_val: float = 1
) -> npt.NDArray[np.float64]:
    """
    Validates that an array is a numpy array and its values fall within the specified range.

    Parameters
    ----------
    array : npt.NDArray[np.float64]
        Array to validate
    field_name : str
        Name of the field for error messages
    min_val : float
        Minimum allowed value (inclusive), default 0
    max_val : float
        Maximum allowed value (inclusive), default 1

    Returns
    -------
    npt.NDArray[np.float64]
        The validated array

    Raises
    ------
    ValueError
        If array is not a numpy array or values are outside the specified range
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f'{field_name} must be a numpy array')
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f'{field_name} must have a numeric dtype, got {array.dtype}')
    if not np.all((array >= min_val) & (array <= max_val)):
        raise ValueError(f'All values in {field_name} must be between {min_val} and {max_val}')
    return array
