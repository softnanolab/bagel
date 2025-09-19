"""
Protein chain and residue representation module.

This module provides classes for representing and manipulating protein chains:
- Residue: Represents a single amino acid with properties like name and mutability
- Chain: Represents a sequence of residues with methods for mutation, addition, and removal

The module supports loading protein structures from PDB files and working with
amino acid sequences programmatically.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from dataclasses import dataclass
from typing import Self, List
import pathlib as pl
from biotite.structure.io.pdb import PDBFile
from biotite.structure import get_residues
from .constants import aa_dict


@dataclass
class Residue:
    """
    Standard amino acid object.

    Attributes:
        name: Type of the amino acid in one-letter format (upper-cased)
        chain_ID: ID of the polymer chain this residue belongs to
        index: Internal index of the residue in the chain (0 to len(chain)-1, i.e. 0-indexed)
        mutable: Whether the residue can be mutated

    Examples:
        >>> residue = Residue("A", "X", 0)
        >>> residue.three_letter_name
        'ALA'
    """

    name: str
    chain_ID: str
    index: int
    mutable: bool = True

    def __post_init__(self) -> None:
        """Validation checks for a residue."""
        assert self.name in aa_dict.keys(), f'Acceptable amino acids are {aa_dict.keys()} name found: {self.name}...'
        assert len(self.chain_ID) < 5, 'chain_ID must be less than 5 characters for compatibility with AtomArrays'

    @property
    def three_letter_name(self) -> str:
        """String representation of amino acid in 3-letters format."""
        return aa_dict[self.name]


@dataclass
class Chain:
    """
    Encodes residues and mutability of a monomeric chain, with functionality to read, write, and mutate the chain.

    Attributes:
        residues: List of Residue objects
    """

    residues: List[Residue]

    def __post_init__(self) -> None:
        """Used for sanity checks."""
        # make sure that the chain ID is same for all residues
        self.my_chain_ID = self.residues[0].chain_ID
        assert all(residue.chain_ID == self.chain_ID for residue in self.residues), (
            'chain_ID must be the same for all residues in the chain'
        )

    @property
    def chain_ID(self) -> str:
        """ID of the monomer chain."""
        return self.my_chain_ID

    @property
    def mutability(self) -> List[bool]:
        """List of mutability of each Residue in Chain."""
        return [residue.mutable for residue in self.residues]

    @property
    def sequence(self) -> str:
        """String (one-letter) representation of amino acids in Chain."""
        return ''.join([residue.name for residue in self.residues])

    @property
    def mutable_residues(self) -> List[Residue]:
        """List of mutable Residues in Chain"""
        return [residue for residue in self.residues if residue.mutable]

    @property
    def mutable_residue_indexes(self) -> List[int]:
        """List of the indexes of mutable Residues in Chain"""
        return [residue.index for residue in self.residues if residue.mutable]

    @property
    def length(self) -> int:
        """Number of amino acids in Chain."""
        return len(self.residues)

    @property
    def __len__(self) -> int:
        """Number of amino acids in Chain."""
        return self.length

    @classmethod
    def from_pdb(cls, file_path: str, chain_id: str) -> Self:
        """Create Chain object from a PDB file string. Residue indices are 0-indexed."""
        path = pl.Path(file_path).resolve()
        assert path.is_file() and (path.suffix == '.pdb'), f'Invalid file_path given for {file_path}'

        structure = PDBFile.read(path).get_structure(model=1)  # assumes only 1 protein in pdb
        assert chain_id in structure.chain_id, f'{chain_id} chain id not found in {file_path} pdb file'

        _, residue_names = get_residues(structure[structure.chain_id == chain_id])
        residue_names = [residue_name[0] for residue_name in residue_names]  # converts 3 letter residue names to 1
        residues = [Residue(residue_name, chain_id, index=i) for i, residue_name in enumerate(residue_names)]
        return cls(residues)

    @classmethod
    def from_cif(cls, cif_data: str) -> Self:
        """Create Chain object from a CIF file string."""
        raise NotImplementedError('CIF file support is not yet implemented')

    def remove_residue(self, index: int) -> None:
        """Remove the Residue at the given index (0-indexed)."""
        index = index if index >= 0 else len(self.residues) + index
        assert self.residues[index].mutable, AssertionError('Cannot delete immutable residue')
        self.residues.pop(index)
        for i in range(index, len(self.residues)):
            self.residues[i].index -= 1
        # Added consistency check to ensure that the indices are correct
        for i in range(len(self.residues)):
            assert self.residues[i].index == i, f'Index of residue {self.residues[i]} is not correct after deletion'

    def add_residue(self, amino_acid: str, index: int) -> None:
        """Add a Residue of type amino_acid at the position specified by index (0-indexed)."""
        index = index if index >= 0 else len(self.residues) + index
        assert index <= self.length, f'Invalid index for {self.length} length chain'
        assert amino_acid in aa_dict.keys(), f'Acceptable amino acids are {aa_dict.keys()}'
        chain_ID = self.residues[0].chain_ID
        self.residues.insert(index, Residue(name=amino_acid, chain_ID=chain_ID, index=index, mutable=True))
        for i in range(index + 1, len(self.residues)):
            self.residues[i].index += 1
        # Added consistency check to ensure that the indices are correct
        for i in range(len(self.residues)):
            assert self.residues[i].index == i, f'Index of residue {self.residues[i]} is not correct after addition'

    def mutate_residue(self, index: int, amino_acid: str) -> None:
        """Change identity of Residue at position specified by index to 'amino_acid'"""
        assert -self.length - 1 <= index <= self.length, f'Invalid index for {self.length} length chain'
        assert self.mutability[index] == 1, 'Index of selected Residue is not mutable'
        assert amino_acid in aa_dict.keys(), f'Acceptable amino acids are {aa_dict.keys()}'
        mutated_residue = self.residues[index]
        mutated_residue.name = amino_acid
        self.residues[index] = mutated_residue
