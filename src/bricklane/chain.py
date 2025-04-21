"""Standard objects to encode residues and shape of chains."""

from dataclasses import dataclass
from typing import Self, List
import pathlib as pl
from biotite.structure.io.pdb import PDBFile
from biotite.structure import get_residues
from .constants import aa_dict


@dataclass
class Residue:
    """Standard amino acid object."""

    name: str  # type of the aminoacid in 1 upper case letter format
    chain_ID: str  # ID of the polymer chain this residue belongs to
    index: int  # internal index of the residue in the chain (0 to len(chain)-1)
    mutable: bool = True  # whether the residue can be mutated

    def __post_init__(self) -> None:
        """Check that the residue is valid."""
        assert self.name in aa_dict.keys(), f'acceptable amino acids are {aa_dict.keys()} name found: {self.name}...'
        assert len(self.chain_ID) < 5, 'chain_ID must be less than 5 characters for compatibility atomarrays'

    @property
    def long_name(self) -> str:
        """String representation of amino acid in 3-letters format."""
        return aa_dict[self.name]


@dataclass
class Chain:
    """Encodes residues and mutability of a monomeric chain, with functionality to read, write, and mutate the chain."""

    residues: List[Residue]

    def __post_init__(self) -> None:
        """Used for sanity checks."""
        # This is to make sure that the chain ID is the same for all residues
        # and equal to that of the first residue
        self.my_chain_ID = self.residues[0].chain_ID
        assert all(residue.chain_ID == self.chain_ID for residue in self.residues), (
            'chain_ID must be the same for all residues in the chain'
        )

    @property
    def chain_ID(self) -> str:
        """ID of the polymer chain."""
        return self.my_chain_ID

    @property
    def mutability(self) -> List[bool]:
        """List of mutability of each residue in chain."""
        return [residue.mutable for residue in self.residues]

    @property
    def sequence(self) -> str:
        """String representation of amino acids in chain."""
        return ''.join([residue.name for residue in self.residues])

    @property
    def mutable_residues(self) -> List[Residue]:
        """list of mutable residues objects in protein"""
        return [residue for residue in self.residues if residue.mutable]

    @property
    def mutable_residue_indexes(self) -> List[int]:
        """list of the indexes of mutable residues in protein"""
        return [residue.index for residue in self.residues if residue.mutable]

    @property
    def length(self) -> int:
        """Number of amino acids in chain."""
        return len(self.residues)

    @classmethod
    def from_pdb(cls, file_path: str, chain_id: str) -> Self:
        """Create Chain object from a protein data bank string. Resets indices so they start at 0"""
        path = pl.Path(file_path).resolve()
        assert path.is_file() and (path.suffix == '.pdb'), 'invalid file_path given'

        structure = PDBFile.read(path).get_structure(model=1)  # assumes only 1 protein in pdb
        assert chain_id in structure.chain_id, f'{chain_id} chain id not found in {file_path} pdb file'

        _, residue_names = get_residues(structure[structure.chain_id == chain_id])
        residue_names = [residue_name[0] for residue_name in residue_names]  # converts 3 letter residue names to 1
        residues = [Residue(residue_name, chain_id, index=i) for i, residue_name in enumerate(residue_names)]
        return cls(residues)

    @classmethod
    def from_cif(cls, cif_data: str) -> Self:
        """Create Chain object from crystallographic information file string."""
        raise NotImplementedError

    def remove_residue(self, index: int) -> None:
        """Remove the residue at the given index."""
        index = index if index >= 0 else len(self.residues) + index
        assert self.residues[index].mutable, AssertionError('cannot delete immutable residue')
        self.residues.pop(index)
        for i in range(index, len(self.residues)):
            self.residues[i].index -= 1
        # Added consistency check to ensure that the indices are correct
        for i in range( len(self.residues) ):
            assert self.residues[i].index == i, f'index of residue {self.residues[i]} is not correct after deletion'

    def add_residue(self, amino_acid: str, index: int) -> None:
        """Add a residue of type amino_acid at the position specificed by index."""
        index = index if index >= 0 else len(self.residues) + index
        assert index <= self.length, f'invalid index for {self.length} length chain'
        assert amino_acid in aa_dict.keys(), f'acceptable amino acids are {aa_dict.keys()}'
        chain_ID = self.residues[0].chain_ID
        self.residues.insert(index, Residue(name=amino_acid, chain_ID=chain_ID, index=index, mutable=True))
        for i in range(index + 1, len(self.residues)):
            self.residues[i].index += 1
        # Added consistency check to ensure that the indices are correct
        for i in range( len(self.residues) ):
            assert self.residues[i].index == i, f'index of residue {self.residues[i]} is not correct after addition'

    def mutate_residue(self, index: int, amino_acid: str) -> None:
        """Change identity of residue at position specified by index to 'amino_acid'"""
        assert -self.length - 1 <= index <= self.length, f'invalid index for {self.length} length chain'
        assert self.mutability[index] == 1, 'index of selected residue is not mutable'
        assert amino_acid in aa_dict.keys(), f'acceptable amino acids are {aa_dict.keys()}'
        mutated_residue = self.residues[index]
        mutated_residue.name = amino_acid
        self.residues[index] = mutated_residue
