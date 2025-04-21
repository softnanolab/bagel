"""Standard template and objects for calculating structural or propery losses."""

from abc import ABC, abstractmethod
from biotite.structure import AtomArray, sasa, annotate_sse, superimpose
import numpy as np
import numpy.typing as npt
from .constants import hydrophobic_residues, max_sasa_values, probe_radius_water, backbone_atoms
from .chain import Residue
from .folding import FoldingMetrics
import warnings
import pandas as pd


# first row is chain_ids and second row is corresponding residue indices.
ResidueGroup = tuple[npt.NDArray[np.str_], npt.NDArray[np.int_]]


def residue_list_to_group(residues: list[Residue]) -> ResidueGroup:
    """Converts list of residue objects to ResidueGroup required by energy term objects"""
    return (np.array([res.chain_ID for res in residues]), np.array([res.index for res in residues]))


class EnergyTerm(ABC):
    """
    Standard energy term to build the loss (total energy) function to be minimized.
    Note that each energy term is a function of the structure and folding metrics.
    Also, note that each energy term has its own __init__ method, however, all common
    terms that must be initialized can be found in the __post__init__ function below.
    Like the __init__ method, __post__init__ is also **automatically** called upon 
    instantiating an object of the class. 
    """
    
    def __post_init__(self) -> None:
        """Checks required attributes have been set after class is initialised"""
        assert hasattr(self, 'name'), 'name attribute must be set in class initialiser'
        self.name: str = self.name
        assert hasattr(self, 'residue_groups'), 'residue_groups attribute must be set in class initialiser'
        self.residue_groups: list[ResidueGroup] = self.residue_groups
        self.value: float = 0.0
        assert hasattr(self, 'inheritable'), 'inheritable attribute must be set in class initialiser'
        if self.name == "template_match" or self.name == "backbone_template_match":
            assert self.inheritable is False, 'template_match energy term should NEVER be inheritable'

    @abstractmethod
    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        """
        Calculates the EnergyTerm's energy given information about the folded structure.
        The result is returned and stored as an internal attribute (.value).

        Parameters
        ----------
        structure: AtomArray
            An array containing positions and attributes of each atom in the structure.
        folding_metrics: FoldingMetrics
            Key statistics dataclass calculated from the state folding process.

        Returns
        -------
        energy : float
            How well the structure satisfies the given criteria. Where possible, this number should be between 0 and 1.
        """
        pass

    def track_residue_removed_from_chain(self, chain_id: str, res_index: int) -> None:
        """Shifts internally stored res_ids on a given chain to reflect a residue has been removed from chain.
        In practice, this means the indexes in residue_groups for all residues after the one removed it are 
        shifted down by 1. Must be called every time a residue is removed from a chain."""
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            shifted_mask = (chain_ids == chain_id) & (res_indices > res_index)
            self.residue_groups[i][1][shifted_mask] -= 1

    def track_residue_added_to_chain(self, chain_id: str, res_index: int) -> None:
        """Shifts internally stored res_indices on a given chain to reflect a residue has been added. 
        In practice, all residues with an index >= res_index are shifted by +1.
        Must be called every time a residue is added."""
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            shifted_mask = (chain_ids == chain_id) & (res_indices >= res_index)
            self.residue_groups[i][1][shifted_mask] += 1

    def remove_residue(self, chain_id: str, res_index: int) -> None:
        """Remove residue from this energy term's calculations.
        Helper function called by the state.remove_residue_from_all_energy_terms function."""
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            remove_mask = (chain_ids == chain_id) & (res_indices == res_index)
            self.residue_groups[i] = [chain_ids[~remove_mask], res_indices[~remove_mask]]  # type: ignore[call-overload]

    def add_residue(self, chain_id: str, res_index: int, parent_res_index: int) -> None:
        """Adds residue to this energy term's calculations, in the same group as its parent residue.
        Helper function called by the state.add_residue_from_all_energy_terms function."""
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            if any((chain_ids == chain_id) & (res_indices == parent_res_index)):
                self.residue_groups[i] = [np.append(chain_ids, chain_id), np.append(res_indices, res_index)]  # type: ignore[call-overload]

    def get_residue_mask(self, structure: AtomArray, residue_group_index: int) -> npt.NDArray[np.bool_]:
        """Creates residue mask from residue group. Structure used to find unique residues in state"""
        residue_group = self.residue_groups[residue_group_index]
        chain_ids, res_indices = residue_group
        residue_mask = np.array([], dtype=bool)
        for chain in pd.unique(structure.chain_id):  # preserves order of chains as fed to input (important)
            # Note: in an atom_array object like structure .res_id is what we call the residue index and is an integer
            chain_res_ids = pd.unique(structure.res_id[structure.chain_id == chain])  # preserves order of residues
            residue_mask = np.append(residue_mask, np.isin(chain_res_ids, res_indices[chain_ids == chain]))
        return residue_mask

    def get_atom_mask(self, structure: AtomArray, residue_group_index: int) -> npt.NDArray[np.bool_]:
        """Creates atom mask from residue group. Structure used to find unique atoms in state"""
        residue_group = self.residue_groups[residue_group_index]
        chain_ids, res_indices = residue_group
        atom_mask = np.full(shape=len(structure), fill_value=False)
        for chain in np.unique(chain_ids):
            chain_mask = structure.chain_id == chain
            # Note: in an atom_array object like structure .res_id is what we call the residue index and is an integer
            atom_mask[chain_mask] = np.isin(structure[chain_mask].res_id, res_indices[chain_ids == chain])
        return atom_mask


class PTMEnergy(EnergyTerm):
    """
    Predicted Template Modelling score energy. This is a measure of how confident the folding model is in its overall
    structure prediction. This translates to how similar a sequence's structure is to the low energy structures the
    model was trained on.
    """

    def __init__(self) -> None:
        """Initialises Predicted Template Modelling Score Energy class.

        Parameters
        ----------
        """
        self.name = 'pTM'
        self.inheritable = False
        self.residue_groups = []

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        assert hasattr(folding_metrics, 'ptm'), 'PTM metric not returned by folding algorith'
        self.value = -folding_metrics.ptm
        return self.value


class PLDDTEnergy(EnergyTerm):
    """
    Predicted Local Distance Difference Test energy. This is the spread of the predicted separation between an atom and
    each of its nearest neighbours. This translates to how confident the model is that the sequence has a single lowest
    energy structure, as opposed to a disordered, constantly changing structure. This energy is averaged over the
    relevant atoms.
    """

    def __init__(self, residues: list[Residue], inheritable: bool = True) -> None:
        """Initialises Local Predicted Local Distance Difference Test Energy class.

        Parameters
        ----------
        residues: list[Residue]
            Which residues to include in the calculation.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        self.name = 'local_pLDDT'
        self.inheritable = inheritable
        self.residue_groups = [residue_list_to_group(residues)]

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        assert hasattr(folding_metrics, 'local_plddt'), 'local_plddt metric not returned by folding algorithm'
        plddt = folding_metrics.local_plddt[0]  # [n_residues] array
        mask = self.get_residue_mask(structure, residue_group_index=0)
        self.value = -np.mean(plddt[mask])
        return self.value


class OverallPLDDTEnergy(EnergyTerm):
    """
    Overall Predicted Local Distance Difference Test energy. This is the spread of the predicted separation between an
    atom and each of its nearest neighbours. This translates to how confident the model is that the sequence has a
    single lowest energy structure, as opposed to a disordered, constantly changing structure. This energy is averaged
    over all atoms.
    """

    def __init__(self) -> None:
        """Initialises Overall Predicted Local Distance Difference Test Energy class.

        Parameters
        ----------
        """
        self.name = 'global_pLDDT'
        self.inheritable = False
        self.residue_groups = []

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        assert hasattr(folding_metrics, 'local_plddt'), 'local_plddt metric not returned by folding algorithm'
        plddt = folding_metrics.local_plddt[0]  # [n_residues] array
        self.value = -np.mean(plddt)
        return self.value


class SurfaceAreaEnergy(EnergyTerm):
    """
    Energy term proportional to the amount of exposed surface area. This is measured by dividing the mean SASA
    (Solvent Accessible Surface Area) of the relevant atoms by the maximum possible SASA.
    """

    def __init__(
        self,
        residues: list[Residue] | None = None,
        inheritable: bool = True,
        probe_radius: float | None = None,
        max_sasa: float | None = None,
    ) -> None:
        """
        Initialises Surface Area Energy Class.

        Parameters
        ----------
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. Considers all residues by default.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        probe_radius: float or None, default=None
            The VdW-radius of the solvent molecules used in the SASA calculation. Default is the water VdW-radius.
        max_sasa: float or None, default=None
            The maximum SASA value used if normalization is enabled. Default is the full surface area of a Sulfur atom.
        """
        self.name = f'{"selective_" if residues is not None else ""}surface_area'
        self.inheritable = inheritable
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.probe_radius = probe_radius_water if probe_radius is None else probe_radius
        self.max_sasa = max_sasa_values['S'] if max_sasa is None else max_sasa

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        if len(self.residue_groups) != 0:
            atom_mask: npt.NDArray[np.bool_] = self.get_atom_mask(structure, residue_group_index=0)
        else:
            atom_mask = np.full(shape=len(structure), fill_value=True)

        sasa_values = sasa(structure, probe_radius=self.probe_radius)
        self.value = np.mean(sasa_values[atom_mask]) / self.max_sasa
        return self.value


class HydrophobicEnergy(EnergyTerm):
    """
    Energy Proportional to the amount of hydrophobic residues present. This is measured by the fraction of selected
    atoms that belong to hydrophobic residues (valine, isoleucine, leucine, phenylalanine, methionine, tryptophan).
    """

    def __init__(
        self,
        residues: list[Residue] | None = None,
        inheritable: bool = True,
        surface_only: bool = False,
    ) -> None:
        """
        Initialises hydrophobic energy class.

        Parameters
        ----------
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. If not set, simply considers **all** residues by default.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        surface_only: bool
            Whether to only consider the atoms exposed to water at the surface. If False, interior atoms are included
            in the calculation. If true, result is scaled by normalised solute accessible surface area values.
        """
        selective = 'selective_' if residues is not None else ''
        surface = 'surface_' if surface_only else ''
        self.name = f'{selective}{surface}hydrophobic'
        self.inheritable = inheritable
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.surface_only = surface_only

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        if len(self.residue_groups) > 0:
            relevance_mask: npt.NDArray[np.bool_] = self.get_atom_mask(structure, residue_group_index=0)
        else:
            relevance_mask = np.full(shape=len(structure), fill_value=True)

        hydrophobic_mask = np.isin(structure.res_name, hydrophobic_residues)

        self.value = len(structure[relevance_mask & hydrophobic_mask]) / len(structure[relevance_mask])
        if self.surface_only:
            normalized_sasa = sasa(structure, probe_radius=probe_radius_water) / max_sasa_values['S']
            self.value *= np.mean(normalized_sasa[relevance_mask & hydrophobic_mask])

        return self.value


class PAEEnergy(EnergyTerm):
    """
    Energy that drives down the uncertainty in the predicted distances between two groups of residues. This uncertainty
    is measured by calculating the average normalised predicted alignment error of all the relevant residue pairs.
    """

    def __init__(
        self,
        group_1_residues: list[Residue],
        group_2_residues: list[Residue] | None = None,
        cross_term_only: bool = True,
        inheritable: bool = True,
    ) -> None:
        """
        Initialises the alignment error energy class.

        Parameters
        ----------
        group_1_residues: list[Residue]
            Which residues to include in the first group.
        group_2_residues: list[Residue] or None, default=None
            Which residues to include in the second group. If set to None, will use the same residues as in group 1.
        cross_term_only: bool, default=True
            Whether to only consider the uncertainty in distance between group 1 and group 2 atoms. If set to False,
            also considers the uncertainty in distances between atoms within the same group.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        self.name = f'{"cross_" if cross_term_only else ""}alignment_error'
        self.inheritable = inheritable
        group_2_residues = group_1_residues if group_2_residues is None else group_2_residues
        self.residue_groups = [residue_list_to_group(group_1_residues), residue_list_to_group(group_2_residues)]
        self.cross_term_only = cross_term_only

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        assert hasattr(folding_metrics, 'pae'), 'pae metric not returned by folding algorith'
        pae = folding_metrics.pae[0]  # [n_residues, n_residues] pairwise predicted alignment error matrix
        max_pae = 30  # approximate max. Sometimes pae can be higher

        group_1_mask = self.get_residue_mask(structure, residue_group_index=0)
        group_2_mask = self.get_residue_mask(structure, residue_group_index=1)
        pae_mask = np.full(shape=pae.shape, fill_value=False)

        if self.cross_term_only:  # only PAEs between an atom in group 1 and an atom in group 2
            pae_mask[group_1_mask[:, np.newaxis] & group_2_mask[np.newaxis, :]] = True
            # in case PAE symmetry is not enforced
            pae_mask[group_2_mask[:, np.newaxis] & group_1_mask[np.newaxis, :]] = True
        else:  # cross term PAEs plus PAEs between atoms in the same group
            pae_mask[(group_1_mask | group_2_mask)[:, np.newaxis] & (group_1_mask | group_2_mask)[np.newaxis, :]] = True

        diagonal_mask = np.eye(len(pae), dtype=bool)
        pae_mask[diagonal_mask] = False  # should ignore uncertainty in distance between atom and itself

        self.value = np.mean(pae[pae_mask]) / max_pae
        return self.value


class PAEEnergyV2(EnergyTerm):
    """
    Energy that drives down the uncertainty in the predicted distances between two groups of residues. This uncertainty
    is measured by calculating the average normalised predicted alignment error of all the relevant residue pairs.
    """

    def __init__(
        self,
        group_1_residues: list[Residue],
        group_2_residues: list[Residue] | None = None,
        cross_term_only: bool = True,
        inheritable: bool = True,
    ) -> None:
        """
        Initialises the alignment error energy class.

        Parameters
        ----------
        group_1_residues: list[Residue]
            Which residues to include in the first group.
        group_2_residues: list[Residue] or None, default=None
            Which residues to include in the second group. If set to None, will use the same residues as in group 1.
        cross_term_only: bool, default=True
            Whether to only consider the uncertainty in distance between group 1 and group 2 atoms. If set to False,
            also considers the uncertainty in distances between atoms within the same group.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        warnings.warn(message='This class has known issues and is to be removed', category=DeprecationWarning)
        self.name = f'{"cross_" if cross_term_only else ""}alignment_error'
        self.inheritable = inheritable
        group_2_residues = group_1_residues if group_2_residues is None else group_2_residues
        self.residue_groups = [residue_list_to_group(group_1_residues), residue_list_to_group(group_2_residues)]
        self.cross_term_only = cross_term_only

    def find_residue_in_pae(self, chain_id: str, res_id: int, ordered_chain_lengths: list[tuple[str, int]]) -> int:
        """Finds the number of a residue in the PAE matrix
        This assume ordered_chain_lengths[ ( chain_id, chain_length) ] is ordered in the same
        way the different chain_ids appear in the PDB/CIF file, which must be the same way they have
        been fed to the folding algorithm.
        NOTE: it is important that ordered_chain_length is a list of tuple (with an order that matters)
        and not a dictionary because the order of the chains in the PDB/CIF file is important.
        """
        offset = 0
        print(f'ordered_chain_lengths = {ordered_chain_lengths}')
        for id, length in ordered_chain_lengths:
            # print( f"current id = {id}, length = {length} chain_id = {chain_id}" )
            if id == chain_id:
                break
            offset += length
        # print( f"chain_id = {chain_id}, res_id = {res_id}, offset = {offset}" )
        return offset + res_id

    def structure_to_chain_lengths(self, structure: AtomArray) -> list[tuple[str, int]]:
        ordered_chain_ids_list = pd.unique(structure.chain_id)
        # print( "ordered_chain_ids_list: ", ordered_chain_ids_list )
        ordered_chain_lengths = []
        for chain_id in ordered_chain_ids_list:
            chain_structure = structure[structure.chain_id == chain_id]
            # In the next line, we are getting the unique residue ids for the chain
            chain_res_ids = pd.unique(chain_structure.res_id)
            ordered_chain_lengths.append((chain_id, len(chain_res_ids)))
        return ordered_chain_lengths

    def calculate_pae_mask(self, structure: AtomArray) -> npt.NDArray[np.bool_]:
        """Creates residue mask from residue group to be used in pae."""
        residue_group_1 = self.residue_groups[0]
        residue_group_2 = self.residue_groups[1]
        chain_ids_1, res_ids_1 = residue_group_1
        chain_ids_2, res_ids_2 = residue_group_2

        # First constructe an array with the correct dimensions.
        # It has to be N x N where N is the number of residues in the structure.
        ordered_chain_lengths = self.structure_to_chain_lengths(structure)
        N = np.sum([length for chain_id, length in ordered_chain_lengths])
        residue_mask_1 = np.zeros((N, N), dtype=bool)
        residue_mask_2 = np.zeros((N, N), dtype=bool)

        # Now mask the residues that are present in residue_group_1
        for i in range(len(chain_ids_1)):
            chain_id = chain_ids_1[i]
            res_id = res_ids_1[i]
            residue_id = self.find_residue_in_pae(chain_id, res_id, ordered_chain_lengths)
            residue_mask_1[residue_id, :] = True
        # Now mask the residues that are present in residue_group_2
        for j in range(len(chain_ids_2)):
            chain_id = chain_ids_2[j]
            res_id = res_ids_2[j]
            residue_id = self.find_residue_in_pae(chain_id, res_id, ordered_chain_lengths)
            residue_mask_2[:, residue_id] = True

        # Now we have to combine the two masks
        if self.cross_term_only:
            # only PAEs between an atom in group 1 and an atom in group 2
            pae_mask = residue_mask_1 * residue_mask_2
        else:
            # cross term PAEs plus PAEs between atoms in the same group
            pae_mask = residue_mask_1 + residue_mask_2
            pae_mask = pae_mask > 0
        return pae_mask

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        assert hasattr(folding_metrics, 'pae'), 'pae metric not returned by folding algorith'
        pae = folding_metrics.pae[0]  # [n_residues, n_residues] pairwise predicted alignment error matrix
        assert len(pae.shape) == 2, 'PAE matrix has incorrect shape'
        max_pae = 30  # NOTE: this is an approximate maximum value for the PAE, not exactly the maximum
        pae_mask = self.calculate_pae_mask(structure)
        self.value = np.mean(pae[pae_mask]) / max_pae
        return self.value


class RingSymmetryEnergy(EnergyTerm):
    """
    Energy that maximises the symmetry of different groups. Symmetry is measured by finding the centroid of the backbone
    of each group and checking how consistently they are spaced from one another.
    """

    def __init__(
        self,
        symmetry_groups: list[list[Residue]],
        direct_neighbours_only: bool = False,
        inheritable: bool = True,
    ) -> None:
        """Initialises ring symmetry energy class.

        Parameters
        ----------
        *symmetry_groups: list[Residue]
            A list of at least length 2, with each element containing a list of residues corresponding to a symmetry
            group.
        direct_neighbours_only: bool, default=False
            Whether to compare the spacing of each each group to its direct neighbour (compare group i to group i+1
            only), or each group to all other groups. Defaults to the latter.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        self.name = f'{"neighbour_" if direct_neighbours_only else ""}ring_symmetry'
        self.inheritable = inheritable
        assert (len(symmetry_groups) > 1) and (len(symmetry_groups[0]) >= 1), 'Multiple symmetry groups required.'
        self.residue_groups = [residue_list_to_group(symmetry_group) for symmetry_group in symmetry_groups]
        self.direct_neighbours_only: bool = direct_neighbours_only

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        num_groups = len(self.residue_groups)
        centroids = np.zeros(shape=(num_groups, 3))
        backbone_mask = np.isin(structure.atom_name, backbone_atoms)

        for i in range(num_groups):
            symmetry_group_mask = self.get_atom_mask(structure, residue_group_index=i)
            centroids[i] = np.mean(structure[symmetry_group_mask & backbone_mask].coord, axis=0)
        if self.direct_neighbours_only:
            neighbour_displacements = centroids - np.roll(centroids, shift=1, axis=0)
            neighbour_distances = np.linalg.norm(neighbour_displacements, axis=1)
            self.value = np.std(neighbour_distances)
        else:
            displacement_matrix = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distance_matrix = np.linalg.norm(displacement_matrix, axis=2)
            unique_distances = distance_matrix[~np.tri(N=num_groups, dtype=bool)]
            self.value = np.std(unique_distances)

        return self.value


class SeparationEnergy(EnergyTerm):
    """
    Energy that minimizes the distance between two groups of residues. The position of each group is 
    defined as the centroid of the backbone atoms of the residues belonging of that group.
    """

    def __init__(
        self,
        group_1_residues: list[Residue],
        group_2_residues: list[Residue],
        normalize: bool = True,
        inheritable: bool = True,
    ) -> None:
        """
        Initialises separation energy class.

        Parameters
        ----------
        group_1_residues: list[Residue]
            Which residues to include in the first group.
        group_2_residues: list[Residue]
            Which residues to include in the second group.
        normalize: bool, default=True
            Whether the distance calculated is divided by the number of atoms in both groups.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        self.name = f'{"normalized_" if normalize else ""}separation'
        self.inheritable = inheritable
        self.residue_groups = [residue_list_to_group(group_1_residues), residue_list_to_group(group_2_residues)]
        self.normalize = normalize

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        backbone_mask = np.isin(structure.atom_name, backbone_atoms)
        group_1_mask = self.get_atom_mask(structure, residue_group_index=0)
        group_2_mask = self.get_atom_mask(structure, residue_group_index=1)

        group_1_atoms = structure[backbone_mask & group_1_mask]
        group_2_atoms = structure[backbone_mask & group_2_mask]
        group_1_centroid = np.mean(group_1_atoms.coord, axis=0)
        group_2_centroid = np.mean(group_2_atoms.coord, axis=0)
        distance = np.linalg.norm(group_1_centroid - group_2_centroid)

        if self.normalize:
            distance /= len(group_1_atoms) + len(group_2_atoms)

        self.value = distance  # type: ignore
        return self.value


class GlobularEnergy(EnergyTerm):
    """
    Energy proportional to the moment of inertia of the structure around its centroid. This energy is minimized when 
    the atoms belonging to a structure have the lowest possible distance from the centre, and, due to excluded volume
    effects that prevent collapse to a single point, helps forcing structures to 
    be as close as possible to a spherically distributed cloud of points. 
    """

    def __init__(self, residues: list[Residue] | None = None, normalize: bool = True, inheritable: bool = True) -> None:
        """
        Initialises globular energy class.

        Parameters
        ----------
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. Considers all residues by default.
        normalize: bool, default=True
            Whether the mean centroid distance calculated is divided by the number of atoms considered.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        self.name = f'{"normalized_" if normalize else ""}globular'
        self.inheritable = inheritable
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.normalize = normalize

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        backbone_mask = np.isin(structure.atom_name, backbone_atoms)
        if len(self.residue_groups) > 0:
            selected_mask = self.get_atom_mask(structure, residue_group_index=0)
        else:
            selected_mask = np.full(shape=len(structure), fill_value=True)

        relevant_atoms = structure[backbone_mask & selected_mask]
        centroid = np.mean(relevant_atoms.coord, axis=0, keepdims=True)
        centroid_distances = np.linalg.norm(relevant_atoms.coord - centroid, axis=1)
        if self.normalize:
            centroid_distances /= len(relevant_atoms)

        self.value = np.std(centroid_distances)
        return self.value


class TemplateMatchEnergy(EnergyTerm):
    """
    Energy that drives the structure to match an input-provided template. The difference with the template is always calculated
    by automatically considering the rotation and translation that best maximize the overlap with the template.
    """

    def __init__(
        self,
        template_atoms: AtomArray,
        residues: list[Residue],
        backbone_only: bool = False,
        distogram_separation: bool = False,
    ) -> None:
        """
        Initialises template match energy class.

        Parameters
        ----------
        template_atoms: AtomArray
            An array of atoms that specify the desired positions of the structure.
        residues: list[Residue]
            Which residues in the structure to compare to the template.
        backbone_only: bool, default=False
            Whether to only consider backbone atoms in the template and strucutre. Considers all atoms by default.
        distogram_separation: bool, default=False
            Whether strucutre - template separation is measured by taking the root mean square of the difference between
            the two pairwise distance matrices. By default, the root mean square of the difference in positions is used
            instead.
        """
        self.name = f'{"backbone_" if backbone_only else ""}template_match'
        self.residue_groups = [residue_list_to_group(residues)]
        self.template_atoms = template_atoms
        self.backbone_only = backbone_only
        self.distogram_separation = distogram_separation
        self.inheritable = False 

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        structure_atoms = structure[self.get_atom_mask(structure, residue_group_index=0)]
        template_atoms = self.template_atoms
        if self.backbone_only:
            structure_atoms = structure_atoms[np.isin(structure_atoms.atom_name, backbone_atoms)]
            template_atoms = template_atoms[np.isin(template_atoms.atom_name, backbone_atoms)]
        assert len(structure_atoms) == len(template_atoms), 'Different number of atoms in template and given residues'
        template_atoms = superimpose(fixed=structure_atoms, mobile=template_atoms)[0]  # tranlsation and rotation fit

        if not self.distogram_separation:
            distances = np.linalg.norm(structure_atoms.coord - template_atoms.coord, axis=1)
            separation = np.mean(distances**2) ** 0.5
        else:
            structure_disp_matrix = structure_atoms.coord[:, np.newaxis, :] - structure_atoms.coord[np.newaxis, :, :]
            structure_distance_matrix = np.linalg.norm(structure_disp_matrix, axis=2)
            template_disp_matrix = template_atoms.coord[:, np.newaxis, :] - template_atoms.coord[np.newaxis, :, :]
            template_distance_matrix = np.linalg.norm(template_disp_matrix, axis=2)

            distance_matrix_difference = structure_distance_matrix - template_distance_matrix
            unique_distance_matrix_differences = distance_matrix_difference[~np.tri(N=len(template_atoms), dtype=bool)]
            separation = np.mean(unique_distance_matrix_differences**2) ** 0.5

        self.value = separation
        return self.value


class SecondaryStructureEnergy(EnergyTerm):
    """
    Energy that drives the secondary structure of the selected residues to a given type. Calculated by finding the
    fraction of selected residues with the wrong secondary structure. Secondary structure types include alpha-helix,
    beta-sheet, and coil.
    """

    def __init__(self, residues: list[Residue], target_secondary_structure: str, inheritable: bool = True) -> None:
        """
        Initialises the secondary structure energy class.

        Parameters
        ----------
        residues: list[Residue]
            Which residues to include in the calculation.
        target_secondary_structure: str
            Which secondary structure type to drive towards. Options are 'alpha-helix', 'beta-sheet', or 'coil'.
        inheritbale: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        self.name = f'{target_secondary_structure.lower()}'
        self.inheritable = inheritable
        self.residue_groups = [residue_list_to_group(residues)]
        options = ('alpha-helix', 'beta-sheet', 'coil')
        assert target_secondary_structure in options, f'{target_secondary_structure} not recognised. options: {options}'
        self.target_secondary_structure = target_secondary_structure

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        target_label = self.target_secondary_structure[0]  # How Biotite labels secondary structures
        calculated_labels = annotate_sse(structure)
        selection_mask = self.get_residue_mask(structure, residue_group_index=0)

        self.value = np.mean(calculated_labels[selection_mask] != target_label)
        return self.value


class EllipsoidEnergy(EnergyTerm):
    """
    Energy that drives the overall shape of the structure towards an ellipsoid of given aspect ratio. This is found by
    finding the principle cartesian axes of the backbone atoms using principle component analysis. The positions of the
    atoms in this new basis are then inserted into the standard equation of an ellipsoid (x/a + y/b + z/c - 1 = 0, where
    a, b, and c are the ideal length, width, and depth of the desired ellipsoid with a volume equal to a sphere with a
    radius equal to the radius of gyration). If the atoms lie outside of this ellipsoid (the left hand side is greater
    than 0), the atoms experience a spring force type energy that attracts them back into the ellipsoid. All atoms also
    experience a recipricol exponential type repulsive energy that evenly distributes them inside the volume.
    """

    def __init__(
        self,
        aspect_ratio: tuple[float, float, float],
        k_attractive: float = 1.0,
        k_repulsive: float = 10.0,
    ) -> None:
        """
        Initializes elipsoid energy class.

        Parameters
        ----------
        aspect_ratio: tuple[float, float, float]
            The desired relative length, width, and height of the ellipsoid.
        k_attractive: float, default=1.0
            Constant of proportionality for the spring force type energy that attracts backbone atoms outside of the
            ellipsoid back into it.
        k_repulsive: float, default=10.0
            Constant of proportionality for the recipricol exponential type repulsive energy that evenly distributes
            all backbone atoms within the ellipsoid volume.
        """
        self.name = 'ellipsoid'
        self.inheritable = False  # always considers all residues in structure
        self.residue_groups = []
        self.k_attractive = k_attractive
        self.k_repulsive = k_repulsive

        # largest aspect ratio first (corresponds to the first principle axis that captures the most positional variance)
        self.aspect_ratio = np.sort(aspect_ratio)[::-1]
        self.aspect_ratio /= np.prod(self.aspect_ratio) ** (1 / 3)  # ensures product of aspect ratios equals 1
        warnings.warn('This energy is yet to have any unit tests and is not guaranteed to work')

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        atoms = structure[np.isin(structure.atom_name, backbone_atoms)]

        # transforming to principle component axes
        centered_coords = atoms.coord - np.mean(atoms.coord, axis=0, keepdims=True)
        variances, basis_vectors = np.linalg.eigh(np.cov(centered_coords, rowvar=False))  # PCA step
        # first principle axis should encode the most positional variance
        sorted_basis_vectors = basis_vectors[:, np.argsort(variances)[::-1]]  # each col in basis_vectors in a vector
        transformed_coords = centered_coords @ sorted_basis_vectors  # basis transformation
        x, y, z = transformed_coords.T

        # calculating ellipsoid dimensions with desired aspect ratios and volume
        bond_length = 1.5  # standard distance in angstroms between each backbon atom
        radius_of_gyration = np.sqrt(len(atoms) / 6) * bond_length  # standard value for theta conditions
        # makes 4/3 * π * radius of gyration cubed = 4/3 * π * a * b * c
        a, b, c = self.aspect_ratio * radius_of_gyration  # ensures a*b*c = radius of gyration cubed

        # calculating attractive energy
        relative_distances = x / a + y / b + z / c - 1
        outer_mask = relative_distances > 0  # mask for all atoms that lie outside of ellipsoid
        attractive_energy = np.sum(self.k_attractive * relative_distances[outer_mask])

        # calculating repulsive energy
        pairwise_displacement_matrix = transformed_coords[:, np.newaxis, :] - transformed_coords[np.newaxis, :, :]
        pairwise_distance_matrix = np.linalg.norm(pairwise_displacement_matrix, axis=2)
        repulsive_energy = self.k_repulsive * np.exp(-pairwise_distance_matrix / max([a, b, c])).mean()

        self.value = attractive_energy + repulsive_energy
        return self.value


class CuboidEnergy(EnergyTerm):
    """
    Energy that drives the overall shape of the structure towards an supercube of given dimensions. This is found by
    finding the principle cartesian axes of the backbone atoms using principal component analysis. The positions of the
    atoms in this new basis are then inserted into the standard equation of a cuboid ((x/a)^2n + (y/b)^2n + (z/c)^2n - 1
    = 0, where a, b,and c are the ideal length, width, and depth of the desired cuboid with a volume equal to a sphere
    with a radius equal to the radius of gyration and n is the vertex sharpness). If the atoms lie outside of this
    cuboid (the left hand side is greater than 0), the atoms experience a spring force type energy that attracts them
    back into the cuboid. All atoms also experience a pairwise exponential type repulsive energy that disfavours atoms 
    to be too close to each other, so that they should evenly distributes them inside the supercube volume.
    """

    def __init__(
        self,
        aspect_ratio: tuple[float, float, float],
        k_attractive: float = 1.0,
        k_repulsive: float = 10.0,
        sharpness: float = 2.0,
    ) -> None:
        """
        Initializes cuboid energy class.

        Parameters
        ----------
        aspect_ratio: tuple[float, float, float]
            The desired relative length, width, and height of the cuboid.
        k_attractive: float, default=1.0
            Constant of proportionality for the spring force type energy that attracts backbone atoms outside of the
            cuboid back into it.
        k_repulsive: float, default=10.0
            Constant of proportionality for the recipricol exponential type repulsive energy that evenly distributes
            all backbone atoms within the cuboid volume.
        sharpness: float, default=2.0
            A metric for how sharp the coreners of the cuboid should ideally be. The higher the value, the closer to a
            perfect right angled vertex.
        """
        self.name = 'cuboid'
        self.inheritable = False  # always considers all residues in structure
        self.residue_groups = []
        self.k_attractive = k_attractive
        self.k_repulsive = k_repulsive
        self.sharpness = sharpness

        # largest aspect ratio first (as first principle axis captures the most positional variance)
        self.aspect_ratio = np.sort(aspect_ratio)[::-1]
        self.aspect_ratio /= np.prod(self.aspect_ratio) ** (1 / 3)  # ensures product of aspect ratios equals 1
        warnings.warn('This energy is yet to have any unit tests and is not guaranteed to work')

    def compute(self, structure: AtomArray, folding_metrics: FoldingMetrics) -> float:
        atoms = structure[np.isin(structure.atom_name, backbone_atoms)]

        # transforming to principle component axes
        centered_coords = atoms.coord - np.mean(atoms.coord, axis=0, keepdims=True)
        variances, basis_vectors = np.linalg.eigh(np.cov(centered_coords, rowvar=False))  # PCA step
        # firts principle axis should encode the most positional variance
        sorted_basis_vectors = basis_vectors[:, np.argsort(variances)[::-1]]  # each col in basis_vectors in a vector
        transformed_coords = centered_coords @ sorted_basis_vectors  # basis transformation
        x, y, z = transformed_coords.T

        # calculating cuboid dimensions with desired aspect ratios and volume
        monomer_separation = 5  # standard distance in angstroms between monomers
        radius_of_gyration = np.sqrt(len(atoms) / 6) * monomer_separation  # standard value for theta conditions
        # makes 4/3 * π * radius of gyration cubed = a * b * c
        a, b, c = self.aspect_ratio * radius_of_gyration * (np.pi * 4 / 3) ** (1 / 3)

        # calculating attractive energy
        n = self.sharpness
        relative_distances = (x / a) ** (2 * n) + (y / b) ** (2 * n) + (z / c) ** (2 * n) - 1
        outer_mask = relative_distances > 0  # mask for all atoms that lie outside of cuboid
        attractive_energy = np.sum(self.k_attractive * relative_distances[outer_mask])

        # calculating repulsive energy
        pairwise_displacement_matrix = transformed_coords[:, np.newaxis, :] - transformed_coords[np.newaxis, :, :]
        pairwise_distance_matrix = np.linalg.norm(pairwise_displacement_matrix, axis=2)
        repulsive_energy = self.k_repulsive * np.exp(-pairwise_distance_matrix / max([a, b, c])).mean()

        self.value = attractive_energy + repulsive_energy
        return self.value
