"""
Standard template and objects for calculating structural or propery losses.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import ABC, abstractmethod
from biotite.structure import AtomArray, sasa, annotate_sse, superimpose
import numpy as np
import numpy.typing as npt
from .constants import hydrophobic_residues, max_sasa_values, probe_radius_water, backbone_atoms
from .chain import Residue, Chain
import warnings
import pandas as pd
from .oracles import Oracle, OracleResult, OraclesResultDict
from .oracles.folding import FoldingResult, FoldingOracle
from .oracles.embedding import EmbeddingResult, EmbeddingOracle


# first row is chain_ids and second row is corresponding residue indices.
ResidueGroup = tuple[npt.NDArray[np.str_], npt.NDArray[np.int_]]


def residue_list_to_group(residues: list[Residue]) -> ResidueGroup:
    """Converts list of residue objects to ResidueGroup required by energy term objects"""
    return (np.array([res.chain_ID for res in residues]), np.array([res.index for res in residues]))


# TODO: add weight attributes here to the energy terms
class EnergyTerm(ABC):
    """
    Standard energy term to build the loss (total energy) function to be minimized.
    Note that each energy term is a function of the structure and folding metrics.
    Also, note that each energy term has its own __init__ method, however, all common
    terms that must be initialized can be found in the __post__init__ function below.
    Like the __init__ method, __post__init__ is also **automatically** called upon
    instantiating an object of the class.
    """

    def __init__(
        self,
        name: str,
        oracle: Oracle,
        inheritable: bool,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises EnergyTerm class.

        For development purposes, follow the order convention of the __init__ method. `name` is defined
        within the __init__ method, it's not an argument. `oracle` is always passed as an argument. `inheritable`
        is passed in depending on the energy term. `weight` is last, as it's optional.

        Parameters
        ----------
        name: str
            The name of the energy term.
        oracle: Oracle
            The oracle to use for the energy term.
        inheritable: bool
            Whether the energy term is inheritable.
        weight: float = 1.0
            The weight of the energy term.
        """
        self.name = name
        self.oracle = oracle
        self.weight = weight
        self.inheritable = inheritable
        self.residue_groups: list[ResidueGroup] = []

    def __post_init__(self) -> None:
        # TODO: add general assertion checks for any energy term (0-body, 1-body, etc.)
        """Checks required attributes have been set after class is initialised"""
        assert hasattr(self, 'name'), 'name attribute must be set in class initialiser'
        assert hasattr(self, 'residue_groups'), 'residue_groups attribute must be set in class initialiser'
        assert hasattr(self, 'inheritable'), 'inheritable attribute must be set in class initialiser'
        if self.name == 'template_match' or self.name == 'backbone_template_match':
            assert self.inheritable is False, 'template_match energy term should NEVER be inheritable'

        assert self.oracle is not None, 'oracle attribute must be set in class initialiser'
        assert isinstance(self.oracle, Oracle), 'oracle attribute must be an instance of Oracle'

    @abstractmethod
    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        """
        Calculates the EnergyTerm's energy given information about the folded structure.
        The result is returned and stored as an internal attribute (.value).

        Parameters
        ----------
        oracles_result: OraclesResultDict
            Dictionary mapping oracles to their results. This is used to get the relevant
            information for the energy term.

        Returns
        -------
        (unweighted_energy, weighted_energy) : tuple[float, float]
            unweighted_energy : float
                How well the structure satisfies the given criteria. Where possible, this number should be between 0 and 1.
            weighted_energy : float
        The unweighted energy multiplied by the energy term's weight.
        """
        pass

    def shift_residues_indices_after_removal(self, chain_id: str, res_index: int) -> None:
        """
        Shifts internally stored res_indices on a given chain to reflect a residue has been removed from the chain.

        In practice, this means the indexes in ``residue_groups`` for all residues after the one removed are
        shifted down by 1. Must be called every time a residue is removed from a chain.

        For instance, if implementing a new mutation scheme in ``mutation.py``, this method must be called every time
        a residue is removed from a chain (see :class:`~bagel.mutation.GrandCanonical` for an example).
        """
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            shifted_mask = (chain_ids == chain_id) & (res_indices > res_index)
            self.residue_groups[i][1][shifted_mask] -= 1

    def shift_residues_indices_before_addition(self, chain_id: str, res_index: int) -> None:
        """
        Shifts internally stored res_indices on a given chain to reflect a residue has been added.
        In practice, all residues with an index >= res_index are shifted by +1.
        Must be called every time a residue is added.
        """
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            shifted_mask = (chain_ids == chain_id) & (res_indices >= res_index)
            self.residue_groups[i][1][shifted_mask] += 1

    def remove_residue(self, chain_id: str, res_index: int) -> None:
        """
        Remove residue from this energy term's calculations.
        Helper function called by the state.remove_residue_from_all_energy_terms function.
        """
        for i, residue_group in enumerate(self.residue_groups):
            chain_ids, res_indices = residue_group
            remove_mask = (chain_ids == chain_id) & (res_indices == res_index)
            self.residue_groups[i] = [chain_ids[~remove_mask], res_indices[~remove_mask]]  # type: ignore[call-overload]

    def add_residue(self, chain_id: str, res_index: int, parent_res_index: int) -> None:
        """
        Adds residue to this energy term's calculations, in the same group as its parent residue.
        Helper function called by the state.add_residue_from_all_energy_terms function.
        """
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
        # TODO: Is this possibly causing an issue with the sorting? Need a unit test for this.
        # Check whether pd.unique is necessary here.
        for chain in np.unique(chain_ids):
            chain_mask = structure.chain_id == chain
            # Note: in an atom_array object like structure .res_id is what we call the residue index and is an integer
            atom_mask[chain_mask] = np.isin(structure[chain_mask].res_id, res_indices[chain_ids == chain])
        return atom_mask


class PTMEnergy(EnergyTerm):
    """
    Predicted Template Modelling score energy. This is a measure of how confident the folding model is in its overall
    structure prediction.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises Predicted Template Modelling Score Energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        weight: float
            The weight of the energy term.
        """
        super().__init__(name='pTM', inheritable=True, oracle=oracle, weight=weight)
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'ptm' in self.oracle.result_class.model_fields, 'PTMEnergy requires oracle to return ptm in result_class'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        folding_result = oracles_result[self.oracle]
        assert hasattr(folding_result, 'ptm'), 'PTM metric not returned by folding algorithm'
        value = -folding_result.ptm
        return value, value * self.weight


class ChemicalPotentialEnergy(EnergyTerm):
    """
    An energy term that purely depends on the number of residues present in a system.
    For some choices of parameters, this is equivalent to a chemical potential contribution to the grand-canonical
    free energy Omega = E - mu * N
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        power: float = 1.0,
        target_size: int = 0,
        chemical_potential: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises Chemical Potential Energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        power: float
            The power to raise the number of residues to.
        target_size: int
            The target size of the system.
        chemical_potential: float
            The chemical potential of the system.
        weight: float
            The weight of the energy term.
        """
        super().__init__(name='chem_pot', inheritable=True, oracle=oracle, weight=weight)
        self.power = power
        self.target_size = target_size
        self.chemical_potential = chemical_potential
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'ChemicalPotentialEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)

        # Count unique combinations of chain_id and res_id
        num_residues = len(set(zip(structure.chain_id, structure.res_id)))
        value = self.chemical_potential * (abs(num_residues - self.target_size)) ** self.power

        return value, value * self.weight


class PLDDTEnergy(EnergyTerm):
    """
    Predicted Local Distance Difference Test energy. This is the spread of the predicted separation between an atom and
    each of its nearest neighbours. This translates to how confident the model is that the sequence has a single lowest
    energy structure, as opposed to a disordered, constantly changing structure. This energy is averaged over the
    relevant atoms.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[Residue] | None,
        inheritable: bool = True,
        weight: float = 1.0,
    ) -> None:
        """Initialises Local Predicted Local Distance Difference Test Energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        residues: list[Residue]
            Which residues to include in the calculation.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        """
        if isinstance(self, OverallPLDDTEnergy):
            name = 'global_pLDDT'
        elif isinstance(self, PLDDTEnergy):
            name = 'local_pLDDT'
        else:
            raise ValueError(f'Unknown energy term type: {type(self)}')
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        if residues is not None:
            self.residue_groups = [residue_list_to_group(residues)]
        else:
            self.residue_groups = []
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'local_plddt' in self.oracle.result_class.model_fields, (
            'PLDDTEnergy requires oracle to return local_plddt in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        folding_result = oracles_result[self.oracle]
        assert hasattr(folding_result, 'local_plddt'), 'local_plddt metric not returned by folding algorithm'
        assert folding_result.local_plddt.shape[0] == 1, 'batch size equal to 1 is required'
        plddt = folding_result.local_plddt[0]  # [n_residues] array
        assert hasattr(folding_result, 'structure'), 'structure not returned by folding algorithm'
        if len(self.residue_groups) != 0:
            mask = self.get_residue_mask(folding_result.structure, residue_group_index=0)
        else:  # if no residues are selected, consider all atoms
            n_residues = sum([c.length for c in folding_result.input_chains])
            mask = np.full(shape=n_residues, fill_value=True)
        value = -np.mean(plddt[mask])
        return value, value * self.weight


class OverallPLDDTEnergy(PLDDTEnergy):
    """
    Overall Predicted Local Distance Difference Test energy.
    """

    def __init__(self, oracle: FoldingOracle, weight: float = 1.0) -> None:
        """Initialises Overall Predicted Local Distance Difference Test Energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        weight: float
            The weight of the energy term.
        """
        super().__init__(oracle=oracle, inheritable=True, weight=weight, residues=None)
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'local_plddt' in self.oracle.result_class.model_fields, (
            'OverallPLDDTEnergy requires oracle to return local_plddt in result_class'
        )
        self.residue_groups = []


class SurfaceAreaEnergy(EnergyTerm):
    """
    Energy term proportional to the amount of exposed surface area. This is measured by dividing the mean SASA
    (Solvent Accessible Surface Area) of the relevant atoms by the maximum possible SASA.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        inheritable: bool = True,
        residues: list[Residue] | None = None,
        probe_radius: float | None = None,
        max_sasa: float | None = None,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises Surface Area Energy Class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. Considers all residues by default.
        probe_radius: float or None, default=None
            The VdW-radius of the solvent molecules used in the SASA calculation. Default is the water VdW-radius.
        max_sasa: float or None, default=None
            The maximum SASA value used if normalization is enabled. Default is the full surface area of a Sulfur atom.
        """
        name = 'surface_area' if residues is None else f'{"selective_" if residues is not None else ""}surface_area'
        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.probe_radius = probe_radius_water if probe_radius is None else probe_radius
        self.max_sasa = max_sasa_values['S'] if max_sasa is None else max_sasa
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'SurfaceAreaEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        if len(self.residue_groups) != 0:
            atom_mask: npt.NDArray[np.bool_] = self.get_atom_mask(structure, residue_group_index=0)
        else:
            atom_mask = np.full(shape=len(structure), fill_value=True)

        sasa_values = sasa(structure, probe_radius=self.probe_radius)
        value = np.mean(sasa_values[atom_mask]) / self.max_sasa
        return value, value * self.weight


class HydrophobicEnergy(EnergyTerm):
    """
    Energy Proportional to the amount of hydrophobic residues present. This is measured by the fraction of selected
    atoms that belong to hydrophobic residues (valine, isoleucine, leucine, phenylalanine, methionine, tryptophan).
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        inheritable: bool = True,
        residues: list[Residue] | None = None,
        surface_only: bool = False,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises hydrophobic energy class.

        Parameters
        ----------
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. If not set, simply considers **all** residues by default.
        surface_only: bool
            Whether to only consider the atoms exposed to water at the surface. If False, interior atoms are included
            in the calculation. If true, result is scaled by normalised solute accessible surface area values.
        weight: float = 1.0
            The weight of the energy term.
        """
        super().__init__(name='hydrophobic', inheritable=inheritable, oracle=oracle, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.surface_only = surface_only
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'HydrophobicEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        if len(self.residue_groups) > 0:
            relevance_mask: npt.NDArray[np.bool_] = self.get_atom_mask(structure, residue_group_index=0)
        else:
            relevance_mask = np.full(shape=len(structure), fill_value=True)

        hydrophobic_mask = np.isin(structure.res_name, hydrophobic_residues)

        value = len(structure[relevance_mask & hydrophobic_mask]) / len(structure[relevance_mask])
        if self.surface_only:
            normalized_sasa = sasa(structure, probe_radius=probe_radius_water) / max_sasa_values['S']
            value *= np.mean(normalized_sasa[relevance_mask & hydrophobic_mask])

        return value, value * self.weight


class PAEEnergy(EnergyTerm):
    """
    Energy that drives down the uncertainty in the predicted distances between two groups of residues. This uncertainty
    is measured by calculating the average normalised predicted alignment error of all the relevant residue pairs.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[list[Residue]],
        inheritable: bool = True,
        cross_term_only: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises the alignment error energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        residues: tuple[list[Residue], list[Residue]]
            Which residues to include in the first and second group.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        cross_term_only: bool, default=True
            Whether to only consider the uncertainty in distance between group 1 and group 2 atoms. If set to False,
            also considers the uncertainty in distances between atoms within the same group.
        weight: float = 1.0
            The weight of the energy term.
        """
        name = f'{"cross_" if cross_term_only else ""}PAE'
        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
        self.cross_term_only = cross_term_only
        if len(residues) == 1:
            self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[0])]
        else:
            self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'pae' in self.oracle.result_class.model_fields, 'PAEEnergy requires oracle to return pae in result_class'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        folding_result = oracles_result[self.oracle]
        structure = oracles_result.get_structure(self.oracle)
        assert hasattr(folding_result, 'pae'), 'pae metric not returned by folding algorithm'
        assert folding_result.pae.shape[0] == 1, 'batch size equal to 1 is required'
        pae = folding_result.pae[0]  # [n_residues, n_residues] pairwise predicted alignment error matrix
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

        value = np.mean(pae[pae_mask]) / max_pae
        return value, value * self.weight


class RingSymmetryEnergy(EnergyTerm):
    """
    Energy that maximises the symmetry of different groups. Symmetry is measured by finding the centroid of the backbone
    of each group and checking how consistently they are spaced from one another.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        symmetry_groups: list[list[Residue]],
        inheritable: bool = True,
        direct_neighbours_only: bool = False,
        weight: float = 1.0,
    ) -> None:
        """Initialises ring symmetry energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        symmetry_groups: list[list[Residue]]
            A list of at least length 2, with each element containing a list of residues corresponding to a symmetry
            group.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        direct_neighbours_only: bool, default=False
            Whether to compare the spacing of each each group to its direct neighbour (compare group i to group i+1
            only), or each group to all other groups. Defaults to the latter.
        weight: float = 1.0
            The weight of the energy term.
        """
        name = f'{"neighbour_" if direct_neighbours_only else ""}ring_symmetry'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        assert (len(symmetry_groups) > 1) and (len(symmetry_groups[0]) >= 1), 'Multiple symmetry groups required.'
        self.residue_groups = [residue_list_to_group(symmetry_group) for symmetry_group in symmetry_groups]
        self.direct_neighbours_only: bool = direct_neighbours_only
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'RingSymmetryEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        num_groups = len(self.residue_groups)
        centroids = np.zeros(shape=(num_groups, 3))
        backbone_mask = np.isin(structure.atom_name, backbone_atoms)

        for i in range(num_groups):
            symmetry_group_mask = self.get_atom_mask(structure, residue_group_index=i)
            centroids[i] = np.mean(structure[symmetry_group_mask & backbone_mask].coord, axis=0)
        if self.direct_neighbours_only:
            neighbour_displacements = centroids - np.roll(centroids, shift=1, axis=0)
            neighbour_distances = np.linalg.norm(neighbour_displacements, axis=1)
            value = np.std(neighbour_distances)
        else:
            displacement_matrix = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            distance_matrix = np.linalg.norm(displacement_matrix, axis=2)
            unique_distances = distance_matrix[~np.tri(N=num_groups, dtype=bool)]
            value = np.std(unique_distances)

        return value, value * self.weight


class SeparationEnergy(EnergyTerm):
    """
    Energy that minimizes the distance between two groups of residues. The position of each group is
    defined as the centroid of the backbone atoms of the residues belonging of that group.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: tuple[list[Residue], list[Residue]],
        normalize: bool = True,
        inheritable: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises separation energy class.

        Parameters
        ----------
        residues: tuple[list[Residue],list[Residue]]
            A tuple containing two lists of residues, those to include in the first [0] and second [1] group.
        normalize: bool, default=True
            Whether the distance calculated is divided by the number of atoms in both groups.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        """
        name = f'{"normalized_" if normalize else ""}separation'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        self.normalize = normalize
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'SeparationEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
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

        value = float(distance)
        return value, value * self.weight


class GlobularEnergy(EnergyTerm):
    """
    Energy proportional to the moment of inertia of the structure around its centroid. This energy is minimized when
    the atoms belonging to a structure have the lowest possible distance from the centre, and, due to excluded volume
    effects that prevent collapse to a single point, helps forcing structures to
    be as close as possible to a spherically distributed cloud of points.
    """

    def __init__(
        self,
        oracle: Oracle,
        residues: list[Residue] | None = None,
        inheritable: bool = True,
        normalize: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises globular energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. Considers all residues by default.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        normalize: bool, default=True
            Whether the mean centroid distance calculated is divided by the number of atoms considered.
        weight: float = 1.0
            The weight of the energy term.
        """
        name = f'{"normalized_" if normalize else ""}globular'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.normalize = normalize
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'GlobularEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
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

        value = np.std(centroid_distances)
        return value, value * self.weight


class TemplateMatchEnergy(EnergyTerm):
    """
    Energy that drives the structure to match an input-provided template. The difference with the template is always calculated
    by automatically considering the rotation and translation that best maximize the overlap with the template.
    """

    def __init__(
        self,
        oracle: Oracle,
        template_atoms: AtomArray,
        residues: list[Residue],
        backbone_only: bool = False,
        distogram_separation: bool = False,
        weight: float = 1.0,
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
        name = f'{"backbone_" if backbone_only else ""}template_match'
        super().__init__(name=name, oracle=oracle, inheritable=False, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)]
        self.template_atoms = template_atoms
        self.backbone_only = backbone_only
        self.distogram_separation = distogram_separation
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'TemplateMatchEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
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

        value = separation
        return value, value * self.weight


class SecondaryStructureEnergy(EnergyTerm):
    """
    Energy that drives the secondary structure of the selected residues to a given type. Calculated by finding the
    fraction of selected residues with the wrong secondary structure. Secondary structure types include alpha-helix,
    beta-sheet, and coil.
    """

    def __init__(
        self,
        oracle: Oracle,
        residues: list[Residue],
        target_secondary_structure: str,
        inheritable: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Initialises the secondary structure energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        residues: list[Residue]
            Which residues to include in the calculation.
        target_secondary_structure: str
            Which secondary structure type to drive towards. Options are 'alpha-helix', 'beta-sheet', or 'coil'.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        """
        name = f'{target_secondary_structure.lower()}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)]
        options = ('alpha-helix', 'beta-sheet', 'coil')
        assert target_secondary_structure in options, f'{target_secondary_structure} not recognised. options: {options}'
        self.target_secondary_structure = target_secondary_structure
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'SecondaryStructureEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        target_label = self.target_secondary_structure[0]  # How Biotite labels secondary structures
        calculated_labels = annotate_sse(structure)
        selection_mask = self.get_residue_mask(structure, residue_group_index=0)

        value = np.mean(calculated_labels[selection_mask] != target_label)
        return value, value * self.weight


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
        oracle: Oracle,
        aspect_ratio: tuple[float, float, float],
        k_attractive: float = 1.0,
        k_repulsive: float = 10.0,
        weight: float = 1.0,
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
        name = 'ellipsoid'
        super().__init__(name=name, oracle=oracle, inheritable=True, weight=weight)
        self.residue_groups = []
        self.k_attractive = k_attractive
        self.k_repulsive = k_repulsive

        # largest aspect ratio first (corresponds to the first principle axis that captures the most positional variance)
        self.aspect_ratio = np.sort(aspect_ratio)[::-1]
        self.aspect_ratio /= np.prod(self.aspect_ratio) ** (1 / 3)  # ensures product of aspect ratios equals 1
        warnings.warn('This energy is yet to have any unit tests and is not guaranteed to work')

        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'EllipsoidEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        atoms = structure[np.isin(structure.atom_name, backbone_atoms)]

        # transforming to principle component axes
        centered_coords = atoms.coord - np.mean(atoms.coord, axis=0, keepdims=True)
        variances, basis_vectors = np.linalg.eigh(np.cov(centered_coords, rowvar=False))
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

        value = attractive_energy + repulsive_energy
        return value, value * self.weight


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
        oracle: Oracle,
        aspect_ratio: tuple[float, float, float],
        k_attractive: float = 1.0,
        k_repulsive: float = 10.0,
        sharpness: float = 2.0,
        weight: float = 1.0,
    ) -> None:
        """
        Initializes cuboid energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
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
        weight: float = 1.0
            The weight of the energy term.
        """
        name = 'cuboid'
        super().__init__(name=name, oracle=oracle, inheritable=True, weight=weight)
        self.residue_groups = []
        self.k_attractive = k_attractive
        self.k_repulsive = k_repulsive
        self.sharpness = sharpness

        # largest aspect ratio first (as first principle axis captures the most positional variance)
        self.aspect_ratio = np.sort(aspect_ratio)[::-1]
        self.aspect_ratio /= np.prod(self.aspect_ratio) ** (1 / 3)  # ensures product of aspect ratios equals 1
        warnings.warn('This energy is yet to have any unit tests and is not guaranteed to work')

        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'CuboidEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
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

        value = attractive_energy + repulsive_energy
        return value, value * self.weight


class EmbeddingsSimilarityEnergy(EnergyTerm):
    """
    Energy terms measuring the cosine similarity between current embeddings and embeddings of a template.
    See paper: Rajendran et al. 2025 - to be published
    """

    def __init__(
        self,
        oracle: EmbeddingOracle,
        residues: list[Residue],
        reference_embeddings: npt.NDArray[np.float64],
        weight: float = 1.0,
    ) -> None:
        """
        Initialises EmbeddingsSimilarityEnergy class.

        Parameters
        ----------
        oracle: EmbeddingOracle
            The oracle that will be used to calculate the embeddings.
        residues: list[Residue]
            Which residues to include in the calculation.
        reference_embeddings: np.ndarray
            The reference embeddings to compare to.
        inheritable: bool, default=False
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        """
        name = 'embeddings_similarity'
        super().__init__(name=name, oracle=oracle, inheritable=False, weight=weight)
        # with the current implementation, the energy term is not inheritable, as reference embeddings would change
        # and would need to be changed dynamically, which is not fully supported yet
        self.residue_groups = [residue_list_to_group(residues)]
        self.reference_embeddings = reference_embeddings
        assert self.reference_embeddings.shape[0] == len(self.residue_groups[0]), (
            f'Number of reference embeddings ({self.reference_embeddings.shape[0]}) does not'
            f'match number of residues to include in energy term ({len(self.residue_groups[0])})'
        )

        assert isinstance(self.oracle, EmbeddingOracle), 'Oracle must be an instance of EmbeddingOracle'
        assert 'embeddings' in self.oracle.result_class.model_fields, (
            'EmbeddingsSimilarityEnergy requires oracle to return embeddings in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        embeddings = oracles_result.get_embeddings(self.oracle)
        chains = oracles_result[self.oracle].input_chains
        assert isinstance(embeddings, np.ndarray), (
            f'Embeddings is expected to be a numpy array, not type: {type(embeddings)}'
        )
        assert len(embeddings.shape) == 2, (
            f'Embeddings is expected to be a 2D tensor, not shape: {embeddings.shape}. This does not work with batches.'
        )

        # The following generate a 2D numpy array of shape (n_conserved_residues, n_features)
        # where n_conserved_residues is the number of residues in the reference embeddings
        # and n_features is the number of features in the embeddings.
        # Note that n_conserved_residues must be equal to len(self.residue_groups[0])
        conserved_embeddings = embeddings[self.conserved_index_list(chains)]

        assert conserved_embeddings.shape == self.reference_embeddings.shape, (
            f'Conserved embeddings shape {conserved_embeddings.shape} does not match reference embeddings {self.reference_embeddings.shape}'
        )
        # The following generates a 1D tensor of shape (n_conserved_residues)
        cosine = np.sum(conserved_embeddings * self.reference_embeddings, axis=1)
        similarity = np.mean(cosine)

        value = 1.0 - similarity
        return value, value * self.weight

    def conserved_index_list(self, chains: list[Chain]) -> list[int]:
        """Returns the indices of the conserved residues (stored in .residue_group[0]) in the pLM embedding array."""
        # TODO: This is a completely untested function and needs to be tested!!!!
        # TODO: Also, it might not be fully necessary, but need to think about it
        # This is only relevant with a multimer, which is yet to be implemented, let's do other things first
        conserved_chain_id, conserved_res_id = self.residue_groups[0]
        global_index_list = []
        # Chains already has chains in the correct order
        offset = 0
        for i, chain in enumerate(chains):
            chain_id = chain.chain_ID
            for j, residue in enumerate(chain.residues):
                residue_global_index = offset + j
                # Check if the residue is in the conserved residues
                for k in range(len(conserved_chain_id)):
                    if chain_id == conserved_chain_id[k] and residue.index == conserved_res_id[k]:
                        global_index_list.append(residue_global_index)
            offset += len(chain.residues)
        return global_index_list
