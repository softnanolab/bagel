"""
Standard template and objects for calculating structural or propery losses.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from abc import ABC, abstractmethod
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Literal, Callable
from biotite.structure import AtomArray, sasa, annotate_sse, superimpose

from .constants import hydrophobic_residues, max_sasa_values, probe_radius_water, backbone_atoms
from .chain import Residue, Chain
from .oracles import Oracle, OracleResult, OraclesResultDict
from .oracles.folding import FoldingResult, FoldingOracle
from .oracles.embedding import EmbeddingResult, EmbeddingOracle
from .oracles.folding.utils import reorder_atoms_in_template


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

    EnergyTerms can be inheritable, which is only relevant for :class:`~bagel.mutation.GrandCanonical`.
    In that type of simulation when adding a new residues, the "inheritable" attribute decides whether or not
    the new residue will be added to the residues for which this term is calculated. In general, a new residue
    inherits all energy terms of one of its neighbours (chosen randomly to be the left or right neighbour),
    if these terms are inheritable.
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
        """Checks required attributes have been set after class is initialised"""
        assert hasattr(self, 'name'), 'name attribute must be set in class initialiser'
        assert hasattr(self, 'residue_groups'), 'residue_groups attribute must be set in class initialiser'
        assert hasattr(self, 'inheritable'), 'inheritable attribute must be set in class initialiser'
        assert hasattr(self, 'weight'), 'weight attribute must be set in class initialiser'
        if 'template_match' in self.name:
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
        for chain in np.unique(chain_ids):
            chain_mask = structure.chain_id == chain  # gets all atoms from a given chain

            # Note: in an atom_array object like structure .res_id is what we call the residue index and is an integer
            chain_res_ids = structure[chain_mask].res_id  # gets all residues indices from a given chain
            chain_res_ids_in_group = res_indices[chain_ids == chain]  # gets all residue indices in the residue group

            # for that specific chain, check if the residue indices are in the residue group
            atom_mask[chain_mask] = np.isin(chain_res_ids, chain_res_ids_in_group)
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
        name: str | None = None,
    ) -> None:
        """
        Initialises Predicted Template Modelling Score Energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        weight: float
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'pTM'
        else:
            name = f'pTM_{name}'
        super().__init__(name=name, inheritable=True, oracle=oracle, weight=weight)
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'ptm' in self.oracle.result_class.model_fields, 'PTMEnergy requires oracle to return ptm in result_class'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        folding_result = oracles_result[self.oracle]
        assert hasattr(folding_result, 'ptm'), 'PTM metric not returned by folding algorithm'
        value = -folding_result.ptm
        return value, value * self.weight


class ChemicalPotentialEnergy(EnergyTerm):
    r"""
    An energy term that purely depends on the number of residues present in a system.
    Note for statistical mechanics: for some choice of parameters, adding this term is equivalent to making a simulation
    in the grand-canonical ensemble, where the free-energy that is minimized is:

    .. math::

        \Omega = E - \mu N

    where :math:`\Omega` is the grand potential, :math:`E` is the energy, :math:`\mu` is the chemical potential,
    and :math:`N` is the number of residues.
    """

    def __init__(
        self,
        oracle: Oracle,
        power: float = 1.0,
        target_size: int = 0,
        chemical_potential: float = 1.0,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises Chemical Potential Energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        power: float
            The power to raise the number of residues to.
        target_size: int
            The target size of the system.
        chemical_potential: float
            The chemical potential of the system.
        weight: float
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'chem_pot'
        else:
            name = f'chem_pot_{name}'
        super().__init__(name=name, inheritable=True, oracle=oracle, weight=weight)
        self.power = power
        self.target_size = target_size
        self.chemical_potential = chemical_potential
        assert isinstance(self.oracle, Oracle), 'Input to oracle not an Oracle object'
        assert 'input_chains' in self.oracle.result_class.model_fields, (
            'ChemicalPotentialEnergy requires oracle to return input_chains in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        input_chains = oracles_result.get_input_chains(self.oracle)  # get the input chains from the oracle result

        # Count all residues in all input chains
        num_residues = sum(chain.length for chain in input_chains)
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
        name: str | None = None,
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
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if isinstance(self, OverallPLDDTEnergy):
            base_name = 'global_pLDDT'
        elif isinstance(self, PLDDTEnergy):
            base_name = 'local_pLDDT'
        else:
            raise ValueError(f'Unknown energy term type: {type(self)}')

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

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

    def __init__(self, oracle: FoldingOracle, weight: float = 1.0, name: str | None = None) -> None:
        """Initialises Overall Predicted Local Distance Difference Test Energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        weight: float
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        super().__init__(oracle=oracle, inheritable=True, weight=weight, residues=None, name=name)
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
        name: str | None = None,
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
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = (
            'surface_area' if residues is None else f'{"selective_" if residues is not None else ""}surface_area'
        )

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

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
        mode: Literal['surface', 'core', 'all'] = 'all',
        surface_only: bool = False,
        core_only: bool = False,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises hydrophobic energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        residues: list[Residue] or None, default=None
            Which residues to include in the calculation. If not set, simply considers **all** residues by default.
        mode: Literal['surface', 'core', 'all'] = 'all'
            Selection of which atoms contribute to the hydrophobicity score:
            - 'surface': counts hydrophobic residues at the surface, weighted by normalised SASA
            - 'core': counts hydrophobic residues in the core, weighted by 1 - normalised SASA
            - 'all': counts all hydrophobic residues, no SASA weighting
            Normalisation uses `max_sasa_values['S']` and the probe radius `probe_radius_water`.
        surface_only: bool, default=False
            Deprecated. Use `mode='surface'` instead.
        core_only: bool, default=False
            Deprecated. Use `mode='core'` instead.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'hydrophobic'
        else:
            name = f'hydrophobic_{name}'

        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []

        # Backwards compatibility for deprecated flags
        self.surface_only = surface_only
        self.core_only = core_only
        if surface_only and core_only:
            raise ValueError('Only one of surface_only or core_only can be True at the same time.')
        if surface_only or core_only:
            warnings.warn(
                "Parameters 'surface_only' and 'core_only' are deprecated and will be removed in v0.2.0. "
                "Use 'mode' instead (e.g., mode='surface' or mode='core').",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = 'surface' if surface_only else 'core'
        self.mode: Literal['surface', 'core', 'all'] = mode
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

        if self.mode == 'surface':
            normalized_sasa = sasa(structure, probe_radius=probe_radius_water) / max_sasa_values['S']
            value *= np.mean(normalized_sasa[relevance_mask & hydrophobic_mask])
        elif self.mode == 'core':
            normalized_sasa = 1.0 - sasa(structure, probe_radius=probe_radius_water) / max_sasa_values['S']
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
        name: str | None = None,
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
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = f'{"cross_" if cross_term_only else ""}PAE'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

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


class LISEnergy(EnergyTerm):
    """
    Energy representing the Local Interaction Score [], a function of the PAE matrix.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[list[Residue]],
        pae_cutoff: float = 12.0,
        intensive: bool = True,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises the alignment error energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        residues: tuple[list[Residue], list[Residue]]
            Which residues to include in the first and second group.
        pae_cutoff: float = 12.0
            The cutoff value for the PAE, in Angstroms, below which the interaction is considered "local".
        intensive: bool, default=True
            If True, the LIS is averaged over the number of residue pairs, otherwise it's an extensive sum.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = f'LIS'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

        self.pae_cutoff = pae_cutoff
        self.intensive = intensive  # if True, LIS is an average otherwise scales with number of residue pairs bonded

        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
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

        group_1_mask = self.get_residue_mask(structure, residue_group_index=0)
        group_2_mask = self.get_residue_mask(structure, residue_group_index=1)
        pae_mask = np.full(shape=pae.shape, fill_value=False)

        pae_mask[group_1_mask[:, np.newaxis] & group_2_mask[np.newaxis, :]] = True
        # in case PAE symmetry is not enforced
        pae_mask[group_2_mask[:, np.newaxis] & group_1_mask[np.newaxis, :]] = True

        diagonal_mask = np.eye(len(pae), dtype=bool)
        pae_mask[diagonal_mask] = False  # should ignore uncertainty in distance between atom and itself
        selected_pae = pae[pae_mask]

        # selected_pae only contains the correct pairs now, use it to calculate the LIS score.

        # Step 1: take only values where pae < pae_cutoff
        cutoff = self.pae_cutoff
        threshold_mask = selected_pae < cutoff
        selected_pae = selected_pae[threshold_mask]

        if len(selected_pae) == 0:
            value = 0.0
        else:
            # Step 2: For those values that remain, the LIS score is given by:
            lis_scores = (cutoff - selected_pae) / cutoff
            if self.intensive:
                value = -np.mean(lis_scores)  # Negative because you want to be interpreted as an energy
            else:
                # 0.5 is to avoid double-counting of LIS pairs, which you would if PAE(ij) is asymmetric due
                # to masking above
                value = -0.5 * np.sum(lis_scores)  # Negative because you want to be interpreted as an energy

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
        name: str | None = None,
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
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = f'{"neighbour_" if direct_neighbours_only else ""}ring_symmetry'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

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
        function: Callable[[float], float] | None = None,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises separation energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        residues: tuple[list[Residue],list[Residue]]
            A tuple containing two lists of residues, those to include in the first [0] and second [1] group.
        function: Callable[[float], float] | None
            Optional callable f(x) applied to the centroid distance x (in Å) before weighting.
            If None, the identity function is used (i.e., energy equals the distance).
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'separation'
        else:
            name = f'separation_{name}'

        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        self.function: Callable[[float], float] | None = function
        if self.function is not None:
            assert callable(self.function), 'Function must be callable and accept a single float argument'
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

        value = float(distance)
        if self.function is not None:
            value = float(self.function(float(distance)))

        return value, value * self.weight


class FlexEvoBindEnergy(EnergyTerm):
    """
    Energy that minimizes the 'average minimum distance' between two groups of residues.
    In practice, for each residue in the first group, it finds the closest residue in the second group and
    calculates the minimum distance between them. The minimum is over all possible pairs of atoms that
    make up the two residues. The average is over all the residues in the first group.
    Symmetrise this operation by doing the same but looking at residues from group 2 and
    what is their minimum distance when looking at residues to group one.

    This energy is a symmetrised version of the minimum separation component of the loss used to design peptide binders in:
    'Li, Q., Vlachos, E.N. & Bryant, P. Design of linear and cyclic peptide binders from protein sequence information. Commun Chem 8, 211 (2025)'
    DOI https://doi.org/10.1038/s42004-025-01601-3

    Note in this reference the explanation of Eq.1 is misleading. Here, the average of the minimum distance
    is over all the the residues in the first group.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: tuple[list[Residue], list[Residue]],
        plddt_weighted: bool = False,
        symmetrized: bool = True,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises separation energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        residues: tuple[list[Residue],list[Residue]]
            A tuple containing two lists of residues, those to include in the first [0] and second [1] group.
        plddt_weighted: bool
            A bool indicating whether the result need to be weighted by the plddt of the residues considered.
            If True, this definition is closer to the EvoBind energy in the reference below
        symmetrized: bool
            A bool indicating whether or not the calculation of the minimum distances need to be symmetrized between residues in
            residued[0] and those in residues[1]. Otherwise the minimum distances are those between any atom in residues from
            residues[0] and those in residues in residues[1], but not vice versa.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'flex_evo'
        else:
            name = f'flex_evo_{name}'

        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residues = residues
        self.symmetrized = symmetrized
        self.plddt_weighted = plddt_weighted
        self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'FlexEvoBindEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)

        if self.symmetrized:
            indices = [0, 1]
        else:
            indices = [0]

        values_list = []
        counts_list = []

        for main in indices:
            # Get the mask for all the atoms belonging to any residue in group 2
            partner = 1 if main == 0 else 0
            partner_mask = self.get_atom_mask(structure, residue_group_index=partner)
            partner_atoms = structure[partner_mask]
            if len(partner_atoms) == 0:
                # Nothing to compare against for this direction
                continue

            # Get the chain_ids and res_ids for the residues in the first group
            chain_ids, res_ids = self.residue_groups[main]

            min_distances = []  # List to store the minimum distances for each residue in the main group

            # Now iterate over each residue in the first group
            for chain_id, res_id in zip(chain_ids, res_ids):
                # Extract from the structure the atoms corresponding to the residues with current chain_id and res_id
                curr_residue_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)

                # Get the atoms corresponding to the current residue
                curr_residue_atoms = structure[curr_residue_mask]
                if len(curr_residue_atoms) == 0:
                    continue
                # Vectorized min distance between atoms of current residue and all partner atoms
                diff = partner_atoms.coord[np.newaxis, :, :] - curr_residue_atoms.coord[:, np.newaxis, :]
                dist_mat = np.linalg.norm(diff, axis=2)
                min_dist = float(np.min(dist_mat))
                min_distances.append(min_dist)  # Store the minimum distance for this residue

            # Calculate the average of these minimum distances
            if len(min_distances) == 0:
                # No valid atoms for any residue in this direction; skip contribution
                continue
            average_min_distance = float(np.mean(min_distances))
            value = average_min_distance

            # If plddt_weighted is True, divide by the average pLDDT of the residues in the group
            # In this case, this energy term is the EvoBind loss function mentioned above, but symmetrized.
            # in the sense that what is the binder and what is the hotspot does not matter.
            if self.plddt_weighted:
                folding_result = oracles_result[self.oracle]
                assert hasattr(folding_result, 'local_plddt'), 'local_plddt metric not returned by folding algorithm'
                assert folding_result.local_plddt.shape[0] == 1, 'batch size equal to 1 is required'
                plddt = folding_result.local_plddt[0]
                assert hasattr(folding_result, 'structure'), 'structure not returned by folding algorithm'
                main_mask = self.get_residue_mask(structure, residue_group_index=main)

                mask_count = int(np.count_nonzero(main_mask))
                if mask_count > 0:
                    average_plddt = float(np.mean(plddt[main_mask]))
                    denom = average_plddt if average_plddt > 0.0 else np.finfo(float).eps
                    value /= denom

            # Scale value by the number of residues in the group, so that eventually you can
            # calculate a weighted average between residues in the binder and hotspot
            valid_count = len(min_distances)
            value *= valid_count

            # save calculated value
            values_list.append(value)
            counts_list.append(valid_count)

        # calculate the (weighted) average over saved values
        total_count = sum(counts_list)
        value = float(np.sum(values_list) / total_count) if total_count > 0 else 0.0

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
        weight: float = 1.0,
        name: str | None = None,
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
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'globular'
        else:
            name = f'globular_{name}'

        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
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
        name: str | None = None,
    ) -> None:
        """
        Initialises template match energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
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
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = f'{"backbone_" if backbone_only else ""}template_match'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

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
        template_atoms = reorder_atoms_in_template(self.template_atoms)
        if self.backbone_only:
            structure_atoms = structure_atoms[np.isin(structure_atoms.atom_name, backbone_atoms)]
            template_atoms = template_atoms[np.isin(template_atoms.atom_name, backbone_atoms)]
        if len(structure_atoms) != len(template_atoms):
            raise ValueError(
                'Different number of atoms in template and given residues: '
                f'template_atoms={len(template_atoms)}, structure_atoms={len(structure_atoms)}, '
                f'template={template_atoms}, structure={structure_atoms}'
            )
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
        name: str | None = None,
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
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = f'{target_secondary_structure.lower()}'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

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
        name: str | None = None,
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
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'embeddings_similarity'
        else:
            name = f'embeddings_similarity_{name}'

        super().__init__(name=name, oracle=oracle, inheritable=False, weight=weight)
        # with the current implementation, the energy term is not inheritable, as reference embeddings would change
        # and would need to be changed dynamically, which is not fully supported yet
        self.residue_groups = [residue_list_to_group(residues)]
        # Normalise the reference embeddings to unit length
        reference_embeddings = reference_embeddings / np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        self.reference_embeddings = reference_embeddings
        assert self.reference_embeddings.shape[0] == len(self.residue_groups[0][0]), (
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
        # Normalise the conserved embeddings to unit length
        conserved_embeddings = conserved_embeddings / np.linalg.norm(conserved_embeddings, axis=1, keepdims=True)

        # The following generates a 1D tensor of shape (n_conserved_residues)
        cosine = np.sum(conserved_embeddings * self.reference_embeddings, axis=1)
        similarity = np.mean(cosine)

        value = 1.0 - similarity
        return value, value * self.weight

    def conserved_index_list(self, chains: list[Chain]) -> list[int]:
        """Returns the indices of the conserved residues (stored in .residue_group[0]) in the pLM embedding array."""
        conserved_chain_id, conserved_res_id = self.residue_groups[0]
        global_index_list = []

        # Create a mapping of (chain_id, res_index) to global index
        offset = 0
        chain_res_to_global = {}
        for chain in chains:
            for j, residue in enumerate(chain.residues):
                chain_res_to_global[(chain.chain_ID, residue.index)] = offset + j
            offset += len(chain.residues)

        # Process residues in the order they appear in conserved_chain_id and conserved_res_id
        for chain_id, res_id in zip(conserved_chain_id, conserved_res_id):
            global_index_list.append(chain_res_to_global[(str(chain_id), int(res_id))])

        return global_index_list
