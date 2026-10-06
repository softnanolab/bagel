"""
Standard template and objects for calculating structural or property losses.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Literal, Callable, Any, TYPE_CHECKING
from biotite.structure import AtomArray, CellList, sasa, annotate_sse, superimpose
from .constants import (
    hydrophobic_residues,
    max_sasa_values,
    probe_radius_water,
    backbone_atoms,
    hydropathy_index,
    max_theoretical_sasa_for_residues,
    max_residue_sasa,
    vdw_radii,
    default_vdw_radius,
    salt_bridge_residues,
    aa_dict,
)
from .chain import Residue, Chain
from .oracles import Oracle, OracleResult, OraclesResultDict
from .oracles.folding import FoldingResult, FoldingOracle
from .oracles.embedding import EmbeddingResult, EmbeddingOracle
from .oracles.folding.utils import reorder_atoms_in_template

if TYPE_CHECKING:  # State imports this module, so it can only be imported for type checking
    from .state import State


# first row is chain_ids and second row is corresponding residue indices.
ResidueGroup = tuple[npt.NDArray[np.str_], npt.NDArray[np.int_]]


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

    def get_embedding_residue_mask(
        self,
        input_chains: list[Chain],
        residue_group_index: int,
        chain_index: npt.NDArray[np.int_] | None = None,
        residue_index: npt.NDArray[np.int_] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Boolean mask over per-residue embedding rows for a stored residue group.

        Per-residue embeddings follow the input order: chains in ``input_chains``
        order, residues in each chain's ``0..len-1`` order. Each row's identity
        ``(chain_ID, index)`` is reconstructed from ``input_chains`` and matched
        against the term's residue group.

        If ``chain_index`` / ``residue_index`` (as reported by the oracle) are
        provided, they are used as a redundant cross-check: ``chain_index`` is the
        0-based chain ordinal mapped back to its ``chain_ID`` via ``input_chains``,
        and both must agree with the reconstruction — otherwise a ``ValueError`` is
        raised, catching any row/residue misalignment.
        """
        chain_ids_in_order = [chain.chain_ID for chain in input_chains]
        row_chain_ids = np.array([res.chain_ID for chain in input_chains for res in chain.residues])
        row_res_indices = np.array([res.index for chain in input_chains for res in chain.residues], dtype=int)

        if chain_index is not None and residue_index is not None:
            reported_chain = np.asarray(chain_index, dtype=int)
            reported_res = np.asarray(residue_index, dtype=int)
            n_rows = row_chain_ids.shape[0]
            if reported_chain.shape[0] != n_rows or reported_res.shape[0] != n_rows:
                raise ValueError(
                    'Oracle chain/residue index length does not match the number of residues in input_chains '
                    f'({reported_chain.shape[0]} / {reported_res.shape[0]} vs {n_rows}).'
                )
            if np.any(reported_chain < 0) or np.any(reported_chain >= len(chain_ids_in_order)):
                raise ValueError('Oracle chain_index refers to a chain ordinal outside input_chains.')
            reported_chain_ids = np.array([chain_ids_in_order[c] for c in reported_chain])
            if not np.array_equal(reported_chain_ids, row_chain_ids) or not np.array_equal(
                reported_res, row_res_indices
            ):
                raise ValueError(
                    'Oracle-reported chain/residue indices disagree with the input_chains ordering; '
                    'the embedding rows may be misaligned with the residues.'
                )

        chain_ids, res_indices = self.residue_groups[residue_group_index]
        mask = np.zeros(row_chain_ids.shape[0], dtype=bool)
        for cid in np.unique(chain_ids):
            wanted = res_indices[chain_ids == cid]
            mask |= (row_chain_ids == cid) & np.isin(row_res_indices, wanted)
        return mask


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

    The energy of this term is

    .. math::

        \mu \, d^{p}, \qquad d = \max(N_{\min} - N, \, 0, \, N - N_{\max})

    where :math:`p` is ``power`` and :math:`d` is the distance from :math:`N` to the allowed range of sizes
    :math:`[N_{\min}, N_{\max}]`, given by ``target_size``. The energy is therefore zero anywhere inside the range,
    and outside it grows with the deviation from the closest boundary. If ``target_size`` is a single number the range
    is that one size, and :math:`d = |N - N_{\mathrm{target}}|`. For the term to restrain the size, ``chemical_potential``
    must be positive, otherwise it rewards moving away from the range.
    """

    def __init__(
        self,
        oracle: Oracle,
        power: float = 1.0,
        target_size: int | tuple[int, int] = 0,
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
            The exponent to which the distance from the allowed range of sizes is raised. Must be positive.
        target_size: int | tuple[int, int]
            The target size of the system, in number of residues. Either a single size, or a range
            (min_size, max_size), both included, within which the energy is zero.
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
        if isinstance(target_size, (tuple, list)):
            assert len(target_size) == 2, 'a target_size range must be given as (min_size, max_size)'
            self.min_size, self.max_size = target_size
        else:
            self.min_size = self.max_size = target_size
        assert self.min_size <= self.max_size, 'min_size of target_size cannot be larger than max_size'
        assert power > 0, 'power must be positive'
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
        # distance from the allowed range of sizes: zero inside it, and from the closest boundary outside it
        distance = max(self.min_size - num_residues, 0, num_residues - self.max_size)
        value = self.chemical_potential * distance**self.power

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


class HydropathyEnergy(EnergyTerm):
    """
    Energy based on the Kyte-Doolittle hydropathy values. If the selected mode is 'all', then this is calculated by taking the
    GRAVY score (arithmetic mean of the Kyte-Doolittle hydropathy values) of the relevant residues as a measure of the
    hydrophobicity of the structure. If the selected mode is 'core' or 'surface', then the weighted mean of Kyte-Doolittle hydropathy
    values (also called hydropathy indices) of the residues is taken as a measure of the hydrophobicity of the structure,
    where the weights are the normalised SASA of the residues. This is done to retain the residue-level correlation between hydropathy and exposure.

    The hydropathy index is a measure of the hydrophobicity of an amino acid, with higher values indicating more
    hydrophobic amino acids. This energy term encourages the presence of hydrophobic residues in the structure,
    which can be important for protein folding and stability.

    Ref: https://williams.chemistry.gatech.edu/course_Information/6572/papers/kytoe_hydrophobicity_1982.pdf

    Note that this energy term is different from the `HydrophobicEnergy` term, which simply counts the fraction of selected
    atoms that belong to hydrophobic residues (valine, isoleucine, leucine, phenylalanine, methionine, tryptophan).
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        inheritable: bool = True,
        residues: list[Residue] | None = None,
        mode: Literal['surface', 'core', 'all'] = 'all',
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises hydropathy energy class based on Kyte-Doolittle hydropathy values.

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
            - 'surface': calculates weighted mean of hydropathic indices for the residues at the surface, weighted by normalised SASA
            - 'core': calculates weighted mean of hydropathic indices for the residues in the core, weighted by 1 - normalised SASA
            - 'all': calculates GRAVY score (arithmetic mean of hydropathic indices) for all hydropathic residues, no SASA weighting
            Normalisation uses 'max_theoretical_sasa_for_residues' specific to each residue and the probe radius
            `probe_radius_water`.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'hydropathic'
        else:
            name = f'hydropathic_{name}'

        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.mode: Literal['surface', 'core', 'all'] = mode
        if self.mode not in ['surface', 'core', 'all']:
            raise ValueError(f"Invalid mode: {mode}. Expected one of: 'surface', 'core', 'all'.")
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'HydropathyEnergy requires oracle to return structure in result_class'
        )

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        # TODO: optimize this function, as it can be quite slow for large structures due to the for loop over residues.
        # - By vectorizing the calculation of residue hydropathy index and SASA values, rather than using a for loop over residues

        structure = oracles_result.get_structure(self.oracle)

        # Handling empty structure early
        if len(structure) == 0:
            return 0.0, 0.0

        # Handling unknown residues early by checking if any residues in the structure are not
        # in the hydropathy index, and if so, warning and returning 0 energy
        # (as we cannot calculate hydropathy index or SASA for unknown residues)

        res_names = structure.res_name
        # Check: is each residue in hydropathy index (i.e., is it a known amino acid)
        valid_mask = np.isin(res_names, list(hydropathy_index.keys()))
        unknown_mask = ~valid_mask

        # Warn if unknown residues exist
        if np.any(unknown_mask):
            unknown_residues = np.unique(res_names[unknown_mask])
            unknown_residue_names = tuple(sorted(map(str, unknown_residues)))
            warnings.warn(
                f'Unknown residues encountered: {unknown_residue_names} (count={len(unknown_residues)}). '
                'These residues will be removed from the structure before calculating the energy. '
                'This may affect the energy calculation if any of the user-specified residues were unknown.',
                UserWarning,
            )

        # Filter structure BEFORE any calculations
        structure = structure[valid_mask]

        # Get residue-level information: chain_ids and res_ids for the relevant residues.
        if len(self.residue_groups) > 0:
            # Residues are selected, so only consider those residues for the energy calculation
            # Validate that all user-specified residues exist in the structure
            chain_ids_orig, res_ids_orig = self.residue_groups[0]
            valid_residues_mask = np.zeros(len(chain_ids_orig), dtype=bool)
            for i, (chain_id, res_id) in enumerate(zip(chain_ids_orig, res_ids_orig)):
                atom_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
                valid_residues_mask[i] = np.any(atom_mask)

            # Warn if any user-specified residues were not found
            if not np.all(valid_residues_mask):
                removed_residues = list(zip(chain_ids_orig[~valid_residues_mask], res_ids_orig[~valid_residues_mask]))
                warnings.warn(
                    f'User-specified residues not found in structure: {removed_residues}. '
                    'These residues will be skipped in the energy calculation.',
                    UserWarning,
                )
            chain_ids = chain_ids_orig[valid_residues_mask]
            res_ids = res_ids_orig[valid_residues_mask]

        else:
            # All residues are relevant if no specific group is selected
            # Use structured arrays to get unique chain_id and res_id pairs while preserving res_id dtype.
            residue_ids = np.empty(
                len(structure),
                dtype=[('chain_id', structure.chain_id.dtype), ('res_id', structure.res_id.dtype)],
            )
            residue_ids['chain_id'] = structure.chain_id
            residue_ids['res_id'] = structure.res_id
            unique_residue_indices = np.unique(residue_ids, return_index=True)[1]
            unique_residue_indices = np.sort(unique_residue_indices)  # preserve order as they appear in structure

            chain_ids = residue_ids['chain_id'][unique_residue_indices]
            res_ids = residue_ids['res_id'][unique_residue_indices]

            # Ensure that the chain_ids and res_ids arrays are aligned and 1D
            assert len(chain_ids) == len(res_ids), 'ResidueGroup arrays must be aligned'

        # Compute atom-level SASA values once on the filtered structure, as they will be reused for each residue
        # Unknown residues have already been removed above, before this SASA calculation
        atom_sasa_values = sasa(structure, probe_radius=probe_radius_water)

        # Initialize containers for hydropathy indices and residue SASA values
        residue_hydropathy_indices = []
        normalized_residue_sasa_values = []

        # Iterate over each residue in the group (or all residues if no group specified) and calculate its hydropathic value and mean SASA
        for chain_id, res_id in zip(chain_ids, res_ids):
            # find the indices of atoms that belong to this res_id and chain_id
            atom_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
            atom_indices = np.where(atom_mask)[0]

            # Skip if residue not found (already validated above for user-specified residues)
            if len(atom_indices) == 0:
                continue

            # Get residue name (same for all atoms in residue)
            res_name = structure.res_name[atom_indices[0]]

            # Get hydropathic value for this residue (unknown residues have been already removed)
            residue_hydropathy_indices.append(hydropathy_index[res_name])

            # Calculate total SASA for this residue by summing the SASA values of its atoms
            # this makes more sense than averaging the SASA values, as larger residues will
            # naturally have higher SASA and should contribute more to the energy
            res_sasa = np.sum(atom_sasa_values[atom_indices])

            max_sasa = max_theoretical_sasa_for_residues.get(res_name, max_residue_sasa)
            if max_sasa > 0:
                norm_res_sasa = res_sasa / max_sasa
            else:
                norm_res_sasa = 0.0
            # Clamp to [0, 1] as real protein SASA can exceed theoretical per-residue maximums
            norm_res_sasa = float(np.clip(norm_res_sasa, 0.0, 1.0))
            normalized_residue_sasa_values.append(norm_res_sasa)

        # Convert to numpy arrays for efficient indexing
        residue_hydropathy_indices_arr: npt.NDArray[np.floating[Any]] = np.array(residue_hydropathy_indices)
        normalized_residue_sasa_values_arr: npt.NDArray[np.floating[Any]] = np.array(normalized_residue_sasa_values)

        if len(residue_hydropathy_indices_arr) == 0:
            # if no relevant residues, return 0 energy
            return 0.0, 0.0

        # Compute energy based on mode
        value: float = float(np.mean(residue_hydropathy_indices_arr))

        # Apply SASA weighting if in surface or core mode
        # NOTE: the following code calculates the weighted mean of the hydropathy indices of the residues for the 'surface' and 'core' modes,
        # where the weights are the normalised SASA values of the residues. This is different from canonical GRAVY energy calculation,
        # where the GRAVY score is simply the mean of the hydropathy indices of the residues, without any weighting.
        if self.mode == 'surface':
            normalized_sasa = normalized_residue_sasa_values_arr
            sum_of_weights = np.sum(normalized_sasa)
            if sum_of_weights > 0:
                value = float(np.sum(residue_hydropathy_indices_arr * normalized_sasa) / sum_of_weights)
            else:
                value = 0.0  # if all residues are fully buried, then surface contribution is 0
        elif self.mode == 'core':
            normalized_sasa = 1.0 - normalized_residue_sasa_values_arr
            sum_of_weights = np.sum(normalized_sasa)
            if sum_of_weights > 0:
                value = float(np.sum(residue_hydropathy_indices_arr * normalized_sasa) / sum_of_weights)
            else:
                value = 0.0  # if all residues are fully exposed, then core contribution is 0
        return float(value), float(value * self.weight)


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


class ipSAEEnergy(EnergyTerm):
    r"""
    Energy representing the ipSAE score of Dunbrack (2025, bioRxiv), a function of
    the PAE matrix that measures the confidence in the relative placement of two groups of residues, focusing on the
    pairs that the model is confident about.

    Like ipTM, ipSAE is a predicted TM-score restricted to inter-group residue pairs. Unlike ipTM, for each residue *i*
    of the first group (the one the structures are aligned on) only the residues *j* of the second group with
    :math:`\mathrm{PAE}_{ij} <` ``pae_cutoff`` are used, and the TM-score length scale :math:`d_0` is set by how many
    there are, rather than by the size of the whole complex:

    .. math:: n_0(i) = |\{j : \mathrm{PAE}_{ij} < \mathrm{cutoff}\}|, \quad
        d_0(i) = \max\left(1, 1.24\,(\max(n_0(i), 27) - 15)^{1/3} - 1.8\right)

    .. math:: \mathrm{ipSAE}_{1\to 2} = \max_{i} \frac{1}{n_0(i)} \sum_{j:\, \mathrm{PAE}_{ij} < \mathrm{cutoff}}
        \frac{1}{1 + (\mathrm{PAE}_{ij} / d_0(i))^2}

    The maximum is over residues *i* with at least one such partner. This makes the score insensitive to parts of the
    groups that are far from the interface, or disordered, which dilute ipTM even when the interface is well predicted.
    The score is between 0 and 1, and is returned with a negative sign so that it is an energy to minimise.

    The score is directional, since the PAE matrix is not symmetric: row *i* is the residue the structures are aligned
    on and column *j* the residue whose position error is measured. ``direction`` selects the direction used, or how
    the two directions are combined. The default, 'min', is the smaller of the two directions (the stricter score,
    which needs the interface to be confident both ways); 'max' is the ipSAE_max of the original implementation and
    'mean' the average of the two.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[list[Residue]],
        pae_cutoff: float = 10.0,
        direction: Literal['max', 'min', 'mean', '1to2', '2to1'] = 'min',
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises the ipSAE energy class.

        Parameters
        ----------
        oracle: Oracle
            The oracle to use for the energy term.
        residues: tuple[list[Residue], list[Residue]]
            Which residues to include in the first and second group.
        pae_cutoff: float = 10.0
            The cutoff value for the PAE, in Angstroms, below which a residue pair is used. The default is that of the
            original implementation.
        direction: {'max', 'min', 'mean', '1to2', '2to1'}, default='min'
            '1to2' aligns on residues of the first group and scores those of the second; '2to1' does the opposite.
            'max', 'min' and 'mean' are the larger, the smaller and the average of the two.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = 'ipSAE'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

        assert pae_cutoff > 0, 'pae_cutoff must be positive'
        assert direction in ('max', 'min', 'mean', '1to2', '2to1'), (
            "direction must be 'max', 'min', 'mean', '1to2' or '2to1'"
        )
        self.pae_cutoff = pae_cutoff
        self.direction = direction

        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
        if len(residues) == 1:
            self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[0])]
        else:
            self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'pae' in self.oracle.result_class.model_fields, (
            'ipSAEEnergy requires oracle to return pae in result_class'
        )

    def _directional_ipsae(
        self,
        pae: npt.NDArray[np.float64],
        aligned_mask: npt.NDArray[np.bool_],
        scored_mask: npt.NDArray[np.bool_],
    ) -> float:
        """ipSAE aligning on the residues in ``aligned_mask`` and scoring those in ``scored_mask``."""
        pair_mask = aligned_mask[:, np.newaxis] & scored_mask[np.newaxis, :]
        pair_mask &= ~np.eye(len(pae), dtype=bool)  # uncertainty in distance between a residue and itself is ignored
        valid = pair_mask & (pae < self.pae_cutoff)

        n0 = valid.sum(axis=1)  # for each aligned residue, how many scored residues are confidently placed
        has_partner = n0 > 0
        if not np.any(has_partner):
            return 0.0

        # d0 of the TM-score, with the same lower bounds as the original implementation
        d0 = np.maximum(1.0, 1.24 * (np.maximum(n0, 27) - 15.0) ** (1.0 / 3.0) - 1.8)
        tm_terms = 1.0 / (1.0 + (pae / d0[:, np.newaxis]) ** 2)
        per_residue = np.sum(tm_terms * valid, axis=1) / np.maximum(n0, 1)
        return float(np.max(per_residue[has_partner]))

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        folding_result = oracles_result[self.oracle]
        structure = oracles_result.get_structure(self.oracle)
        assert hasattr(folding_result, 'pae'), 'pae metric not returned by folding algorithm'
        assert folding_result.pae.shape[0] == 1, 'batch size equal to 1 is required'
        pae = np.asarray(folding_result.pae[0], dtype=np.float64)  # [n_residues, n_residues], in Angstroms

        group_1_mask = self.get_residue_mask(structure, residue_group_index=0)
        group_2_mask = self.get_residue_mask(structure, residue_group_index=1)

        if self.direction == '1to2':
            ipsae = self._directional_ipsae(pae, group_1_mask, group_2_mask)
        elif self.direction == '2to1':
            ipsae = self._directional_ipsae(pae, group_2_mask, group_1_mask)
        else:
            both = [
                self._directional_ipsae(pae, group_1_mask, group_2_mask),
                self._directional_ipsae(pae, group_2_mask, group_1_mask),
            ]
            if self.direction == 'max':
                ipsae = max(both)
            elif self.direction == 'min':
                ipsae = min(both)
            else:
                ipsae = float(np.mean(both))

        value = -ipsae  # negative because you want it to be interpreted as an energy
        return value, value * self.weight


class SaltBridgeEnergy(EnergyTerm):
    r"""
    Energy that rewards salt bridges *between* two groups of residues, and optionally penalises the repulsive contacts
    between residues of the same charge.

    Following the classic definition of Barlow & Thornton (1983, J. Mol. Biol. 168, 867),
    two charged residues form a salt bridge when any carboxylate oxygen of an acidic residue (Asp: OD1, OD2; Glu: OE1,
    OE2) is within ``distance_cutoff`` (4 Angstrom) of any nitrogen of a basic residue (Lys: NZ; Arg: NE, NH1, NH2; His:
    ND1, NE2). Here, the same atoms and cutoff are used to detect the opposite: two residues of the **same** charge that
    are this close, here called a negative bridge. A pair of residues is counted at most once, however many atom pairs
    are within the cutoff, and only pairs with one residue in each of the two groups count, so bridges internal to a
    group are ignored. The signed number of bridges is

    .. math:: n = n_{\mathrm{salt}} - n_{\mathrm{negative}}

    and the energy is :math:`-n`, which decreases as there are more salt bridges and fewer negative bridges. It is
    a count: it changes in steps, and is unbounded.

    .. note::
        The positions of side chains come from the folding model and are less reliable than those of the backbone,
        especially at interfaces. Histidine is included by default as in the original definition, but it is mostly
        neutral at physiological pH: remove ``'H'`` from ``residue_types`` to ignore it. Charges are assigned by residue
        type only, termini are ignored, and no account is taken of desolvation.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[list[Residue]],
        residue_types: list[str] | tuple[str, ...] = ('D', 'E', 'K', 'R', 'H'),
        distance_cutoff: float = 4.0,
        count_same_charge: bool = True,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises the salt bridge energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        residues: list[list[Residue]]
            Which residues to include in the first and second group. Bridges are counted between the two groups only,
            so the groups cannot share residues.
        residue_types: list[str], default=('D', 'E', 'K', 'R', 'H')
            The types of residue considered, as 1-letter codes. Must be a subset of D, E, K, R and H, the residues with
            a charged group defined. Residues of any other type in the groups are never counted.
        distance_cutoff: float, default=4.0
            Distance in Angstrom between charged atoms, up to and including which two residues are in contact.
        count_same_charge: bool, default=True
            Whether contacts between residues of the same charge count as negative bridges, each subtracting one from
            the number of salt bridges. If False, only salt bridges are counted.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        base_name = 'salt_bridge'

        if name is None:
            name = base_name
        else:
            name = f'{base_name}_{name}'

        super().__init__(name=name, inheritable=inheritable, oracle=oracle, weight=weight)
        assert len(residues) == 2, 'SaltBridgeEnergy requires exactly two groups of residues'
        unsupported = [r for r in residue_types if r not in salt_bridge_residues]
        assert len(unsupported) == 0, (
            f'residue_types can only contain {list(salt_bridge_residues)}, found unsupported types: {unsupported}'
        )
        assert len(residue_types) > 0, 'residue_types cannot be empty'
        assert distance_cutoff > 0, 'distance_cutoff must be positive'
        group_1, group_2 = [set(zip(*residue_list_to_group(group))) for group in residues]
        assert group_1.isdisjoint(group_2), (
            f'the two groups cannot share residues, found shared: {sorted(group_1 & group_2)}'
        )
        self.residue_types = tuple(residue_types)
        self.distance_cutoff = distance_cutoff
        self.count_same_charge = count_same_charge
        self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        # keyed by the 3-letter name in the structure: the sign of the charge and the atoms that carry it
        self._charged_atoms = {aa_dict[letter]: salt_bridge_residues[letter] for letter in self.residue_types}
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'SaltBridgeEnergy requires oracle to return structure in result_class'
        )

    def _charged_sites(
        self, structure: AtomArray, residue_group_index: int
    ) -> list[tuple[int, npt.NDArray[np.float64]]]:
        """The charge sign and the coordinates of the charged atoms of each considered residue in a group."""
        chain_ids, res_ids = self.residue_groups[residue_group_index]
        sites = []
        for chain_id, res_id in dict.fromkeys(zip(chain_ids, res_ids)):  # unique, in order
            residue_atoms = structure[(structure.chain_id == chain_id) & (structure.res_id == res_id)]
            if len(residue_atoms) == 0 or str(residue_atoms.res_name[0]) not in self._charged_atoms:
                continue
            charge, atom_names = self._charged_atoms[str(residue_atoms.res_name[0])]
            coordinates = residue_atoms.coord[np.isin(residue_atoms.atom_name, atom_names)]
            if len(coordinates) > 0:
                sites.append((charge, coordinates))
        return sites

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        sites_1 = self._charged_sites(structure, residue_group_index=0)
        sites_2 = self._charged_sites(structure, residue_group_index=1)

        n_salt_bridges = 0
        n_negative_bridges = 0
        for charge_1, coordinates_1 in sites_1:
            for charge_2, coordinates_2 in sites_2:
                separations = np.linalg.norm(coordinates_1[:, np.newaxis, :] - coordinates_2[np.newaxis, :, :], axis=2)
                if np.min(separations) > self.distance_cutoff:
                    continue
                if charge_1 != charge_2:
                    n_salt_bridges += 1
                elif self.count_same_charge:
                    n_negative_bridges += 1

        value = float(n_negative_bridges - n_salt_bridges)  # minus the signed number of salt bridges
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
    Energy that minimizes the distance between two groups of residues. ``distance`` selects how the
    separation is measured. With 'mean' (the default) the position of each group is the centroid of the
    backbone atoms of its residues and the separation is the distance between the two centroids. With 'min'
    the separation is the minimum distance between any representative side-chain atom (CB, or CA for glycine)
    of a residue in the first group and any representative atom of a residue in the second group.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: tuple[list[Residue], list[Residue]],
        function: Callable[[float], float] | None = None,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
        distance: Literal['mean', 'min'] = 'mean',
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
            Optional callable f(x) applied to the separation x (in Å) before weighting.
            If None, the identity function is used (i.e., energy equals the distance).
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        distance: {'mean', 'min'}, default='mean'
            How the separation between the two groups is measured. 'mean' is the distance between the centroids
            of the backbone atoms of each group. 'min' is the minimum distance between any CB atom (CA for
            glycine) of a residue in the first group and any CB atom (CA for glycine) of a residue in the second.
        """
        if name is None:
            name = 'separation'
        else:
            name = f'separation_{name}'

        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        assert distance in ('mean', 'min'), "distance must be 'mean' or 'min'"
        self.distance: Literal['mean', 'min'] = distance
        self.function: Callable[[float], float] | None = function
        if self.function is not None:
            assert callable(self.function), 'Function must be callable and accept a single float argument'
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'SeparationEnergy requires oracle to return structure in result_class'
        )

    @staticmethod
    def _representative_atom_mask(structure: AtomArray) -> npt.NDArray[np.bool_]:
        """Mask selecting the CB atom of each residue, or the CA atom for glycine (which has no CB)."""
        is_glycine = structure.res_name == 'GLY'
        mask = ((structure.atom_name == 'CB') & ~is_glycine) | ((structure.atom_name == 'CA') & is_glycine)
        return np.asarray(mask, dtype=bool)

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        group_1_mask = self.get_atom_mask(structure, residue_group_index=0)
        group_2_mask = self.get_atom_mask(structure, residue_group_index=1)

        if self.distance == 'mean':
            backbone_mask = np.isin(structure.atom_name, backbone_atoms)
            group_1_atoms = structure[backbone_mask & group_1_mask]
            group_2_atoms = structure[backbone_mask & group_2_mask]
            group_1_centroid = np.mean(group_1_atoms.coord, axis=0)
            group_2_centroid = np.mean(group_2_atoms.coord, axis=0)
            separation = float(np.linalg.norm(group_1_centroid - group_2_centroid))
        else:
            representative_mask = self._representative_atom_mask(structure)
            group_1_coords = structure[representative_mask & group_1_mask].coord
            group_2_coords = structure[representative_mask & group_2_mask].coord
            assert len(group_1_coords) > 0 and len(group_2_coords) > 0, (
                'SeparationEnergy with distance="min" found no CB (or glycine CA) atoms in one of the residue groups'
            )
            displacements = group_1_coords[:, np.newaxis, :] - group_2_coords[np.newaxis, :, :]
            separation = float(np.min(np.linalg.norm(displacements, axis=2)))

        value = separation
        if self.function is not None:
            value = float(self.function(separation))

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
            f'Number of reference embeddings ({self.reference_embeddings.shape[0]}) does not '
            f'match number of residues to include in energy term ({len(self.residue_groups[0][0])})'
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
        # Note that n_conserved_residues must be equal to len(self.residue_groups[0][0])
        conserved_embeddings = embeddings[self.conserved_index_list(chains)]

        assert conserved_embeddings.shape == self.reference_embeddings.shape, (
            f'Conserved embeddings shape {conserved_embeddings.shape} does not match reference '
            f'embeddings shape {self.reference_embeddings.shape}. The reference embeddings are fixed '
            f'at construction, so this term cannot follow residues added to or removed from its group '
            f'at runtime (e.g. under GrandCanonical sampling); apply it to a fixed set of residues.'
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


class ShapeComplementarityEnergy(EnergyTerm):
    """
    Purely geometric energy measuring how well the surfaces of two groups of residues interlock - a "lock and
    key" fit. It is blind to the chemistry of the residues involved and depends only on their shape.

    This is a variant of the shape complementarity statistic *Sc* of
    `Lawrence & Colman (1993) <https://doi.org/10.1006/jmbi.1993.1648>`_. A dot representation of the
    solvent-excluded surface of each group is built *in isolation*, i.e. as if the other group were absent. The
    interface is then the set of dots that the partner hides from the solvent. For each interface dot *a* on the
    first group, the nearest interface dot *b* on the second group is found and the pair is scored as

    .. math:: s(a) = -(\\mathbf{n}_a \\cdot \\mathbf{n}_b) \\, e^{-w d_{ab}^2}

    where :math:`\\mathbf{n}` are the outward surface normals, :math:`d_{ab}` is the separation of the two dots and
    :math:`w` is ``distance_decay``. This is +1 for two surfaces that face each other and touch, decays towards 0
    as they move apart or turn from facing each other to merely running parallel, and is negative for surfaces
    pointing the same way. The procedure is symmetrised by repeating it from the second group back onto the first.

    .. note::
        Earlier revisions re-weighted each dot pair by the hydrophobicity of the two residues involved. That was
        removed after benchmarking against measured binding free energies: the weighting improved agreement for
        mutations that *remove* a hydrophobic contact and degraded it by a comparable amount for mutations that
        *introduce* one, leaving no net benefit over a mixed set of designs. Chemistry belongs in the terms built
        for it - :class:`HydrophobicEnergy` and :class:`HydropathyEnergy` - rather than folded into a geometric
        one, where it cannot be reasoned about separately.

    **Extensive by default.** Each dot stands for a definite patch of surface: dots are laid uniformly in solid
    angle, so a dot on an atom of radius :math:`r` represents :math:`4 \\pi r^2 / N` of it. The energy is therefore
    a surface integral over the buried interface rather than an average over it,

    .. math:: E = -\\frac{1}{2 A_0} \\sum_{a \\in I_A \\cup I_B} s_a \\, \\delta A_a

    where :math:`A_0` is ``area_scale`` and the half undoes the double counting from summing over both sides. Read
    physically, this is an interfacial free energy per unit area, discounted by :math:`s` wherever the surfaces fit
    badly. Doubling a well-packed interface doubles the energy, so ``weight`` genuinely scales the result rather
    than merely redistributing it. On flat test interfaces the energy follows :math:`a n^2 + b n` - a bulk term plus a
    perimeter correction - to within 1% over a sixteenfold range of area, with the perimeter share falling from
    about 20% of a 16-residue patch to 6% of a 256-residue one.

    Setting ``scaling='intensive'`` instead reports the plain *Sc* statistic, a per-dot average bounded in
    [-1, 1]. That is the right choice for *describing* an interface, and the wrong one for driving a simulation:
    see the warnings below.

    .. note::
        This term is **short ranged**. A dot only counts while the partner actually shields it from the solvent, so
        the energy falls to exactly 0 once the two surfaces are more than about twice the probe radius apart and a
        solvent molecule can slip between them - roughly 3 Angstrom of clearance. Inside that range it varies
        smoothly, but beyond it the landscape is flat, so this term can refine and rank a contact that already
        exists but cannot pull two partners into contact. Pair it with :class:`SeparationEnergy` or
        :class:`FlexEvoBindEnergy` to do that.

    .. warning::
        Being extensive, the energy grows with the size of the system, which matters in two places. In
        :class:`~bagel.mutation.GrandCanonical` sampling it creates a standing incentive to grow the chains, so it
        should be balanced by a :class:`ChemicalPotentialEnergy`. And its magnitude depends on ``area_scale``
        rather than being bounded, so check it is comparable with the other terms in your energy function before
        choosing ``weight``. The default ``area_scale`` of 1000 square Angstrom puts a typical designed interface
        in the region of -0.5, which sits naturally alongside the other terms in this module.

    .. warning::
        With ``scaling='intensive'`` the value is a per-dot average, which has two consequences worth knowing.
        It saturates rather than growing with contact area, so it will not reward a larger interface. Worse,
        because the average runs over the dots that are *currently* buried, a design can improve it by letting a
        badly packed region recede out of contact rather than by repairing it - the offending dots simply leave the
        average. The extensive form has neither problem, which is why it is the default. Intensive values are also
        *not* comparable with published *Sc* figures: Lawrence & Colman build a full analytic molecular surface
        including the concave reentrant patches, and report around 0.70-0.75 for well-packed oligomeric interfaces,
        whereas only the convex contact patches are reconstructed here and a native interface lands nearer 0.5.

    .. note::
        The calculation is sensitive to side chain placement, so it is only as trustworthy as the structure the
        oracle predicts. It is worth pairing it with a confidence term (:class:`PLDDTEnergy`, :class:`PAEEnergy`)
        over the interface residues, so that confidently wrong interfaces are not rewarded.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: tuple[list[Residue], list[Residue]],
        scaling: Literal['extensive', 'intensive'] = 'extensive',
        area_scale: float = 1000.0,
        interface_cutoff: float = 6.0,
        distance_decay: float = 0.5,
        statistic: Literal['mean', 'median'] = 'mean',
        n_surface_points: int = 150,
        probe_radius: float | None = None,
        normal_smoothing: float | None = None,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises shape complementarity energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle to use for the energy term.
        residues: tuple[list[Residue], list[Residue]]
            A tuple containing two lists of residues, defining the two sides of the interface. These are usually
            the residues of the two chains being docked, but any two disjoint sets work, for instance two domains
            of the same chain.
        scaling: {'extensive', 'intensive'}, default='extensive'
            'extensive' integrates the weighted fit over the buried surface, so the energy is proportional to how
            much interface there is as well as how good it is. This is the right choice for an energy function.
            'intensive' instead reports the per-dot average, i.e. the plain Sc statistic bounded in [-1, 1], which
            is useful for describing or reporting on an interface but is a poor thing to minimise.
        area_scale: float, default=1000.0
            Area, in square Angstrom, that the extensive energy is divided by, so that a typical interface gives a
            value of order 1 rather than of order several hundred. Ignored when ``scaling='intensive'``. Multiply
            the reported energy by this to recover the fit-weighted buried area in square Angstrom.
        interface_cutoff: float, default=6.0
            Maximum separation, in Angstrom, at which two surface dots may still be paired. Interface dots with no
            partner within this distance are dropped from the statistic. This mainly trims the ragged rim of the
            interface; the interface itself is defined by burial, not by this cutoff.
        distance_decay: float, default=0.5
            The :math:`w` in :math:`e^{-w d^2}`, in Angstrom^-2. The Lawrence-Colman value is 0.5, which halves the
            contribution of a dot pair separated by about 1.2 Angstrom. Larger values demand tighter packing.
        statistic: {'mean', 'median'}, default='mean'
            How individual dot scores are aggregated when ``scaling='intensive'``; ignored otherwise. Lawrence &
            Colman use the median, which is more robust to a handful of badly matched dots. The mean varies more
            smoothly with the sequence and is therefore the better behaved of the two as an energy.
        n_surface_points: int, default=150
            Number of dots generated per atom before burial is tested. Larger values reduce the discretisation
            noise of the statistic, at a proportional cost in runtime.
        probe_radius: float or None, default=None
            Radius of the rolling solvent probe. It sets both how much of the crevice between neighbouring
            atoms is smoothed out of each surface, and which dots count as buried by the partner and therefore
            as interface. Defaults to the van der Waals radius of water.
        normal_smoothing: float or None, default=None
            Length scale, in Angstrom, over which surface normals are averaged across neighbouring atoms, which
            imitates the smoothing that the reentrant patches of a true solvent-excluded surface would provide.
            Defaults to the probe radius. Set to 0 to use the raw per-atom radial normals, which makes the term
            measure complementarity atom against atom and therefore much noisier.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that
            new residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        if name is None:
            name = 'shape_complementarity'
        else:
            name = f'shape_complementarity_{name}'

        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues[0]), residue_list_to_group(residues[1])]
        self.scaling: Literal['extensive', 'intensive'] = scaling
        self.area_scale = area_scale
        self.interface_cutoff = interface_cutoff
        self.distance_decay = distance_decay
        self.statistic: Literal['mean', 'median'] = statistic
        self.n_surface_points = n_surface_points
        self.probe_radius = probe_radius_water if probe_radius is None else probe_radius
        self.normal_smoothing = self.probe_radius if normal_smoothing is None else normal_smoothing
        self._unit_sphere = _fibonacci_sphere(n_surface_points)

        assert len(residues[0]) > 0 and len(residues[1]) > 0, 'both residue groups must be non-empty'
        assert scaling in ('extensive', 'intensive'), f'unknown scaling {scaling}'
        assert area_scale > 0, 'area_scale must be positive'
        assert interface_cutoff > 0, 'interface_cutoff must be positive'
        assert distance_decay >= 0, 'distance_decay must be non-negative'
        assert statistic in ('mean', 'median'), f'unknown statistic {statistic}'
        assert n_surface_points > 0, 'n_surface_points must be positive'
        assert self.probe_radius >= 0, 'probe_radius must be non-negative'
        assert self.normal_smoothing >= 0, 'normal_smoothing must be non-negative'
        overlap = {(res.chain_ID, res.index) for res in residues[0]} & {
            (res.chain_ID, res.index) for res in residues[1]
        }
        if overlap:
            warnings.warn(
                f'{len(overlap)} residue(s) appear in both groups of {name}; a residue cannot be complementary to '
                'itself, so the resulting energy will be hard to interpret.'
            )
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'ShapeComplementarityEnergy requires oracle to return structure in result_class'
        )

    def _score_against(
        self,
        dots: npt.NDArray[np.float64],
        normals: npt.NDArray[np.float64],
        areas: npt.NDArray[np.float64],
        other_dots: npt.NDArray[np.float64],
        other_normals: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Scores every dot of one interface against its nearest partner on the other.

        Returns ``(scores, areas)`` for the dots that found a partner, where ``areas`` is the patch of surface
        each dot stands for. Dots with no partner within ``interface_cutoff`` are dropped entirely rather than
        scored as zero, so they contribute neither reward nor penalty.
        """
        empty = np.zeros(0, dtype=np.float64)
        partner, distance = _nearest_dot(dots, other_dots, self.interface_cutoff)
        matched = partner >= 0
        if not matched.any():
            return empty, empty

        partner = partner[matched]
        facing = -np.sum(normals[matched] * other_normals[partner], axis=-1)
        scores = facing * np.exp(-self.distance_decay * distance[matched] ** 2)
        return scores, areas[matched]

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        group_1 = structure[self.get_atom_mask(structure, residue_group_index=0)]
        group_2 = structure[self.get_atom_mask(structure, residue_group_index=1)]
        if len(group_1) == 0 or len(group_2) == 0:
            return 0.0, 0.0

        coords_1 = np.asarray(group_1.coord, dtype=np.float64)
        coords_2 = np.asarray(group_2.coord, dtype=np.float64)
        radii_1 = _atom_vdw_radii(group_1)
        radii_2 = _atom_vdw_radii(group_2)

        # Only atoms close enough to the other group can carry an interface dot, so build surface for those alone.
        # A dot sits up to one van der Waals radius from its own atom and may be paired with a dot up to one radius
        # beyond a partner atom, hence the padding on top of the cutoff.
        seed_radius = self.interface_cutoff + float(radii_1.max()) + float(radii_2.max())
        seed_1 = _has_neighbour_within(coords_1, coords_2, seed_radius)
        seed_2 = _has_neighbour_within(coords_2, coords_1, seed_radius)

        dots_1, normals_1, parents_1 = _molecular_surface_dots(
            coords_1, radii_1, self.probe_radius, self._unit_sphere, seed_1
        )
        dots_2, normals_2, parents_2 = _molecular_surface_dots(
            coords_2, radii_2, self.probe_radius, self._unit_sphere, seed_2
        )

        # the interface is the part of each surface that the partner hides from the solvent. Burial is decided
        # with the raw radial normals, which point exactly along the probe's line of approach to the atom.
        interface_1 = _buried_by_partner(dots_1, normals_1, coords_2, radii_2, self.probe_radius)
        interface_2 = _buried_by_partner(dots_2, normals_2, coords_1, radii_1, self.probe_radius)
        if not interface_1.any() or not interface_2.any():
            return 0.0, 0.0

        # Dots are laid uniformly in solid angle, so each stands for an equal patch of its parent atom's sphere.
        # This is what lets the term be summed into a surface integral rather than only averaged.
        areas_1 = 4.0 * np.pi * radii_1[parents_1][interface_1] ** 2 / self.n_surface_points
        areas_2 = 4.0 * np.pi * radii_2[parents_2][interface_2] ** 2 / self.n_surface_points
        dots_1, normals_1 = dots_1[interface_1], normals_1[interface_1]
        dots_2, normals_2 = dots_2[interface_2], normals_2[interface_2]
        if self.normal_smoothing > 0:
            normals_1 = _smoothed_normals(dots_1, coords_1, radii_1, self.normal_smoothing)
            normals_2 = _smoothed_normals(dots_2, coords_2, radii_2, self.normal_smoothing)

        # symmetrise: score group 1 against group 2 and vice versa, then pool the two sets of dot scores
        scores_12, areas_12 = self._score_against(dots_1, normals_1, areas_1, dots_2, normals_2)
        scores_21, areas_21 = self._score_against(dots_2, normals_2, areas_2, dots_1, normals_1)
        scores = np.concatenate([scores_12, scores_21])
        areas = np.concatenate([areas_12, areas_21])

        if self.scaling == 'extensive':
            # a surface integral of the quality of fit over the buried surface. The half undoes the double
            # counting from summing over both sides of the interface.
            value = -0.5 * float(np.sum(scores * areas)) / self.area_scale
        else:
            value = -_statistic(scores, self.statistic)
        return value, value * self.weight


class SAEnergy(EnergyTerm):
    r"""
    Linear energy over sparse-autoencoder (SAE) features of a protein.

    The oracle (a :class:`~bagel.oracles.embedding.sae.SAE` oracle) returns one
    per-protein feature vector :math:`f \in \mathbb{R}^{F}` — the max activation of
    each of the :math:`F` SAE features across the residues. This term selects a
    user-chosen subset of features and returns the **negated** weighted sum:

    .. math::

        E = -\sum_{i \in \mathcal{I}} c_i \, f_i

    where :math:`\mathcal{I}` are the selected ``feature_indices`` and
    :math:`c_i` are the ``coefficients`` (all ``1`` by default).

    Two defaults make positive coefficients intuitive:

    - **Sign** (``maximize=True``, the default): the linear combination is negated,
      so — because BAGEL **minimizes** energy — a **positive** coefficient *promotes*
      its feature (drives the activation up) and a negative coefficient suppresses
      it. Set ``maximize=False`` to minimize the features instead (positive
      coefficient suppresses).
    - **Normalization** (``normalize_coefficients=True``, the default): the
      coefficients are rescaled so that :math:`\sum_i |c_i| = 1`. This makes the
      energy scale invariant to how many features you select and to the absolute
      size of the coefficients, so ``weight`` alone controls the term's magnitude
      relative to other energies. Only the *relative* coefficients matter.

    This makes it easy to steer a design toward or away from the concepts encoded
    by specific SAE features (e.g. a catalytic-motif feature, a beta-barrel
    feature, etc.).
    """

    def __init__(
        self,
        oracle: EmbeddingOracle,
        feature_indices: list[int],
        coefficients: list[float] | None = None,
        weight: float = 1.0,
        name: str | None = None,
        maximize: bool = True,
        normalize_coefficients: bool = True,
    ) -> None:
        """
        Initialises the SAE energy term.

        Parameters
        ----------
        oracle: EmbeddingOracle
            The SAE oracle whose result exposes a per-protein ``features`` vector.
        feature_indices: list[int]
            Indices of the SAE features to include in the energy. Must be
            non-empty and unique.
        coefficients: list[float] | None = None
            Per-feature linear coefficients :math:`c_i`, aligned with
            ``feature_indices``. Defaults to all ones.
        weight: float = 1.0
            Overall weight of the energy term.
        name: str | None = None
            Optional suffix appended to the energy term name.
        maximize: bool = True
            If ``True`` (default) the linear combination is negated so that, under
            BAGEL's energy minimization, a positive coefficient *promotes* its
            feature. Set to ``False`` to minimize the features instead.
        normalize_coefficients: bool = True
            If ``True`` (default) the coefficients are rescaled so that
            :math:`\\sum_i |c_i| = 1`, making the energy invariant to the number of
            selected features and the absolute coefficient scale. Requires at least
            one non-zero coefficient.
        """
        if name is None:
            name = 'sae'
        else:
            name = f'sae_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=True, weight=weight)

        self.feature_indices, self.coefficients = _prepare_sae_feature_terms(
            feature_indices, coefficients, normalize_coefficients
        )
        self.maximize = bool(maximize)

        assert isinstance(self.oracle, EmbeddingOracle), 'Oracle must be an instance of EmbeddingOracle'
        assert 'features' in self.oracle.result_class.model_fields, (
            'SAEnergy requires the oracle to return a per-protein "features" vector in its result_class'
        )
        _require_supported_sae_model(self.oracle)

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        result = oracles_result[self.oracle]
        features = np.asarray(getattr(result, 'features'), dtype=np.float64)
        if features.ndim != 1:
            raise ValueError(
                f'SAE features are expected to be a 1D per-protein vector, not shape: {features.shape}. '
                'This does not work with batches.'
            )
        max_index = int(self.feature_indices.max())
        if max_index >= features.shape[0]:
            raise IndexError(
                f'feature index {max_index} is out of range for a feature vector of length {features.shape[0]}.'
            )
        linear_combination = float(np.sum(self.coefficients * features[self.feature_indices]))
        value = -linear_combination if self.maximize else linear_combination
        return value, value * self.weight


class ResidueSAEnergy(EnergyTerm):
    r"""
    Linear energy over **per-residue** sparse-autoencoder (SAE) features.

    Where :class:`SAEnergy` acts on the whole-protein max-pooled feature vector,
    this term acts on the dense per-residue activations
    :math:`A \in \mathbb{R}^{R \times F}` (``SAEResult.embeddings``), optionally
    restricted to a group of residues, and pools them over the selected residues to
    one scalar per feature:

    .. math::

        E = -\sum_{i \in \mathcal{I}} c_i \, \operatorname{pool}_{r \in \mathcal{R}} A_{r, i}

    where :math:`\mathcal{R}` are the selected residues, :math:`\operatorname{pool}`
    is ``max`` / ``mean`` / ``sum``, and the sign / normalization conventions match
    :class:`SAEnergy` (positive coefficients *promote* their feature by default; the
    coefficients are L1-normalized so :math:`\sum_i |c_i| = 1`).

    The ``max`` pooling over *all* residues reproduces :class:`SAEnergy`; ``mean``
    rewards a feature that is *pervasive* across the region, ``sum`` rewards total
    feature mass (grows with the region size).

    Residues are selected with BAGEL's usual idiom — a list of :class:`~bagel.chain.Residue`
    objects — and their embedding rows are resolved automatically in the correct
    multichain order (see :meth:`EnergyTerm.get_embedding_residue_mask`).

    Requires the SAE **oracle** to be built with ``include_per_residue=True`` so the
    per-residue activations are populated.
    """

    def __init__(
        self,
        oracle: EmbeddingOracle,
        feature_indices: list[int],
        coefficients: list[float] | None = None,
        weight: float = 1.0,
        name: str | None = None,
        maximize: bool = True,
        normalize_coefficients: bool = True,
        residues: list[Residue] | None = None,
        pooling: Literal['max', 'mean', 'sum'] = 'mean',
    ) -> None:
        """
        Initialises the per-residue SAE energy term.

        Parameters
        ----------
        oracle: EmbeddingOracle
            The SAE oracle. Must be configured with ``include_per_residue=True`` so
            its result exposes per-residue activations in ``embeddings``.
        feature_indices: list[int]
            Indices of the SAE features to include. Must be non-empty and unique.
        coefficients: list[float] | None = None
            Per-feature linear coefficients, aligned with ``feature_indices``.
            Defaults to all ones.
        weight: float = 1.0
            Overall weight of the energy term.
        name: str | None = None
            Optional suffix appended to the energy term name.
        maximize: bool = True
            If ``True`` (default) the linear combination is negated so a positive
            coefficient *promotes* its feature under BAGEL's minimization.
        normalize_coefficients: bool = True
            If ``True`` (default) coefficients are L1-normalized (:math:`\\sum_i |c_i| = 1`).
        residues: list[Residue] | None = None
            Residues over which to pool. ``None`` (default) uses **all** residues.
        pooling: {'max', 'mean', 'sum'} = 'mean'
            How to reduce the selected residues to one value per feature.
        """
        name = 'residue_sae' if name is None else f'residue_sae_{name}'
        # Residue-selective term: freeze the residue set (do not inherit residues
        # added later under grand-canonical moves).
        super().__init__(name=name, oracle=oracle, inheritable=False, weight=weight)

        self.feature_indices, self.coefficients = _prepare_sae_feature_terms(
            feature_indices, coefficients, normalize_coefficients
        )
        self.maximize = bool(maximize)

        if pooling not in ('max', 'mean', 'sum'):
            raise ValueError(f"pooling must be 'max', 'mean', or 'sum'; got {pooling!r}.")
        self.pooling = pooling

        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []

        assert isinstance(self.oracle, EmbeddingOracle), 'Oracle must be an instance of EmbeddingOracle'
        # Require the SAE-specific "features" field (as SAEnergy does), not just the
        # ubiquitous "embeddings"; otherwise a plain embedding oracle (e.g. ESMC)
        # would pass and its raw hidden states would be scored as SAE activations.
        assert {'embeddings', 'features'} <= set(self.oracle.result_class.model_fields), (
            'ResidueSAEnergy requires an SAE oracle exposing per-residue "embeddings" and "features" '
            'in its result_class'
        )
        _require_supported_sae_model(self.oracle)

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        result = oracles_result[self.oracle]
        activations = getattr(result, 'embeddings', None)
        if activations is None:
            raise ValueError(
                'ResidueSAEnergy needs per-residue activations; build the SAE oracle with include_per_residue=True.'
            )
        activations = np.asarray(activations, dtype=np.float64)
        if activations.ndim != 2:
            raise ValueError(
                f'per-residue SAE activations must be 2D (residues, features); got shape {activations.shape}.'
            )

        if self.residue_groups:
            mask = self.get_embedding_residue_mask(
                result.input_chains,
                0,
                getattr(result, 'chain_index', None),
                getattr(result, 'residue_index', None),
            )
            if mask.shape[0] != activations.shape[0]:
                raise ValueError(
                    f'residue mask length ({mask.shape[0]}) does not match the number of activation rows '
                    f'({activations.shape[0]}).'
                )
            rows = activations[mask]
            if rows.shape[0] == 0:
                raise ValueError('No embedding rows matched the selected residues for ResidueSAEnergy.')
        else:
            # Pool over all residues; guard against padded/extra activation rows by
            # reconciling the row count with the residues in input_chains (the
            # selected-residues branch does the same via the mask length).
            expected_rows = sum(len(chain.residues) for chain in result.input_chains)
            if activations.shape[0] != expected_rows:
                raise ValueError(
                    f'per-residue SAE activations have {activations.shape[0]} rows but input_chains has '
                    f'{expected_rows} residues; the activation rows may include padding.'
                )
            rows = activations

        if self.pooling == 'max':
            pooled = rows.max(axis=0)
        elif self.pooling == 'mean':
            pooled = rows.mean(axis=0)
        else:  # 'sum'
            pooled = rows.sum(axis=0)

        max_index = int(self.feature_indices.max())
        if max_index >= pooled.shape[0]:
            raise IndexError(
                f'feature index {max_index} is out of range for a feature vector of length {pooled.shape[0]}.'
            )
        linear_combination = float(np.sum(self.coefficients * pooled[self.feature_indices]))
        value = -linear_combination if self.maximize else linear_combination
        return value, value * self.weight


class BuriedHistidineEnergy(EnergyTerm):
    """
    Rewards histidines that are buried, saturating at `target_count` of them.

    This is the core term of a histidine-based pH switch. Histidine is the only
    canonical side chain whose pKa (~6.0-6.5 free in solution) sits between pH 6
    and pH 7.4. At pH 7.4 the imidazole is neutral and packs acceptably in a core;
    at pH 6 it protonates to the imidazolium cation, and burying a charged group in
    a low-dielectric core with no counter-ion costs a large desolvation penalty.
    That penalty is paid only by the folded state, so each buried histidine
    subtracts from the folding free energy as the pH drops.

    Each histidine contributes a burial score in [0, 1]:

        burial_i = 1 - min(relative_sasa_i / sasa_cutoff, 1)

    i.e. 1 when the side chain is completely occluded, falling linearly to 0 once
    its relative SASA reaches `sasa_cutoff`. The returned energy is

        E = -min(sum_i burial_i, target_count) / target_count       in [-1, 0]

    The saturation matters. Without it the optimiser keeps stuffing in histidines
    forever and you end up with a polyhistidine blob that does not fold at any pH.
    Capping the reward at `target_count` says "six buried histidines is the goal,
    a seventh buys you nothing", which leaves the remaining core positions free to
    be filled by ordinary hydrophobics that pay for the fold at neutral pH.

    Note that this term pulls against :class:`HydropathyEnergy` in 'core' mode:
    histidine's Kyte-Doolittle index is -3.2, so every buried histidine makes the
    burial-weighted hydropathy worse. That tension is deliberate, since it is what
    stops the core becoming all histidine, but it makes the ratio of the two
    weights the most important knob in a pH-switch design.

    Parameters
    ----------
    oracle : FoldingOracle
        Oracle supplying the predicted structure.
    target_count : int, default=6
        Number of buried histidines that saturates the reward. For a ~100-residue
        single domain, 4-8 is a sensible window: below 4 the pH-6 destabilisation
        is likely too small to unfold the protein, above ~8 you are unlikely to
        retain a folded state at pH 7.4 at all.
    sasa_cutoff : float, default=0.15
        Relative SASA at and above which a histidine counts as fully exposed.
        0.15 is a conventional "buried" threshold.
    residues : list[Residue] or None
        Restrict the count to these positions (e.g. only the designed core).
        Default considers every residue in the state.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        target_count: int = 6,
        sasa_cutoff: float = 0.15,
        residues: list[Residue] | None = None,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        name = 'buried_histidine' if name is None else f'buried_histidine_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.target_count = int(target_count)
        self.sasa_cutoff = float(sasa_cutoff)
        assert self.target_count > 0, 'target_count must be positive'
        assert 0.0 < self.sasa_cutoff <= 1.0, 'sasa_cutoff must lie in (0, 1]'
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be a FoldingOracle'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        if len(structure) == 0:
            return 0.0, 0.0

        chain_ids, res_ids, res_names, relative = _relative_residue_sasa(structure)

        selected = np.full(len(res_ids), True)
        if len(self.residue_groups) > 0:
            group_chains, group_indices = self.residue_groups[0]
            selected = np.zeros(len(res_ids), dtype=bool)
            for cid in np.unique(group_chains):
                wanted = group_indices[group_chains == cid]
                selected |= (chain_ids == cid) & np.isin(res_ids, wanted)

        his_mask = (res_names == 'HIS') & selected
        if not np.any(his_mask):
            return 0.0, 0.0

        burial = 1.0 - np.minimum(relative[his_mask] / self.sasa_cutoff, 1.0)
        value = -float(min(burial.sum(), self.target_count)) / self.target_count
        return value, value * self.weight


class HistidineCarboxylateContactEnergy(EnergyTerm):
    """
    Penalises imidazole nitrogens sitting close to Asp/Glu carboxylate oxygens.

    A buried His-Asp or His-Glu pair is the classic pKa-raising motif: the
    carboxylate stabilises the protonated imidazolium, so the folded state
    becomes *more* stable as the pH falls. That inverts the switch sought by
    :class:`BuriedHistidineEnergy`, and an optimiser rewarded only for burying
    histidines will happily build these pairs because they are excellent buried
    hydrogen bonds. This term prices them out.

    Energy = min(n_contacts, max_contacts) / max_contacts, in [0, 1]; a contact is
    any ND1/NE2 atom within `distance_cutoff` of any OD1/OD2/OE1/OE2 atom.

    Set `weight` high enough that one contact outweighs the burial reward for one
    histidine, otherwise the trade is still worth making.

    Parameters
    ----------
    oracle : FoldingOracle
        Oracle supplying the predicted structure.
    distance_cutoff : float, default=4.5
        Distance in Angstrom below which an imidazole nitrogen and a carboxylate
        oxygen count as being in contact.
    max_contacts : int, default=4
        Number of contacts at which the penalty saturates at 1.
    """

    IMIDAZOLE_NITROGENS = ('ND1', 'NE2')
    CARBOXYLATE_OXYGENS = ('OD1', 'OD2', 'OE1', 'OE2')

    def __init__(
        self,
        oracle: FoldingOracle,
        distance_cutoff: float = 4.5,
        max_contacts: int = 4,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        name = 'his_carboxylate' if name is None else f'his_carboxylate_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = []
        self.distance_cutoff = float(distance_cutoff)
        self.max_contacts = int(max_contacts)
        assert self.max_contacts > 0, 'max_contacts must be positive'
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be a FoldingOracle'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        if len(structure) == 0:
            return 0.0, 0.0

        his_mask = (structure.res_name == 'HIS') & np.isin(structure.atom_name, self.IMIDAZOLE_NITROGENS)
        acid_mask = np.isin(structure.res_name, ('ASP', 'GLU')) & np.isin(structure.atom_name, self.CARBOXYLATE_OXYGENS)
        if not np.any(his_mask) or not np.any(acid_mask):
            return 0.0, 0.0

        his_coords = structure.coord[his_mask]
        acid_coords = structure.coord[acid_mask]
        distances = np.linalg.norm(his_coords[:, None, :] - acid_coords[None, :, :], axis=-1)
        n_contacts = int(np.count_nonzero(distances < self.distance_cutoff))

        value = float(min(n_contacts, self.max_contacts)) / self.max_contacts
        return value, value * self.weight


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def residue_list_to_group(residues: list[Residue]) -> ResidueGroup:
    """Converts list of residue objects to ResidueGroup required by energy term objects"""
    return (np.array([res.chain_ID for res in residues]), np.array([res.index for res in residues]))


def _fibonacci_sphere(n_points: int) -> npt.NDArray[np.float64]:
    """
    Generates ``n_points`` approximately uniformly distributed unit vectors on a sphere using the golden-spiral
    (Fibonacci) construction. This is deterministic, so repeated evaluations of an energy term that uses it return
    exactly the same number. That matters for Monte Carlo, where a randomly fluctuating energy would be
    indistinguishable from a real change in the landscape.
    """
    assert n_points > 0, 'n_points must be positive'
    indices = np.arange(n_points, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / n_points
    ring_radius = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    golden_angle = np.pi * (1.0 + np.sqrt(5.0))
    theta = golden_angle * indices
    return np.stack([ring_radius * np.cos(theta), ring_radius * np.sin(theta), z], axis=-1)


def _atom_vdw_radii(structure: AtomArray) -> npt.NDArray[np.float64]:
    """Looks up the van der Waals radius of every atom in ``structure``, falling back to carbon for unknown elements."""
    return np.array(
        [vdw_radii.get(str(element).upper(), default_vdw_radius) for element in structure.element], dtype=np.float64
    )


def _has_neighbour_within(
    coords: npt.NDArray[np.float64], other_coords: npt.NDArray[np.float64], radius: float
) -> npt.NDArray[np.bool_]:
    """Flags every coordinate in ``coords`` that has at least one of ``other_coords`` within ``radius`` of it."""
    if coords.shape[0] == 0 or other_coords.shape[0] == 0:
        return np.zeros(coords.shape[0], dtype=bool)
    cell_list = CellList(other_coords, cell_size=max(radius, 1e-3))
    neighbours = np.atleast_2d(cell_list.get_atoms(coords, radius=radius))
    if neighbours.shape[1] == 0:
        return np.zeros(coords.shape[0], dtype=bool)
    return np.asarray((neighbours >= 0).any(axis=1), dtype=bool)


def _molecular_surface_dots(
    coords: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    probe_radius: float,
    unit_sphere: npt.NDArray[np.float64],
    seed_mask: npt.NDArray[np.bool_],
    chunk_size: int = 20000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """
    Builds a dot representation of the contact part of the solvent-excluded ("molecular") surface of a group of
    atoms - the locus of points where a rolling solvent probe touches the group.

    The construction is the standard one. Dots are laid on the solvent-accessible sphere of radius
    (van der Waals + probe) around each atom flagged in ``seed_mask``, which is the surface traced by the probe
    *centre*. A dot is discarded if it falls inside the accessible sphere of any *other* atom in the group, since
    the probe cannot sit there. Each surviving probe centre is then projected inward by the probe radius onto the
    point where the probe actually touches the atom, giving a dot on the van der Waals surface.

    That last step is what distinguishes this from a plain van der Waals dot surface, and it matters a lot here.
    Testing occlusion with the inflated radii removes the dots lining the narrow crevices between neighbouring
    atoms, which no solvent can reach and which no partner protein can complement either. Left in, those dots
    contribute badly matched, near-random normals and drag the complementarity statistic towards zero.

    The van der Waals surface, not the accessible surface, is the right place to put the final dots because two
    molecules in contact have coincident van der Waals surfaces, whereas their accessible surfaces interpenetrate
    by twice the probe radius. Coincidence at contact is what makes a distance-decay complementarity score
    meaningful. A convenient side effect is that the outward normal at a dot is exactly the radial direction from
    its parent atom.

    Note that only the convex contact patches are produced; the concave reentrant patches that the probe sweeps out
    where it bridges two atoms are not represented. Those are a minority of a protein surface, and omitting them
    costs far less accuracy than including the crevices would.

    Returns
    -------
    (dots, normals, parent_atom_indices)
        Coordinates of the surviving dots on the van der Waals surface, their outward unit normals, and the index
        (into ``coords``) of the atom each dot belongs to.
    """
    seed_indices = np.flatnonzero(seed_mask)
    if seed_indices.size == 0 or coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.int_)

    expanded_radii = radii + probe_radius
    n_points = unit_sphere.shape[0]
    offsets = expanded_radii[seed_indices][:, None, None] * unit_sphere[None, :, :]
    probe_centres = (coords[seed_indices][:, None, :] + offsets).reshape(-1, 3)
    normals = np.tile(unit_sphere, (seed_indices.size, 1))
    parents = np.repeat(seed_indices, n_points)

    max_radius = float(expanded_radii.max())
    cell_list = CellList(coords, cell_size=max(max_radius, 1e-3))

    keep = np.ones(probe_centres.shape[0], dtype=bool)
    for start in range(0, probe_centres.shape[0], chunk_size):
        stop = min(start + chunk_size, probe_centres.shape[0])
        block = probe_centres[start:stop]
        neighbours = np.atleast_2d(cell_list.get_atoms(block, radius=max_radius))
        if neighbours.shape[1] == 0:
            continue
        valid = neighbours >= 0
        safe = np.where(valid, neighbours, 0)
        distances = np.linalg.norm(block[:, None, :] - coords[safe], axis=-1)
        occluded = valid & (safe != parents[start:stop, None]) & (distances < expanded_radii[safe] - 1e-9)
        keep[start:stop] = ~occluded.any(axis=1)

    # project each surviving probe centre inward onto the point where the probe touches the atom
    dots = probe_centres[keep] - probe_radius * normals[keep]
    return dots, normals[keep], parents[keep]


def _smoothed_normals(
    dots: npt.NDArray[np.float64],
    coords: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
    smoothing: float,
    chunk_size: int = 20000,
) -> npt.NDArray[np.float64]:
    """
    Replaces the per-atom radial normal at each surface dot with a normal averaged over the nearby atoms.

    The raw normal on a union-of-spheres surface points straight out of whichever single atom owns the dot, so it
    swings through a large angle from one side of an atom to the other. A real solvent-excluded surface is far
    smoother, because the reentrant patches swept out by the rolling probe bridge neighbouring atoms. Those patches
    are not represented here, so their smoothing effect is imitated directly: the normal at a dot is the weighted
    mean of the radial directions from all atoms within reach, each weighted by ``exp(-(d - r) / smoothing)`` in
    its distance from that atom's surface.

    Without this, complementarity is measured atom against atom and the resulting statistic is dominated by the
    bumpiness of individual side chain atoms rather than by the shape of the interface as a whole.
    """
    normals = np.zeros_like(dots)
    if dots.shape[0] == 0 or coords.shape[0] == 0:
        return normals

    reach = float(radii.max()) + 4.0 * smoothing
    cell_list = CellList(coords, cell_size=max(reach, 1e-3))
    for start in range(0, dots.shape[0], chunk_size):
        stop = min(start + chunk_size, dots.shape[0])
        block = dots[start:stop]
        neighbours = np.atleast_2d(cell_list.get_atoms(block, radius=reach))
        if neighbours.shape[1] == 0:
            continue
        valid = neighbours >= 0
        safe = np.where(valid, neighbours, 0)
        offsets = block[:, None, :] - coords[safe]
        distances = np.maximum(np.linalg.norm(offsets, axis=-1), 1e-9)
        weights = np.where(valid, np.exp(-np.clip(distances - radii[safe], 0.0, None) / smoothing), 0.0)
        normals[start:stop] = np.sum(weights[..., None] * offsets / distances[..., None], axis=1)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # a dot with no usable neighbourhood keeps a zero normal, which simply scores 0 rather than blowing up
    return np.asarray(
        np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-9),
        dtype=np.float64,
    )


def _buried_by_partner(
    dots: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    partner_coords: npt.NDArray[np.float64],
    partner_radii: npt.NDArray[np.float64],
    probe_radius: float,
    chunk_size: int = 20000,
) -> npt.NDArray[np.bool_]:
    """
    Flags the surface dots that the partner group hides from the solvent, which is what defines the interface.

    A dot is solvent accessible if a spherical probe of radius ``probe_radius`` can sit tangent to the surface
    there, i.e. with its centre at ``dot + probe_radius * normal``. The dot is buried by the partner if that probe
    centre clashes with any partner atom, that is if it lies within ``partner_radius + probe_radius`` of it.

    Defining the interface this way, rather than by a plain distance cutoff, keeps out dots that merely happen to
    be near the partner while pointing away from it - for example dots on the far side of an interface atom. Those
    would otherwise be scored as badly matched and would dilute the statistic.
    """
    n_dots = dots.shape[0]
    buried = np.zeros(n_dots, dtype=bool)
    if n_dots == 0 or partner_coords.shape[0] == 0:
        return buried

    probe_centres = dots + probe_radius * normals
    clash_radii = partner_radii + probe_radius
    max_clash_radius = float(clash_radii.max())
    cell_list = CellList(partner_coords, cell_size=max(max_clash_radius, 1e-3))

    for start in range(0, n_dots, chunk_size):
        stop = min(start + chunk_size, n_dots)
        block = probe_centres[start:stop]
        neighbours = np.atleast_2d(cell_list.get_atoms(block, radius=max_clash_radius))
        if neighbours.shape[1] == 0:
            continue
        valid = neighbours >= 0
        safe = np.where(valid, neighbours, 0)
        distances = np.linalg.norm(block[:, None, :] - partner_coords[safe], axis=-1)
        buried[start:stop] = (valid & (distances < clash_radii[safe] - 1e-9)).any(axis=1)

    return buried


def _nearest_dot(
    query_dots: npt.NDArray[np.float64],
    target_dots: npt.NDArray[np.float64],
    cutoff: float,
    chunk_size: int = 20000,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """
    For every dot in ``query_dots``, finds the closest dot in ``target_dots`` lying within ``cutoff``.

    Returns
    -------
    (indices, distances)
        Index into ``target_dots`` of the nearest dot, or -1 if none lies within ``cutoff``. The distance is
        ``np.inf`` wherever the index is -1.
    """
    n_query = query_dots.shape[0]
    indices = np.full(n_query, -1, dtype=np.int_)
    distances = np.full(n_query, np.inf, dtype=np.float64)
    if n_query == 0 or target_dots.shape[0] == 0:
        return indices, distances

    cell_list = CellList(target_dots, cell_size=max(cutoff, 1e-3))
    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        block = query_dots[start:stop]
        neighbours = np.atleast_2d(cell_list.get_atoms(block, radius=cutoff))
        if neighbours.shape[1] == 0:
            continue
        valid = neighbours >= 0
        safe = np.where(valid, neighbours, 0)
        block_distances = np.where(valid, np.linalg.norm(block[:, None, :] - target_dots[safe], axis=-1), np.inf)
        rows = np.arange(block.shape[0])
        closest = np.argmin(block_distances, axis=1)
        best_distance = block_distances[rows, closest]
        found = np.isfinite(best_distance)
        indices[start:stop] = np.where(found, safe[rows, closest], -1)
        distances[start:stop] = best_distance

    return indices, distances


def _statistic(values: npt.NDArray[np.float64], statistic: Literal['mean', 'median']) -> float:
    """Computes the mean or median of ``values``. Returns 0.0 if there is nothing to aggregate."""
    if values.size == 0:
        return 0.0
    return float(np.mean(values) if statistic == 'mean' else np.median(values))


# SAE energy terms are only meaningful for the specific SAE they were designed
# around: the ESMC-6B, layer-60, k64, codebook-16384 model. The feature indices and
# their learned "concepts" are model-specific, so mixing in a different SAE would be
# silently wrong.
REQUIRED_SAE_MODEL = 'ESMC-6B-sae-layer60-k64-codebook16384'


_REQUIRED_SAE_MODEL_TOKENS = ('6b', 'layer60', 'k64', 'codebook16384')


def _require_supported_sae_model(oracle: Oracle) -> None:
    """Raise unless the oracle is configured to use the required SAE model.

    Checks the oracle's ``sae_model_id`` (set by :class:`~bagel.oracles.embedding.sae.SAE`).
    Matching is done on normalized tokens so it is robust to the model's date tag
    (e.g. ``esmc-6b-2024-12-sae-layer60-k64-codebook16384``). The oracle's
    ``sae_identity_tokens`` (the local backend's layer / k / codebook / model, which
    live in config rather than in the repo-id string) are folded in first, so a
    local config that selects the *same* SAE as the Forge default is accepted even
    though its repo id is spelled differently. If the oracle does not expose a model
    id (e.g. a bare test double), the check is skipped.
    """
    model_id = getattr(oracle, 'sae_model_id', None)
    if model_id is None:
        return
    normalized = re.sub(r'[^a-z0-9]', '', str(model_id).lower())
    # Keep the two sources separate: concatenating them could synthesize a
    # required token across the join boundary (e.g. a model id ending in 'k6'
    # followed by identity tokens starting with '4' spuriously forming 'k64').
    identity_tokens = getattr(oracle, 'sae_identity_tokens', '') or ''
    if not all(token in normalized or token in identity_tokens for token in _REQUIRED_SAE_MODEL_TOKENS):
        raise ValueError(
            f'SAE energy terms require an oracle backed by the {REQUIRED_SAE_MODEL} model, '
            f'but this oracle uses {model_id!r}. Build the SAE oracle with the default '
            '(ESMC-6B / layer 60) configuration.'
        )


def _prepare_sae_feature_terms(
    feature_indices: list[int],
    coefficients: list[float] | None,
    normalize_coefficients: bool,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Validate ``feature_indices`` / ``coefficients`` shared by the SAE energy terms.

    Returns the validated integer index array and the coefficient array (L1-normalized
    so :math:`\\sum_i |c_i| = 1` when ``normalize_coefficients`` is set). Raises on empty,
    duplicate, or negative indices, mismatched lengths, or all-zero coefficients under
    normalization.
    """
    feature_indices_array = np.asarray(feature_indices, dtype=int)
    if feature_indices_array.ndim != 1 or feature_indices_array.size == 0:
        raise ValueError('feature_indices must be a non-empty 1D list of feature indices.')
    if np.any(feature_indices_array < 0):
        raise ValueError('feature_indices must be non-negative.')
    if np.unique(feature_indices_array).size != feature_indices_array.size:
        raise ValueError('feature_indices must be unique.')

    if coefficients is None:
        coefficients_array = np.ones(feature_indices_array.size, dtype=np.float64)
    else:
        coefficients_array = np.asarray(coefficients, dtype=np.float64)
        if coefficients_array.shape != feature_indices_array.shape:
            raise ValueError(
                f'coefficients length ({coefficients_array.size}) must match '
                f'feature_indices length ({feature_indices_array.size}).'
            )

    if normalize_coefficients:
        l1_norm = float(np.sum(np.abs(coefficients_array)))
        if l1_norm == 0.0:
            raise ValueError('coefficients must have at least one non-zero value to be normalized.')
        coefficients_array = coefficients_array / l1_norm

    return feature_indices_array, coefficients_array


def _relative_residue_sasa(
    structure: AtomArray,
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int_], npt.NDArray[np.str_], npt.NDArray[np.float64]]:
    """
    Returns ``(chain_ids, res_ids, res_names, relative_sasa)`` with one entry per residue.

    ``relative_sasa`` is the residue's total SASA divided by its maximum theoretical SASA (the Tien et al. values in
    :mod:`bagel.constants`), clipped to [0, 1]. 0 means fully buried, 1 means fully exposed. Summing atom SASA rather
    than averaging it is deliberate and matches :class:`HydropathyEnergy`: larger residues naturally expose more
    surface and should contribute proportionally.
    """
    atom_sasa = sasa(structure, probe_radius=probe_radius_water)

    # Unique (chain_id, res_id) pairs in the order they appear in the structure.
    residue_ids = np.empty(
        len(structure),
        dtype=[('chain_id', structure.chain_id.dtype), ('res_id', structure.res_id.dtype)],
    )
    residue_ids['chain_id'] = structure.chain_id
    residue_ids['res_id'] = structure.res_id
    first_index = np.sort(np.unique(residue_ids, return_index=True)[1])

    chain_ids = residue_ids['chain_id'][first_index]
    res_ids = residue_ids['res_id'][first_index]
    res_names = structure.res_name[first_index]

    relative = np.zeros(len(first_index), dtype=float)
    for i, (chain_id, res_id, res_name) in enumerate(zip(chain_ids, res_ids, res_names)):
        atom_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
        total = float(np.sum(atom_sasa[atom_mask]))
        max_sasa = max_theoretical_sasa_for_residues.get(res_name, max_residue_sasa)
        relative[i] = np.clip(total / max_sasa, 0.0, 1.0) if max_sasa > 0 else 0.0

    return chain_ids, res_ids, res_names, relative


class BinderRMSDEnergy(EnergyTerm):
    """
    Root mean square deviation (RMSD) between the structure of a group of residues in this state and in a
    *reference state*. A typical use is to score a binder by how much it must change shape upon binding: the reference
    state holds the binder folded alone and the state this term belongs to holds the binder co-folded with its target.
    A low value means the binder is already pre-organised in its bound conformation, which is expected to lower the
    entropic cost of binding.

    The C-alpha atoms of the selected residues are superimposed, using the rotation and translation that minimise the
    RMSD, and the distance between the two positions of each residue is then measured.

    Optionally, the mean over residues is weighted by :math:`w_i = \\mathrm{pLDDT}_i^n`, so that residues the model is
    confident about in the reference state count for more, and those whose isolated structure is a guess for less.
    The pLDDT is always taken from the reference state. The result is

    .. math:: \\sqrt{\\frac{\\sum_i w_i \\, d_i^2}{\\sum_i w_i}}

    where :math:`d_i` is the C-alpha distance after superposition. pLDDT is on a 0-1 scale. Because the weights are
    normalised, a low pLDDT only removes a residue from the average: it does not lower the energy, so the optimizer
    gains nothing from making the reference fold less confident. (If every pLDDT is 0, the plain RMSD is returned.)

    .. note::
        The reference state is evaluated through its own ``energy`` property, so it must have at least one energy term
        and its results are cached as for any other state. It must be a member of the same
        :class:`~bagel.system.System` as the state this term is in, and contain the **same** :class:`~bagel.chain.Chain`
        objects for the selected residues: the system is deep-copied at every minimization step, and this is what keeps
        the two states in sync when a residue is mutated. It cannot be the state that contains this term.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[Residue],
        reference_state: State,
        reference_oracle: FoldingOracle | None = None,
        plddt_scaled: bool = False,
        plddt_exponent: float = 2.0,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialises the RMSD energy class.

        Parameters
        ----------
        oracle: FoldingOracle
            The oracle used to fold the state this term belongs to.
        residues: list[Residue]
            The residues to compare. All of them must exist in ``reference_state``.
        reference_state: State
            The state whose structure is the reference, e.g. the binder alone. Must contain every residue in
            ``residues``, with the same chain ID, index and amino acid type.
        reference_oracle: FoldingOracle | None, default=None
            The oracle of ``reference_state`` whose structure (and pLDDT) is used. It may differ from ``oracle``, e.g.
            ESMFold for the isolated binder and Boltz2 for the complex. If None, the reference state must use exactly
            one folding oracle, which is then chosen.
        plddt_scaled: bool, default=False
            Whether to weight the contribution of each residue to the mean by its pLDDT in the reference state,
            raised to ``plddt_exponent``.
        plddt_exponent: float, default=2.0
            The exponent n in :math:`\\mathrm{pLDDT}^n`. Must be greater than 0. Only used if ``plddt_scaled``.
        inheritable: bool, default=True
            If a new residue is added next to a residue included in this energy term, this dictates whether that new
            residue could then be added to this energy term.
        weight: float = 1.0
            The weight of the energy term.
        name: str | None = None
            Optional name to append to the energy term name.
        """
        name = 'binder_rmsd' if name is None else f'binder_rmsd_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        if not plddt_exponent > 0:
            raise ValueError(f'plddt_exponent must be greater than 0, got {plddt_exponent}')
        if len(residues) == 0:
            raise ValueError('At least one residue is required to calculate the RMSD')
        from .state import State  # imported here because state.py imports this module

        if not isinstance(reference_state, State):
            raise TypeError(f'reference_state must be a bagel State, got {type(reference_state).__name__}')
        self.reference_state = reference_state
        self.reference_oracle = reference_oracle
        self.plddt_scaled = plddt_scaled
        self.plddt_exponent = plddt_exponent
        self.residue_groups = [residue_list_to_group(residues)]
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be an instance of FoldingOracle'
        assert 'structure' in self.oracle.result_class.model_fields, (
            'BinderRMSDEnergy requires oracle to return structure in result_class'
        )
        if reference_oracle is not None:
            assert isinstance(reference_oracle, FoldingOracle), 'reference_oracle must be an instance of FoldingOracle'
        self._check_residues_in_reference_state(residues)

    def _check_residues_in_reference_state(self, residues: list[Residue]) -> None:
        """Raise if any residue is missing from ``reference_state`` or has a different amino acid type there."""
        reference_chains = {chain.chain_ID: chain for chain in self.reference_state.chains}
        for residue in residues:
            chain = reference_chains.get(residue.chain_ID)
            if chain is None:
                raise ValueError(
                    f"Residue {residue.chain_ID}:{residue.index} is selected, but chain '{residue.chain_ID}' is not in "
                    f"reference state '{self.reference_state.name}' (chains: {sorted(reference_chains)})"
                )
            match = [r for r in chain.residues if r.index == residue.index]
            if len(match) != 1:
                raise ValueError(
                    f'Residue {residue.chain_ID}:{residue.index} is selected, but is not in reference state '
                    f"'{self.reference_state.name}'"
                )
            if match[0].name != residue.name:
                raise ValueError(
                    f'Residue {residue.chain_ID}:{residue.index} is {residue.name} in this state but '
                    f"{match[0].name} in reference state '{self.reference_state.name}'"
                )

    def _get_reference_result(self) -> FoldingResult:
        """Fold the reference state if needed (through its cache) and return the result of the reference oracle."""
        if self in self.reference_state.energy_terms:
            raise ValueError('reference_state cannot be the state that contains this energy term')
        _ = self.reference_state.energy  # runs the oracles of the reference state, or reuses its cache
        oracle = self.reference_oracle
        if oracle is None:
            candidates = [o for o in self.reference_state.oracles_list if isinstance(o, FoldingOracle)]
            if len(candidates) != 1:
                raise ValueError(
                    f"Reference state '{self.reference_state.name}' has {len(candidates)} folding oracles; "
                    'specify which one to use with reference_oracle'
                )
            oracle = candidates[0]
        elif oracle not in self.reference_state.oracles_list:
            raise ValueError(
                f"reference_oracle is not used by any energy term of reference state '{self.reference_state.name}', "
                'so it is never run'
            )
        result = self.reference_state._oracles_result[oracle]
        assert isinstance(result, FoldingResult), 'Result must be a FoldingResult'
        return result

    @staticmethod
    def _ca_indices(structure: AtomArray, chain_ids: npt.NDArray[np.str_], res_ids: npt.NDArray[np.int_]) -> list[int]:
        """Indices in ``structure`` of the C-alpha of each requested residue, in the requested order."""
        ca_position = {
            (str(c), int(r)): i
            for i, (c, r, a) in enumerate(zip(structure.chain_id, structure.res_id, structure.atom_name))
            if a == 'CA'
        }
        missing = [(c, r) for c, r in zip(chain_ids, res_ids) if (str(c), int(r)) not in ca_position]
        if missing:
            raise ValueError(f'No C-alpha found in the structure for residues (chain, index): {missing}')
        return [ca_position[(str(c), int(r))] for c, r in zip(chain_ids, res_ids)]

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        chain_ids, res_ids = self.residue_groups[0]
        if len(res_ids) == 0:
            return 0.0, 0.0
        # residues can be added or removed during a grand canonical run, so check again that the reference state
        # still has every selected residue
        reference_chains = {chain.chain_ID: {r.index for r in chain.residues} for chain in self.reference_state.chains}
        absent = [(c, i) for c, i in zip(chain_ids, res_ids) if int(i) not in reference_chains.get(str(c), set())]
        if absent:
            raise ValueError(
                f"Residues (chain, index) {absent} are not in reference state '{self.reference_state.name}'"
            )
        reference_result = self._get_reference_result()

        structure = oracles_result.get_structure(self.oracle)
        reference = reference_result.structure
        state_ca = structure[self._ca_indices(structure, chain_ids, res_ids)]
        reference_ca = reference[self._ca_indices(reference, chain_ids, res_ids)]

        reference_ca = superimpose(fixed=state_ca, mobile=reference_ca)[0]  # translation and rotation fit
        deviations = np.linalg.norm(state_ca.coord - reference_ca.coord, axis=1)

        weights = np.ones_like(deviations)
        if self.plddt_scaled:
            assert hasattr(reference_result, 'local_plddt'), 'local_plddt metric not returned by folding algorithm'
            assert reference_result.local_plddt.shape[0] == 1, 'batch size equal to 1 is required'
            plddt = reference_result.local_plddt[0]
            # pLDDT is stored per residue, in the order chains and residues appear in the reference structure
            plddt_position: dict[tuple[str, int], int] = {}
            for chain in pd.unique(reference.chain_id):
                for res in pd.unique(reference.res_id[reference.chain_id == chain]):
                    plddt_position[(str(chain), int(res))] = len(plddt_position)
            selected_plddt = np.array([plddt[plddt_position[(str(c), int(r))]] for c, r in zip(chain_ids, res_ids)])
            scaled_weights = selected_plddt**self.plddt_exponent
            if scaled_weights.sum() > 0:  # otherwise keep the plain RMSD
                weights = scaled_weights

        value = float(np.sqrt(np.sum(weights * deviations**2) / np.sum(weights)))
        return value, value * self.weight
