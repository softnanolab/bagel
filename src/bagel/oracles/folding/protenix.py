"""
Protenix folding oracle for BAGEL.

Protenix is an AlphaFold 3 reproduction that natively supports multi-chain
protein complexes. Unlike ESMFold, it does NOT require:
  - Glycine linkers between chains
  - Positional encoding skips

Protenix outputs all necessary confidence metrics (pLDDT, PAE, pTM, ipTM)
required by BAGEL's energy terms.
"""

import os
import logging
import pathlib as pl
from typing import Any, List, Optional, Type

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from biotite.structure import AtomArray
from modal import App

from ...chain import Chain
from ...constants import atom_order
from .base import FoldingOracle, FoldingResult
from .utils import reindex_chains

# NOTE: boileroom imports are deferred to avoid triggering the @app.cls
# decorator at import time, which can cause Modal deprecation errors and
# prevent the entire bagel.oracles module from loading.
# Import ProtenixFoldBoiler and ProtenixOutput inside methods that need them.

logger = logging.getLogger(__name__)


def validate_array_range(
    array: npt.NDArray[np.float64], field_name: str, min_val: float = 0, max_val: float = 1
) -> npt.NDArray[np.float64]:
    """Validate that array values fall within a specified range."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f'{field_name} must be a numpy array')
    if not np.all((array >= min_val) & (array <= max_val)):
        raise ValueError(f'All values in {field_name} must be between {min_val} and {max_val}')
    return array


class ProtenixFoldResult(FoldingResult):
    """
    Stores results from the Protenix folding algorithm.

    Attributes
    ----------
    input_chains : list[Chain]
        The input chains that were folded.
    structure : AtomArray
        The predicted 3D structure as a biotite AtomArray.
    local_plddt : np.ndarray
        Per-residue pLDDT scores (0 to 1), shape [1, n_residues].
        Extracted from per-atom pLDDT by taking the CA atom value for each residue.
    ptm : np.ndarray
        Predicted TM-score (0 to 1), shape [1].
    pae : np.ndarray
        Predicted Aligned Error matrix, shape [1, n_tokens, n_tokens].
        Values represent the expected position error in Angstroms.
    iptm : np.ndarray
        Interface predicted TM-score (0 to 1), shape [1].
        Particularly useful for multi-chain complexes.
    """

    input_chains: list[Chain]
    structure: AtomArray
    local_plddt: npt.NDArray[np.float64]  # [1, n_residues] per-residue pLDDT (0-1)
    ptm: npt.NDArray[np.float64]  # [1] predicted TM-score (0-1)
    pae: npt.NDArray[np.float64]  # [1, n_tokens, n_tokens] predicted aligned error
    iptm: npt.NDArray[np.float64]  # [1] interface predicted TM-score (0-1)

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'local_plddt', 0, 1)

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'ptm', 0, 1)

    @field_validator('iptm')
    def validate_iptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'iptm', 0, 1)

    def save_attributes(self, filepath: pl.Path) -> None:
        """Save pLDDT, PAE, and ipTM to files."""
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')
        np.savetxt(filepath.with_suffix('.iptm'), self.iptm, fmt='%.6f', header='iptm')


class ProtenixOracle(FoldingOracle):
    """
    Folding oracle that uses Protenix to predict protein structures.

    Protenix natively handles multi-chain protein complexes, so unlike the
    ESMFold oracle, this oracle does NOT:
      - Concatenate chains with ":" separators
      - Insert glycine linkers between chains
      - Apply positional encoding skips

    Instead, each chain is passed as a separate entity to Protenix, which
    models inter-chain interactions directly.

    Parameters
    ----------
    use_modal : bool
        Whether to run on Modal (remote GPU). Default False.
    config : dict
        Configuration for the Protenix model. See ProtenixFold in boileroom
        for supported keys. Common options:

        - ``model_name`` (str): Model checkpoint name. Default "protenix_base_default_v1.0.0".
        - ``n_sample`` (int): Number of structure samples. Default 1.
        - ``n_cycle`` (int): Number of recycling cycles. Default 10.
        - ``n_step`` (int): Number of diffusion steps. Default 200.
        - ``use_msa`` (bool): Whether to use MSA features. Default False.
        - ``seed`` (int): Random seed. Default 101.

    modal_app_context : App or None
        An existing Modal app context to reuse. If None and use_modal is True,
        a new context will be created.

    Examples
    --------
    >>> import bagel as bg
    >>> # Create oracle
    >>> protenix = bg.oracles.ProtenixOracle(use_modal=True)
    >>> # Use in energy terms
    >>> energy = bg.energies.PTMEnergy(oracle=protenix, weight=1.0)
    >>> pae_energy = bg.energies.PAEEnergy(oracle=protenix, residues=[group1, group2], weight=5.0)
    """

    result_class: Type[ProtenixFoldResult] = ProtenixFoldResult

    def __init__(
        self,
        use_modal: bool = False,
        config: dict[str, Any] = {},
        modal_app_context: App | None = None,
    ):
        self.use_modal = use_modal
        self.modal_app_context = modal_app_context

        # Default config -- note: NO glycine_linker or position_ids_skip
        # because Protenix handles multi-chain natively
        self.default_config = {
            'output_pdb': False,
            'output_cif': False,
            'output_atomarray': True,
            'model_name': 'protenix_base_default_v1.0.0',
            'n_sample': 1,
            'n_cycle': 10,
            'n_step': 200,
            'use_msa': False,
            'use_template': False,
            'seed': 101,
            'dtype': 'bf16',
        }
        self._load(config)

        if self.use_modal and self.modal_app_context is None:
            import atexit
            atexit.register(self.__del__)

    def __del__(self) -> None:
        """Cleanup the app context when the object is destroyed."""
        if self.use_modal and hasattr(self, 'modal_app_context') and self.modal_app_context is not None:
            self.modal_app_context.__exit__(None, None, None)
            self.modal_app_context = None

    def _load(self, config: dict[str, Any] = {}) -> None:
        """Load the Protenix model via boileroom."""
        logger.info("[ProtenixOracle] Importing boileroom ...")
        from boileroom import app as boileroom_app  # type: ignore
        from boileroom.models.protenix.protenix import ProtenixFold as ProtenixFoldBoiler  # type: ignore
        logger.info("[ProtenixOracle] Imports done.")

        if self.use_modal and self.modal_app_context is None:
            logger.info("[ProtenixOracle] Starting Modal app context ...")
            self.modal_app_context = boileroom_app.run()
            self.modal_app_context.__enter__()
            logger.info("[ProtenixOracle] Modal app context ready.")
        config = {**self.default_config, **config}
        logger.info("[ProtenixOracle] Creating ProtenixFold instance ...")
        self.model = ProtenixFoldBoiler(config)
        logger.info("[ProtenixOracle] Ready.")

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process chains for Protenix.

        Unlike ESMFold, we pass each chain as a separate sequence.
        Protenix handles multi-chain complexes natively -- no linkers
        or positional encoding hacks are needed.

        Parameters
        ----------
        chains : list[Chain]
            List of Chain objects to fold.

        Returns
        -------
        list[str]
            List of sequence strings, one per chain.
        """
        return [chain.sequence for chain in chains]

    def fold(self, chains: List[Chain]) -> ProtenixFoldResult:
        """
        Fold a list of chains using Protenix.

        Each chain is passed as a separate entity to Protenix, which
        handles multi-chain complexes natively without linker hacks.

        Parameters
        ----------
        chains : List[Chain]
            List of Chain objects to predict structure for.

        Returns
        -------
        ProtenixFoldResult
            Prediction results containing structure, pLDDT, PAE, pTM, and ipTM.
        """
        sequences = self._pre_process(chains)

        if self.use_modal:
            output = self._remote_fold(sequences)
        else:
            logger.debug('Folding with Protenix locally...')
            output = self._local_fold(sequences)

        return self._reduce_output(output, chains)

    def _remote_fold(self, sequences: List[str]) -> Any:
        """Run Protenix on Modal (remote GPU)."""
        return self.model.fold.remote(sequences)

    def _local_fold(self, sequences: List[str]) -> Any:
        """Run Protenix locally."""
        return self.model.fold.local(sequences)

    def _reduce_output(self, output: Any, chains: List[Chain]) -> ProtenixFoldResult:
        """
        Reduce ProtenixOutput to a ProtenixFoldResult for use by BAGEL energy terms.

        Extracts and formats the key metrics:
        - structure: AtomArray with correct chain IDs
        - local_plddt: per-residue pLDDT (from per-atom CA values)
        - ptm: predicted TM-score
        - pae: predicted aligned error matrix
        - iptm: interface predicted TM-score

        Parameters
        ----------
        output : ProtenixOutput
            Raw output from the boileroom Protenix wrapper.
        chains : List[Chain]
            Original chain objects (used for chain ID mapping).

        Returns
        -------
        ProtenixFoldResult
            Formatted results for BAGEL energy terms.
        """
        # Get atom arrays and reindex chains to match BAGEL chain IDs
        # output.atom_array is a list of AtomArrays (one per sample).
        # reindex_chains expects a list with exactly one AtomArray, so
        # we select only the best sample (index 0).
        atoms = output.atom_array
        if isinstance(atoms, list) and len(atoms) > 1:
            atoms = [atoms[0]]  # Use only the first/best sample
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])

        # Extract per-residue pLDDT from per-atom pLDDT
        # Protenix returns per-atom pLDDT; we need per-residue (like ESMFold's CA pLDDT)
        # We take the mean pLDDT of atoms belonging to each token/residue
        plddt_per_atom = output.plddt  # [N_sample, N_atom]
        token_map = output.token_to_atom_map  # [N_atom] maps atom -> token

        # Compute per-residue pLDDT by averaging atom pLDDT per token
        n_sample = plddt_per_atom.shape[0]
        n_tokens = output.pae.shape[-1]  # Number of tokens from PAE matrix shape
        residue_plddt = np.zeros((n_sample, n_tokens))

        for sample_idx in range(n_sample):
            for token_idx in range(n_tokens):
                atom_mask = token_map == token_idx
                if np.any(atom_mask):
                    residue_plddt[sample_idx, token_idx] = np.mean(
                        plddt_per_atom[sample_idx, atom_mask]
                    )

        # Format ptm and iptm to match expected shapes [1] (taking best sample)
        # For BAGEL, we use the first sample (index 0)
        best_sample = 0
        local_plddt = residue_plddt[best_sample:best_sample + 1]  # [1, n_residues]
        ptm = output.ptm[best_sample:best_sample + 1]  # [1]
        iptm = output.iptm[best_sample:best_sample + 1]  # [1]
        pae = output.pae[best_sample:best_sample + 1]  # [1, n_tokens, n_tokens]

        results = self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=local_plddt,
            ptm=ptm,
            pae=pae,
            iptm=iptm,
        )
        return results
