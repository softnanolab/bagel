"""
standard template and objects for structure prediction
"""

import pathlib as pl
import numpy as np
import numpy.typing as npt
from ...chain import Chain
from ...constants import atom_order
from .utils import reindex_chains, validate_array_range
from pydantic import field_validator
from .base import FoldingOracle, FoldingResult
from typing import List, Any, Type
from boileroom.models.boltz.boltz2 import Boltz2Output  # type: ignore
from boileroom.models.boltz.boltz2 import Boltz2 as Boltz2Boiler

from biotite.structure import AtomArray


class Boltz2Result(FoldingResult):
    """
    Stores statistics from the Boltz-2 folding algorithm.
    """

    input_chains: list[Chain]
    structure: AtomArray  # structure of the predicted model
    local_plddt: npt.NDArray[np.float64]  # local ( per residue ) predicted LDDT score (0 to 1)
    ptm: npt.NDArray[np.float64]  # (global) predicted template modelling score (0 to 1)
    pae: npt.NDArray[np.float64]  # pairwise predicted alignment error

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'local_plddt', 0, 1)

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'ptm', 0, 1)

    def save_attributes(self, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')


class Boltz2(FoldingOracle):
    """
    Object that uses Boltz-2 to predict structure of proteins from sequence.
    """

    result_class: Type[Boltz2Result] = Boltz2Result

    def __init__(self, backend: str = "modal", config: dict[str, Any] = {}):
        """
        Initialize Boltz2 oracle.

        Parameters
        ----------
        backend : str
            Backend to use. Only "modal" is supported.
        config : dict[str, Any]
            Configuration dictionary passed to the model
        """
        assert backend == "modal", "Boltz2 requires modal backend"
        self.backend = backend
        self.default_config = {}
        # Always request these fields in the output
        self.required_fields = ['plddt', 'pae']
        self._load(config)

    def _load(self, config: dict[str, Any] = {}) -> None:
        config = {**self.default_config, **config}
        self.model = Boltz2Boiler(backend=self.backend, config=config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for folding.
        Individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def fold(self, chains: List[Chain]) -> Boltz2Result:
        """
        Fold a list of chains using Boltz-2.
        """
        sequences = self._pre_process(chains)
        options = {'include_fields': self.required_fields}
        output = self.model.fold(sequences, options=options)
        return self._reduce_output(output, chains)

    def _reduce_output(self, output: Boltz2Output, chains: List[Chain]) -> Boltz2Result:
        """
        Reduce Boltz2Output (from boileroom.boltz2) to a Boltz2Result object.
        """
        if output.atom_array is None or len(output.atom_array) == 0:
            raise ValueError("Boltz2 output does not contain atom_array")
        
        atoms = output.atom_array
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])
        
        # These fields should always be present since we requested them via include_fields
        if output.plddt is None or len(output.plddt) == 0:
            raise ValueError("Boltz2 output does not contain plddt (requested via include_fields)")
        if output.pae is None or len(output.pae) == 0:
            raise ValueError("Boltz2 output does not contain pae (requested via include_fields)")
        
        # Extract plddt (Boltz2 plddt is per-residue, 1D array)
        plddt_data = output.plddt[0]
        # Normalize from 0-100 to 0-1 if needed
        if len(plddt_data.shape) == 1:
            if np.max(plddt_data) > 1.0:
                plddt_data = plddt_data / 100.0
            local_plddt = plddt_data[None, :]
        else:
            # If somehow 2D, extract CA atoms
            local_plddt = plddt_data[..., atom_order['CA']][None, :]
        
        # Extract pae
        pae = output.pae[0][None, :, :]
        
        # Boltz2 may not have ptm, so we set it to a default value
        ptm = np.array([0.0])[None, :]
        
        results = self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=local_plddt,
            ptm=ptm,
            pae=pae,
        )
        return results

