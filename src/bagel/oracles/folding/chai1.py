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
from boileroom.models.chai.chai1 import Chai1Output  # type: ignore
from boileroom.models.chai.chai1 import Chai1 as Chai1Boiler

from biotite.structure import AtomArray


class Chai1Result(FoldingResult):
    """
    Stores statistics from the Chai-1 folding algorithm.
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


class Chai1(FoldingOracle):
    """
    Object that uses Chai-1 to predict structure of proteins from sequence.
    """

    result_class: Type[Chai1Result] = Chai1Result

    def __init__(self, backend: str = "modal", device: str | None = None, config: dict[str, Any] = {}):
        """
        Initialize Chai1 oracle.

        Parameters
        ----------
        backend : str
            Backend to use. Supported values: "modal", "apptainer"
        device : str | None
            Device to use (e.g., "cuda:0", "cuda:1").
        config : dict[str, Any]
            Configuration dictionary passed to the model
        """
        self.backend = backend
        self.device = device
        self.default_config = {}
        # Always request these fields in the output
        self.required_fields = ['plddt', 'pae', 'ptm']
        self._load(config)

    def _load(self, config: dict[str, Any] = {}) -> None:
        config = {**self.default_config, **config}
        self.model = Chai1Boiler(backend=self.backend, device=self.device, config=config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for folding.
        Individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def fold(self, chains: List[Chain]) -> Chai1Result:
        """
        Fold a list of chains using Chai-1.
        """
        sequences = self._pre_process(chains)
        options = {'include_fields': self.required_fields}
        output = self.model.fold(sequences, options=options)
        return self._reduce_output(output, chains)

    def _reduce_output(self, output: Chai1Output, chains: List[Chain]) -> Chai1Result:
        """
        Reduce Chai1Output (from boileroom.chai1) to a Chai1Result object.
        """
        if output.atom_array is None or len(output.atom_array) == 0:
            raise ValueError("Chai1 output does not contain atom_array")
        
        atoms = output.atom_array
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])
        
        # These fields should always be present since we requested them via include_fields
        if output.plddt is None or len(output.plddt) == 0:
            raise ValueError("Chai1 output does not contain plddt (requested via include_fields)")
        if output.pae is None or len(output.pae) == 0:
            raise ValueError("Chai1 output does not contain pae (requested via include_fields)")
        if output.ptm is None or len(output.ptm) == 0:
            raise ValueError("Chai1 output does not contain ptm (requested via include_fields)")
        
        # Extract plddt (Chai1 plddt is per-residue, 1D array)
        plddt_data = output.plddt[0]
        # Normalize from 0-100 to 0-1 if needed
        if np.max(plddt_data) > 1.0:
            plddt_data = plddt_data / 100.0
        local_plddt = plddt_data[None, :]
        
        # Extract pae
        pae = output.pae[0][None, :, :]
        
        # Extract ptm
        ptm_value = output.ptm[0]
        # ptm might be a scalar or array
        if np.isscalar(ptm_value):
            ptm = np.array([ptm_value])[None, :]
        else:
            ptm = np.array(ptm_value)[None, :]
        
        results = self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=local_plddt,
            ptm=ptm,
            pae=pae,
        )
        return results

