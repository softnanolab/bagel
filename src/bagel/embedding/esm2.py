"""
standard template and objects for structure prediction
"""

import numpy as np
import numpy.typing as npt
from ..chain import Chain
from .utils import reindex_chains
from pydantic import field_validator
from .base import ProteinLanguageModel 
from typing import List, Any
from modalfold import app  # type: ignore
from modalfold.esmfold import ESMFold, ESMFoldOutput  # type: ignore


import logging

logger = logging.getLogger(__name__)

class ESM2Output(BaseModel):
    """
    Stores statistics from the protein language model.
    """

    #TODO: add fields and validators for the output of ESM2
    pass

class Config:
        arbitrary_types_allowed = True  # This is needed for numpy array support

class ESM2(ProteinLanguageModel):

    def __init__(self, name: str, esm2_name : str) -> None:
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        Initialise the ESM2 model.
        WIP: For now we will be using ModalFold to do this reliably without much env issues.
        """
    def __init__(self, use_modal: bool = False, config: dict[str, Any] = {}):
        self.use_modal = use_modal
        self.default_config = {
            'output_pdb': False,
            'output_cif': False,
            'output_atomarray': True,
            'glycine_linker': '',
            'position_ids_skip': 512,
        }
        self._load(config)

        if self.use_modal:
            # Register the cleanup function to be called at exit, so no
            # ephermal app is left running when the object is destroyed
            import atexit

            atexit.register(self.__del__)










        assert esm2_name in ['esm2_t33_650M_UR50S', 'esm2_t48_150M_UR50S'], f"Model {name} not supported"
        super().__init__(name)
        self.model = self._load(esm2_name)
    
    #FROM ESMFold
    def __del__(self) -> None:
        """Cleanup the app context when the object is destroyed or at exit"""
        if self.use_modal and hasattr(self, 'modal_app_context') and self.modal_app_context is not None:  # type: ignore
            self.modal_app_context.__exit__(None, None, None)  # type: ignore
            self.modal_app_context = None

    #FROM ESMFold
    def _load(self, config: dict[str, Any] = {}) -> None:
        if self.use_modal:
            self.modal_app_context = app.run()
            self.modal_app_context.__enter__()  # type: ignore
        config = {**self.default_config, **config}
        self.model = ESMFold(config)


    #Merge with above
    def _load_model(self, esm2_name: str) -> nn.Module:
        """
        Load the ESM2 model.
        """
        #TODO: make compatible with ESM2 in HuggingFace
        from esm import ProteinBertModel, pretrained
        model, alphabet = pretrained.load_model_and_alphabet(esm2_name)
        model.eval()  # Set the model to evaluation mode
        return model

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for calculating the embeddings.
        Here, we assume, that we are using HuggingFace's implementation of ESM2
        Therefore, individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    #Harmonize with fold below 
    def calculate_embeddings(self, state) -> None:
        """
        Calculate the embeddings of the residues in the state.
        """
        chains = self._pre_process(state.chains)
        # TODO: make compatible with whatever is the actual output in ESM2 
        embeddings = self.model( chains )[ "last_hidden_state" ]
        return embeddings


    def fold(self, chains: List[Chain]) -> tuple[AtomArray, ESMFoldingMetrics]:
        """
        Fold a list of chains using ESMFold.
        """
        if self.use_modal:
            return self._reduce_output(self._remote_fold(self._pre_process(chains)), chains)
        else:
            logger.info('Given that use_modal is False, trying to fold with ESMFold locally...')
            logger.info('Assuming that all packages are available locally...')
            # TODO: Hugging Face Cache might need to be set here properly to make it work
            return self._reduce_output(self._local_fold(self._pre_process(chains)), chains)

    def _remote_fold(self, sequence: List[str]) -> ESMFoldOutput:
        return self.model.fold.remote(sequence)

    def _local_fold(self, sequence: List[str]) -> ESMFoldOutput:
        return self.model.fold.local(sequence)

    def _reduce_output(self, output: ESMFoldOutput, chains: List[Chain]) -> tuple[AtomArray, ESMFoldingMetrics]:
        """
        Reduce ESMFoldOutput (from ModalFold) to a ESMFoldingMetrics object
        """
        atoms = output.atom_array
        metrics = ESMFoldingMetrics(
            local_plddt=output.plddt,
            ptm=output.ptm,
            pae=output.predicted_aligned_error,
        )
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])

        return atoms, metrics
