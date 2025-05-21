"""
standard template and objects for structure prediction
"""

import numpy as np
from ..chain import Chain
from .base import EmbeddingResults, EmbeddingOracle
from typing import List, Any
from modalfold import app  
from modalfold.esm2 import ESM2, ESM2Output     

import logging

logger = logging.getLogger(__name__)

class ESM2Results(EmbeddingResults):
    """
    Stores statistics from the protein language model.
    """

    #TODO: add fields and validators for the output of ESM2
    pass

class Config:
        arbitrary_types_allowed = True  # This is needed for numpy array support

class ESM2(EmbeddingOracle):

    def __init__(self, use_modal:bool = False, config:dict[str,Any] ={} ) -> None:
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        Initialise the ESM2 model.
        WIP: For now we will be using ModalFold to do this reliably without much env issues.
        """
        self.use_modal = use_modal
        self.default_config = {
            'output_hidden_states': False,
            'model_name': 'esm2_t33_650M_UR50D',
        }
        if self.use_modal:
            # Register the cleanup function to be called at exit, so no
            # ephermal app is left running when the object is destroyed
            import atexit
            atexit.register(self.__del__)

        self.model = self._load(config)
    
    def __del__(self) -> None:
        """Cleanup the app context when the object is destroyed or at exit"""
        if self.use_modal and hasattr(self, 'modal_app_context') and self.modal_app_context is not None:  # type: ignore
            self.modal_app_context.__exit__(None, None, None)  # type: ignore
            self.modal_app_context = None

    def _load(self, config: dict[str, Any] = {}) -> None:
        if self.use_modal:
            self.modal_app_context = app.run()
            self.modal_app_context.__enter__()  # type: ignore
        config = {**self.default_config, **config}
        return ESM2(config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for calculating the embeddings.
        Here, we assume, that we are using HuggingFace's implementation of ESM2
        Therefore, individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def calculate_embeddings(self, state) -> np.ndarray:
        """
        Calculate the embeddings of the residues in the state.
        """
        chains = self._pre_process(state.chains)
        
        #! @JAKUB: This requires implementing ESM2 in ModalFold
        if self.use_modal:
            return self._post_process(self._remote_embeddings(self._pre_process(state.chains)))
        else:
            logger.info('Given that use_modal is False, trying to fold with ESMFold locally...')
            logger.info('Assuming that all packages are available locally...')
            # TODO: Hugging Face Cache might need to be set here properly to make it work
            return self._post_process(self._local_embeddings(self._pre_process(state.chains)))

    def _remote_embeddings(self, sequence: List[str]) -> ESM2Output:
        return self.model.embeddings.remote(sequence)

    def _local_embeddings(self, sequence: List[str]) -> ESM2Output:
        return self.model.embeddings.local(sequence)

    def _post_process(self, output: ESM2Output) -> np.ndarray:
        """
        Reduce ESM2Output (from ModalFold) to a Tensor of size batch x N_residues x N_features
        containing the embeddings only.
        """
        #! @JAKUB: Not sure this is correct, I think this only works if N_batch = 1, I made an assertion for that 
        assert output["last_hidden_state"].shape[0] == 1, f"Return next only works correctly for batch of size 1, got {len(output['last_hidden_state'].shape)}D tensor"

        #Return the embeddings as a 2D tensor of size N_residues x N_features
        embedding = output["last_hidden_state"].reshape(-1, output["last_hidden_state"].shape[-1])
        #Remove first and last since these are not residues but embedding of the start and end of the sequence
        embedding = embedding[1:-1]
        #Make it into a numpy array 
        embedding.detach().numpy()
        return embedding