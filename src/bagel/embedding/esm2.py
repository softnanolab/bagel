"""
standard template and objects for structure prediction
"""

import numpy as np
import numpy.typing as npt
from ..chain import Chain
#from .utils import reindex_chains
from pydantic import field_validator
from .base import ProteinLanguageModel 
from typing import List, Any
from modalfold import app  
from modalfold.esm2 import ESM2, ESM2Output 


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

    def __init__(self, name: str, esm2_name : str, use_modal:bool = False, config:dict[str,Any] ={} ) -> None:
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        Initialise the ESM2 model.
        WIP: For now we will be using ModalFold to do this reliably without much env issues.
        """
        self.use_modal = use_modal
        self.default_config = {
            'output_embeddings': True,
            'output_all_layers': False,
            'esm2_name': 'esm2_t33_650M_UR50S',
        }
        self._load(config)

        if self.use_modal:
            # Register the cleanup function to be called at exit, so no
            # ephermal app is left running when the object is destroyed
            import atexit
            atexit.register(self.__del__)

        models_name = [ "esm2_t48_15B_UR50D", "esm2_t36_3B_UR50D", "esm2_t33_650M_UR50D", 
                        "esm2_t30_150M_UR50D", "esm2_t12_35M_UR50D", "esm2_t6_8M_UR50D", ]	

        assert config[ esm2_name ] in models_name, f"Model {esm2_name} not supported"

        self.name = name
        #! @JAKUB: This will require implementing ESM2 in modalfold.
        self.model = self._load(config[ esm2_name ])
    
    #FROM ESMFold
    def __del__(self) -> None:
        """Cleanup the app context when the object is destroyed or at exit"""
        if self.use_modal and hasattr(self, 'modal_app_context') and self.modal_app_context is not None:  # type: ignore
            self.modal_app_context.__exit__(None, None, None)  # type: ignore
            self.modal_app_context = None

    # Need to code ESM2 in modal
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
        #! @JAKUB: Assuming here this is the ESM2 convention / this is how it works in 
        #! its ModalFold implementation.
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def calculate_embeddings(self, state) -> Tensor:
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

    def _post_process(self, output: ESM2Output) -> Tensor:
        """
        Reduce ESM2Output (from ModalFold) to a Tensor of size batch x N_residues x N_features
        containing the embeddings only.
        """
        #! @JAKUB: Not sure this is correct, I think this only works if N_batch = 1, I made an assertion for that 
        assert output["last_hidden_state"].shape[0] == 1, f"Return next only works correctly for batch of size 1, got {len(output['last_hidden_state'].shape)}D tensor"
        
        return output["last_hidden_state"].reshape(-1, output["last_hidden_state"].shape[-1]) 