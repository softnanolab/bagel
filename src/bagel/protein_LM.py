"""
Standard object to encode the tertiary structure, losses, and folding logic for a chain or complex of chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import dataclass, abstractmethod
import logging
from torch import nn, Tensor
from typing import Optional
from .chain import Chain

logger = logging.getLogger(__name__)

@dataclass
class proteinLanguageModel:
    """
    An Embedding is a N_residues x N_features matrix that contains the embeddings of the residues in the state.
    """
    name: str
    model : nn

    @abstractmethod
    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for calculating the embeddings.
        """
        pass

    @abstractmethod 
    def calculate_embeddings(self, state) -> None:
        """
        Calculate the embeddings of the residues in the state.
        """
        pass

class ESM2(proteinLanguageModel):

    def __init__(self, name: str, esm2_name : str) -> None:
        """
        Initialise the ESM2 model.
        """
        assert esm2_name in ['esm2_t33_650M_UR50S', 'esm2_t48_150M_UR50S'], f"Model {name} not supported"
        super().__init__(name)
        self.model = self._load_model(esm2_name)

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
    
    def calculate_embeddings(self, state) -> None:
        """
        Calculate the embeddings of the residues in the state.
        """
        chains = self._pre_process(state.chains)
        # TODO: make compatible with whatever is the actual output in ESM2 
        embeddings = self.model( chains )[ "last_hidden_state" ]
        return embeddings
