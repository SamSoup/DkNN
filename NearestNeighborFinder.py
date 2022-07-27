from typing import List, Dict, Tuple
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
from utils import get_layer_representations
import numpy as np
import torch
import abc

# TODO: add Doc-string and comments where need be

class AbstractNearestNeighbor(abc.ABC):
    """   
    Interface for K Nearest Neighbors Search, whereby the only functionality
    required is that given a batch of tensors, return the k nearest neighbor per 
    layer for the examples in the batch
    """
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.array], layer_dim: int):
        """
        Initialize basic parameters that all KNN finders will need

        Args:
            k (int): number of nearest neighbors to obtain per sample per layer
            layers_to_save (List[int]): the list of layers to save
            database (Dict[int, np.array]): the database of example representations per layer
            layer_dim (int): the hidden size of each layer
        """

        self.K = K
        self.layers_to_save = layers_to_save
        self.database = database
        self.layer_dim = layer_dim

    @abc.abstractmethod
    def nearest_neighbors(self, hidden_states: Tuple[torch.tensor]) -> np.ndarray:
        """
        Given a batch of input example tensors, return the k nearest neighbor per layer
        Original reference: Andoni et. al. https://arxiv.org/pdf/1509.02897.pdf

        Args:
            hidden_states (Tuple[torch.tensor]): (num_layers+1, batch_size, max_seq_len, embedding_dim)

        Returns:
            np.ndarray: the collection of neighbors (col) per example (row)
        """

        raise NotImplementedError
    
class KDTreeNearestNeighbor(AbstractNearestNeighbor):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.array], layer_dim: int):
        super().__init__(K, layers_to_save, database, layer_dim)
        self.trees = {
            layer: KDTree(self.database[layer][:, :self.layer_dim], metric="l2") 
                for layer in self.layers_to_save
        }

    def nearest_neighbors(self, hidden_states: Tuple[torch.tensor]) -> np.ndarray:
        # print("***** Running DkNN - Nearest Neighbor Search (One Batch) KD Tree *****")
        # for each example in the batch, find their total list of neighbors
        # the batch size may be not equivalent to self.args.eval_batch_size, if the 
        # number of training examples is *not* an exact multiple of self.args.eval_batch_size
        # so we use the first layer hidden state's first dimension (presumed batch size)
        neighbors = np.zeros((hidden_states[0].shape[0], len(self.layers_to_save) * self.K))
        # progress_bar = tqdm(range(len(self.layers_to_save)))
        for l, layer in enumerate(self.layers_to_save):
            tree = self.trees[layer]
            dist, batch_indices = tree.query(get_layer_representations(hidden_states[layer]), k=self.K) # (batch_size, K)
            # based on idx, look up tag in database per batch of example
            for i, neighbor_indices in enumerate(batch_indices):
                labels = self.database[layer][neighbor_indices][:, -1] # label is the last element of the 2d np array
                neighbors[i, l * self.K : (l+1) * self.K] = labels
            # progress_bar.update(1)
        return neighbors

class LocalitySensitiveHashingNearestNeighbor(AbstractNearestNeighbor):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.array], layer_dim: int):
        super().__init__(K, layers_to_save, database, layer_dim)
        # TODO

    def nearest_neighbors(self, hidden_states: Tuple[torch.tensor]) -> np.array:
        print("***** Running DkNN - Nearest Neighbor Search (One Batch) LSH *****")

class AbstractNearestNeighborFactory(abc.ABC):
    """
    Abstract Factory to produce different Nearest Neighbor getters
    """
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.array], layer_dim: int):
        self.K = K
        self.layers_to_save = layers_to_save
        self.database = database
        self.layer_dim = layer_dim

    @abc.abstractmethod
    def createNearestNeighborFunction(self) -> AbstractNearestNeighbor:
        raise NotImplementedError

class KDTreeNearestNeighborFactory(AbstractNearestNeighborFactory):
    def createNearestNeighborFunction(self) -> AbstractNearestNeighbor:
        return KDTreeNearestNeighbor(
            K=self.K, 
            layers_to_save=self.layers_to_save, 
            database=self.database, 
            layer_dim=self.layer_dim
        )

class LocalitySensitiveHashingNearestNeighborFactory(AbstractNearestNeighborFactory):
    def createNearestNeighborFunction(self) -> AbstractNearestNeighbor:
        return LocalitySensitiveHashingNearestNeighbor(            
            K=self.K, 
            layers_to_save=self.layers_to_save, 
            database=self.database, 
            layer_dim=self.layer_dim
        )
