from collections import defaultdict
from typing import List, Dict, Tuple, Union, Callable
from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import pairwise_distances
from tqdm.auto import tqdm
from itertools import combinations
import numpy as np
import torch
import abc
import time

class AbstractNearestNeighbor(abc.ABC):
    """
    Interface for K Nearest Neighbors Search, whereby the only functionality
    required is that given a batch of tensors, return the k nearest neighbor per 
    layer for the examples in the batch
    """
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], layer_dim: int):
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
    def nearest_neighbors(self, layer_reps: Dict[int, torch.tensor], output_neighbor_id: bool=False) -> np.ndarray:
        """
        Given a batch of input example tensors, return the k nearest neighbor per layer
        Original reference: Andoni et. al. https://arxiv.org/pdf/1509.02897.pdf

        Args:
            layer_reps (Tuple[torch.tensor]): (num_layers+1, batch_size, embedding_dim)

        Returns:
            np.ndarray: the collection of neighbors (col) per example (row)
        """
        raise NotImplementedError

class KDTreeNearestNeighbor(AbstractNearestNeighbor):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], layer_dim: int, 
                 dist_metric: DistanceMetric, leaf_size: int):
        print("***** Running DkNN - Initializing KD Trees *****")
        super().__init__(K, layers_to_save, database, layer_dim)
        self.trees = {
            layer: KDTree(self.database[layer][:, :self.layer_dim], leaf_size=leaf_size, metric=dist_metric) 
                for layer in self.layers_to_save
        }

    def nearest_neighbors(self, layer_reps: Dict[int, torch.tensor], output_neighbor_id: bool=False) -> Tuple[np.ndarray]:
        # print("***** Running DkNN - Nearest Neighbor Search (One Batch) KD Tree *****")
        # for each example in the batch, find their total list of neighbors
        # the batch size may be not equivalent to self.args.eval_batch_size, if the 
        # number of training examples is *not* an exact multiple of self.args.eval_batch_size
        # so we use the first layer hidden state's first dimension (presumed batch size)
        batch_size = next(iter(layer_reps.values())).shape[0] # ugly but works
        neighbors = np.zeros((batch_size, len(self.layers_to_save) * self.K))
        distances = np.zeros(neighbors.shape)
        neighbor_ids = None
        if output_neighbor_id:
            neighbor_ids = np.zeros(neighbors.shape)
        # progress_bar = tqdm(range(len(self.layers_to_save)))
        for l, layer in enumerate(self.layers_to_save):
            tree = self.trees[layer]
            dist, batch_indices = tree.query(layer_reps[layer], k=self.K) # (batch_size, K)
            # based on idx, look up tag in database per batch of example
            for i, neighbor_indices in enumerate(batch_indices):
                labels = self.database[layer][neighbor_indices][:, -1] # label is the last element of the 2d np array
                if output_neighbor_id:
                    neighbor_ids[i, l * self.K : (l+1) * self.K] = self.database[layer][neighbor_indices][:, -2] # tag is the second to last element
                neighbors[i, l * self.K : (l+1) * self.K] = labels
            distances[:, l * self.K : (l+1) * self.K] = dist
            # progress_bar.update(1)
        return distances, neighbors, neighbor_ids

from lshashpy3 import LSHash
# from utils import l2norm
# Index building seems to take quite a while
class LocalitySensitiveHashingNearestNeighbor(AbstractNearestNeighbor):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], 
                 layer_dim: int, dist_metric: Callable[[np.ndarray, np.ndarray], float], 
                 num_hash_funct: int):
        print("***** Running DkNN - Initializing LSH Tables *****")
        super().__init__(K, layers_to_save, database, layer_dim)
        self.dist_metric = dist_metric
        start = time.time()
        self.query_tables = {}
        # create one LSH object per layer, and index all training examples
        for layer in layers_to_save:
            self.query_tables[layer] = LSHash(num_hash_funct, layer_dim)
            for ex in database[layer]:
                self.query_tables[layer].index(ex[:layer_dim], extra_data=f"{ex[-1]} {ex[-2]}") # label, tag
        end = time.time()
        print(f"Initializing tables took {end - start}")

    def nearest_neighbors(self, layer_reps: Dict[int, torch.tensor], output_neighbor_id: bool=False) -> np.ndarray:
        batch_size = next(iter(layer_reps.values())).shape[0] # ugly but works
        neighbors = np.zeros((batch_size, len(self.layers_to_save) * self.K))
        distances = np.zeros(neighbors.shape)
        neighbor_ids = None
        if output_neighbor_id:
            neighbor_ids = np.zeros(neighbors.shape)
        for l, layer in enumerate(self.layers_to_save):
            hidden_state_batch = layer_reps[layer]
            for i, h in enumerate(hidden_state_batch):
                # res = self.query_tables[layer].query(h, num_results=self.K, distance_func=l2norm)
                res = self.query_tables[layer].query(h, num_results=self.K, distance_func=self.dist_metric)
                for i, ((vec, extra_data), distance) in enumerate(res):
                    label, tag = extra_data.split(" ")
                    neighbors[i, l * self.K + i] = int(float(label))
                    distances[i, l * self.K + i] = distance
                    if output_neighbor_id: 
                        neighbor_ids[i, l * self.K + i] = int(float(tag))
        return distances, neighbors, neighbor_ids

# from lshashing import LSHRandom
# class LocalitySensitiveHashingNearestNeighbor(AbstractNearestNeighbor):
#     def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], 
#                  layer_dim: int, num_hash_funct: int):
#         print("***** Running DkNN - Initializing LSH Tables *****")
#         super().__init__(K, layers_to_save, database, layer_dim)
#         # start = time.time()
#         self.num_hash_funct = num_hash_funct
#         self.tables = {}
#         for layer in layers_to_save:
#             lshashing = LSHRandom(database[layer][:, :layer_dim], hash_len = self.num_hash_funct, num_tables = 2)
#             self.tables[layer] = lshashing
#         # end = time.time()
#         # print(f"Initializing tables took {end - start}")

#     def nearest_neighbors(self, hidden_states: Tuple[torch.tensor], output_neighbor_id: bool=False) -> np.ndarray:
#         batch_size = hidden_states[0].shape[0]
#         neighbors = np.zeros((batch_size, len(self.layers_to_save) * self.K))
#         neighbor_ids = None
#         if output_neighbor_id:
#             neighbor_ids = np.zeros(neighbors.shape)
#         for l, layer in enumerate(self.layers_to_save):
#             hidden_state_batch = get_layer_representations(hidden_states[layer]).cpu().detach().numpy()
#             for i, hidden_state in enumerate(hidden_state_batch):
#                 results = self.tables[layer].knn_search(self.database[layer][:, :self.layer_dim], hidden_state, k = self.K, buckets = 4, radius = 2)
#                 indices = [ r.index for r in results ]
#                 # get their id based on index, and label
#                 exs = self.database[layer][indices]
#                 neighbors[i, l * self.K : (l+1) * self.K] = exs[:, -1]
#                 if output_neighbor_id:
#                     neighbor_ids[i, l * self.K : (l+1) * self.K] = exs[:, -2]
#         return neighbors, neighbor_ids

class LocalitySensitiveHashingNearestNeighborCustom(AbstractNearestNeighbor):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], 
                 layer_dim: int, dist_metric: Callable[[np.ndarray, np.ndarray], float], num_hash_funct: int):
        print("***** Running DkNN - Initializing LSH Tables *****")
        super().__init__(K, layers_to_save, database, layer_dim)
        # start = time.time()
        self.dist_metric = dist_metric
        self.num_hash_funct = num_hash_funct
        self.hash_vectors = np.random.randn(layer_dim, num_hash_funct)
        self.powers_of_two = np.power(2, np.arange(num_hash_funct - 1, -1, step=-1, dtype=object))
        # pre-computed powers of two for fast binary to integer conversion
        # encode the database and construct the lsh table
        self.table = defaultdict(lambda: defaultdict(list)) # layer -> {bin_index: example tags}
        for layer in layers_to_save:
            bin_indices_bits = database[layer][:, :layer_dim].dot(self.hash_vectors) >= 0 # (len(training_set), num_hash_funct)
            bin_indices = bin_indices_bits.dot(self.powers_of_two) # (len(training_set))
            # bin_indices = convert_boolean_array_to_str(bin_indices_bits) # [len(training_set)]
            for i, bin_index in enumerate(bin_indices):
                self.table[layer][bin_index].append(
                    int(database[layer][i, -2]) # tag of this example
                )
            print(f"Table for layer {layer} has {len(self.table[layer].keys())} bins")
        # end = time.time()
        # print(f"Initializing tables took {end - start}")

    def find_candidate_sets(self, hidden_state: torch.tensor, layer: int) -> List[List[int]]:
        # start = time.time()
        candidate_sets = [[] for _ in range(hidden_state.shape[0])]
        bin_index_bits = hidden_state.dot(self.hash_vectors) >= 0 # (batch_size, num_hash_funct) 

        # search nearby bins and collect candidates
        min_num_of_neighbors_in_batch = 0
        search_radius = 0
        while min_num_of_neighbors_in_batch < self.K:
            # print(f"Minimum is {min_num_of_neighbors_in_batch}, search radius is {search_radius}")
            for different_bits in combinations(range(self.num_hash_funct), search_radius):
                # flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector
                index = list(different_bits)
                alternate_bits = bin_index_bits.copy()
                alternate_bits[:, index] = np.logical_not(alternate_bits[:, index])
                # keys = convert_boolean_array_to_str(alternate_bits)
                keys = alternate_bits.dot(self.powers_of_two)
                for i, key in enumerate(keys):
                    # key = int(key, 2)
                    if key in self.table[layer]:
                        candidate_sets[i].extend(self.table[layer][key])
            min_num_of_neighbors_in_batch = min(map(len, candidate_sets))
            search_radius += 1
        # end = time.time()
        # print(f"A batch of nearest neighbor candidates took {end - start}")
        return candidate_sets

    def nearest_neighbors(self, layer_reps: Tuple[torch.tensor], output_neighbor_id: bool=False) -> np.ndarray:
        neighbors = np.zeros((layer_reps[0].shape[0], len(self.layers_to_save) * self.K))
        neighbor_ids = None
        if output_neighbor_id:
            neighbor_ids = np.zeros(neighbors.shape)
        for l, layer in enumerate(self.layers_to_save):
            hidden_state_batch = layer_reps[layer]
            candidate_sets = self.find_candidate_sets(hidden_state_batch, layer)
            for candidate_set in candidate_sets:
                labels = self.database[layer][candidate_set][:,-1]
                distances_batch = pairwise_distances( # O(n^2) operation
                    hidden_state_batch,
                    self.database[layer][candidate_set][:, :self.layer_dim], 
                    metric=self.dist_metric,
                    n_jobs=64
                ) # (batch_size=layer_reps[0].shape[0], len(candidate_sets[i]))
                for i, distances in enumerate(distances_batch):
                    indices = np.argsort(distances)
                    distances_batch[i, :] = distances_batch[i, :][indices]
                    labels_top_k = labels[indices][:self.K]
                    neighbors[i, l * self.K : (l+1) * self.K] = labels_top_k
                    if output_neighbor_id:
                        neighbor_ids[i, l * self.K : (l+1) * self.K] = self.database[layer][candidate_set][:,-2][indices][:self.K]
        return neighbors, neighbor_ids

class AbstractNearestNeighborFactory(abc.ABC):
    """
    Abstract Factory to produce different Nearest Neighbor getters
    """
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], layer_dim: int):
        self.K = K
        self.layers_to_save = layers_to_save
        self.database = database
        self.layer_dim = layer_dim

    @abc.abstractmethod
    def createNearestNeighborFunction(self) -> AbstractNearestNeighbor:
        raise NotImplementedError

class KDTreeNearestNeighborFactory(AbstractNearestNeighborFactory):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], layer_dim: int, 
                 dist_metric: DistanceMetric, leaf_size: int):
        super().__init__(K, layers_to_save, database, layer_dim)
        self.dist_metric = dist_metric
        self.leaf_size = leaf_size

    def createNearestNeighborFunction(self) -> AbstractNearestNeighbor:
        return KDTreeNearestNeighbor(
            K=self.K, 
            layers_to_save=self.layers_to_save, 
            database=self.database, 
            layer_dim=self.layer_dim,
            dist_metric=self.dist_metric,
            leaf_size=self.leaf_size
        )

class LocalitySensitiveHashingNearestNeighborFactory(AbstractNearestNeighborFactory):
    def __init__(self, K:int, layers_to_save: List[int], database: Dict[int, np.ndarray], 
                 layer_dim: int, dist_metric: Callable[[np.ndarray, np.ndarray], float], num_hash_funct: int):
        super().__init__(K, layers_to_save, database, layer_dim)
        self.dist_metric = dist_metric
        self.num_hash_funct = num_hash_funct

    def createNearestNeighborFunction(self) -> AbstractNearestNeighbor:
        return LocalitySensitiveHashingNearestNeighbor(
            K=self.K, 
            layers_to_save=self.layers_to_save, 
            database=self.database, 
            layer_dim=self.layer_dim,
            dist_metric=self.dist_metric,
            num_hash_funct=self.num_hash_funct
        )
