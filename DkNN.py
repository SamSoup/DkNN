from typing import List, Dict, Union, Any, Tuple, Optional
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
from utils import get_layer_representations
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DkNN:
    """
    Informal interface for Deep K Nearest Neighbors Search, whereby the only functionality
    required is that given a batch of tensors, return the k nearest neighbor per layer
    for the examples in the batch
    """
    def __init__(self, k:int, layers_to_save: List[int], database: Dict[int, np.array], 
                 layer_dim: int, label_list: List[Any] = None):
        """
        Initialize basic parameters that all KNN classifiers will need

        Args:
            k (int): number of nearest neighbors to obtain per sample per layer
            layers_to_save (List[int]): the list of layers to save
            database (Dict[int:np.array]): the database of example representations per layer
            layer_dim (int): the hidden size of each layer
        """

        self.k = k
        self.layers_to_save = layers_to_save
        self.database = database
        self.layer_dim = layer_dim
        self.label_list = label_list
        self.label_to_id = {label: i for i, label in enumerate(label_list)}

    def nearest_neighbors(self, hidden_states: Tuple[torch.tensor]) -> np.array:
        """
        Given a batch of input example tensors, return the k nearest neighbor per layer
        Andoni et. al. https://arxiv.org/pdf/1509.02897.pdf
        Possible algorithms:
        
        1. space partitioning with indexing
        2. dimension reduction 
        3. sketching 
        4. Locality-Sensitive Hashing (LSH): sub-linear query time and sub-quadratic space complexity
            - idea: bucket similar samples through a hash function so as to maximize collision for
            similar vectors (layer representations here)
            - locality-sensitive hash functions: probability of collision is higher for "nearby" points 
            than for points that are far apart
            - how sensitive depends on the rho = \frac{log 1/p_1}{log 1/p_2}, where p_1 and p_2 are the
            collision probability for near and far points (depending on some distance r)
            - cross-polytope vs. hyperplane LSH
        5. KDTree: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree

        Args:
            hidden_states (Tuple[torch.tensor]): (num_layers+1, batch_size, max_seq_len, embedding_dim)

        Returns:
            np.array: the collection of neighbors (col) per example (row)
        """
        pass

    def compute_loss_and_logits_DkNN(self, neighbors: np.ndarray, labels: Optional[torch.tensor]):
        """
        Compute loss and logits using the nearest k neighbors

        Args:
            neighbors (np.ndarray): an array of (batch_size, num of layers * k)
            labels: (Optional[torch.tensor]): labels for the batch of input examples, if any
        """
        # first find the log-probabilities for each class 
        probs = np.zeros((neighbors.shape[0], len(self.label_list)))
        for i, label in enumerate(self.label_list):
            label_id = self.label_to_id[label]
            prob = (neighbors == label_id).sum(axis=1) / neighbors.shape[1]
            probs[:, i] = prob
        logits = torch.log(torch.from_numpy(probs).to(device)) # (self.args.eval_batch_size, len(self.label_list))
        if labels is not None:
            # Negative log likelihood loss for C potential classes
            loss = F.nll_loss(logits, labels) # torch.tensor(len(self.label_list))
        else:
            loss = None
        # TODO: Note that here we elected to do NLL loss and is may NOT be what the
        # model is expecting; but since DkNN does not rely on back-prop (yet), 
        # this is not an issue YET
        # TODO: infinite loss sometimes...
        return loss, logits

class DkNN_KD_TREE(DkNN):
    def __init__(self, k:int, layers_to_save: List[int], database: List[np.array], 
                 layer_dim: int, label_list: List[Any] = None):
        DkNN.__init__(k, layers_to_save, database, layer_dim, label_list)
        self.trees = {
            layer: KDTree(self.database[layer][:, :self.layer_dim], metric="l2") 
                for layer in self.layers_to_save
        }

    def nearest_neighbors(self, hidden_states: Tuple[torch.tensor]) -> np.array:
        print("***** Running DkNN - Nearest Neighbor Search (One Batch) KD Tree *****")
        # for each example in the batch, find their total list of neighbors
        # the batch size may be not equivalent to self.args.eval_batch_size, if the 
        # number of training examples is *not* an exact multiple of self.args.eval_batch_size
        # so we use the first layer hidden state's first dimension (presumed batch size)
        neighbors = np.zeros((hidden_states[0].shape[0], len(self.layers_to_save) * self.k))
        progress_bar = tqdm(range(len(self.layers_to_save)))
        for l, layer in enumerate(self.layers_to_save):
            tree = self.trees[layer]
            dist, batch_indices = tree.query(get_layer_representations(hidden_states[layer]), k=self.k) # (batch_size, k)
            # based on idx, look up tag in database per batch of example
            for i, neighbor_indices in enumerate(batch_indices):
                labels = self.database[layer][neighbor_indices][:, -1] # label is the last element of the 2d np array
                labels_ids = list(map(lambda l: self.label_to_id[l], labels))
                neighbors[i, l * self.k : (l+1) * self.k] = labels_ids
            progress_bar.update(1)
        return neighbors

class DkNN_LSH(DkNN):
    def __init__(self, k:int, layers_to_save: List[int], database: List[np.array], 
                layer_dim: int, label_list: List[Any] = None):
        DkNN.__init__(k, layers_to_save, database, layer_dim, label_list)
