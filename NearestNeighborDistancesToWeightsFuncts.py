from typing import Callable
import numpy as np

class NearestNeighborDistancesToWeightsFuncts:
    """
    This class hosts various methods for converting distances used in nearest
    neighbor calculations to weights, using difference techniques
    
    All function take as input an array of dimensions (batch_size, l x k), where 
    l is the total number of layers and k are the number of neighbors retrieved
    per layer; each element (i, j) is the distance from the ith inference sample 
    to the (j mod k)th neighbor of layer (j / k) 
    """
    def __init__(self, K:int):
        self.K = K
        self.name_to_fct = {
            "normalized_linear_mapping_dudani": self.normalized_linear_mapping_dudani,
            "inverse": self.inverse,
            "inverse_squared": self.inverse_squared,
            "uniform": self.uniform
        }

    def uniform(self, distances: np.ndarray):
        return np.ones(distances.shape)

    def normalized_linear_mapping_dudani(self, distances: np.ndarray):
        max_d = np.max(distances, axis=1)
        min_d = np.min(distances, axis=1)
        denominator = max_d - min_d
        pass

    def inverse(self, distances: np.ndarray):
        return 1 / (distances + 1e-8)
    
    def inverse_squared(self, distances: np.ndarray):
        return 1 / (np.square(distances) + 1e-8)

    # def rank_weight(self, distances: np.ndarray):
    #     return distances.shape[1] - 

    def get(self, name: str) -> Callable[[np.ndarray], np.ndarray]:
        return self.name_to_fct[name]
