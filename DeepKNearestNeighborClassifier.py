from typing import Tuple, Callable, Dict
from NearestNeighborLogits import AbstractLogits
from NearestNeighborFinders import AbstractNearestNeighbor
import torch
import numpy as np

class DeepKNearestNeighborClassifier:
    """
    Deep K Nearest Neighbor Classifer
    """
    def __init__(self, NearestNeighborFunction: AbstractNearestNeighbor,
                 dist_to_weight_fct: Callable[[np.ndarray], np.ndarray],
                 LogitsFunction: AbstractLogits,
                 LossFunction: Callable[[torch.tensor, torch.tensor], torch.tensor]):
        self.nn = NearestNeighborFunction
        self.dist_to_weight_fct = dist_to_weight_fct
        self.logits = LogitsFunction
        self.loss = LossFunction

    def compute_loss_and_logits(self, layer_reps: Dict[int, torch.tensor], labels: torch.tensor, device: torch.device,
                                output_neighbors: bool = False) -> Tuple:
        distances, neighbor_labels, neighbor_ids = self.nn.nearest_neighbors(layer_reps, output_neighbors)
        weights = self.dist_to_weight_fct(distances)
        logits = self.logits.compute_logits(neighbor_labels, weights)
        logits = torch.from_numpy(logits).to(device)
        loss = self.loss(logits, labels) if labels is not None else None
        return loss, logits, neighbor_ids, distances
