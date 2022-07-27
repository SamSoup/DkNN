from typing import Tuple, Callable
from NearestNeighborLogits import AbstractLogits
from NearestNeighborFinder import AbstractNearestNeighbor
import torch

class DeepKNearestNeighborClassifier:
    """
    Deep K Nearest Neighbor Classifer
    """
    def __init__(self, NearestNeighborFunction: AbstractNearestNeighbor, 
                 LogitsFunction: AbstractLogits,
                 LossFunction: Callable[[torch.tensor, torch.tensor], torch.tensor]):
        self.nn = NearestNeighborFunction
        self.logits = LogitsFunction
        self.loss = LossFunction    
    
    def compute_loss_and_logits(self, hidden_states: Tuple[torch.tensor], labels: torch.tensor, device: torch.device):
        neighbors = self.nn.nearest_neighbors(hidden_states)
        logits = self.logits.compute_logits(neighbors)
        logits = torch.from_numpy(logits).to(device)
        loss = self.loss(logits, labels) if labels is not None else None
        return loss, logits
