from typing import Tuple, Callable
from NearestNeighborLogits import AbstractLogits
from NearestNeighborFinder import AbstractNearestNeighbor
from utils import compute_confidence, compute_credibility
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

    def compute_loss_and_logits(self, hidden_states: Tuple[torch.tensor], labels: torch.tensor, device: torch.device,
                                output_neighbors: bool = False) -> Tuple:
        neighbor_labels, neighbor_ids = self.nn.nearest_neighbors(hidden_states, output_neighbors)
        logits = self.logits.compute_logits(neighbor_labels)
        logits = torch.from_numpy(logits).to(device)
        loss = self.loss(logits, labels) if labels is not None else None
        return loss, logits, neighbor_ids
