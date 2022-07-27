import abc
import torch
import numpy as np
from typing import List
from utils import compute_nonconformity_score

# TODO: add Doc-string and comments where need be

class AbstractLogits(abc.ABC):
    def __init__(self, label_list: List[int]):
        self.label_list = label_list
        self.label_to_id = {label: i for i, label in enumerate(label_list)}

    @abc.abstractmethod
    def compute_logits(self, neighbors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LogProbabilityLogits(AbstractLogits):
    def compute_logits(self, neighbors: np.ndarray) -> np.ndarray:
        # first find the log-probabilities for each class 
        probs = np.zeros((neighbors.shape[0], len(self.label_list)))
        for i, label in enumerate(self.label_list):
            label_id = self.label_to_id[label]
            # probability of each class = # of examples in each class / total neighbors
            probs[:, i] = (neighbors == label_id).sum(axis=1) / neighbors.shape[1]
        logits = np.log(probs) 
        return logits # (self.args.eval_batch_size, len(self.label_list))

class ConformalLogits(AbstractLogits):
    def __init__(self, label_list: List[int], scores: np.ndarray):
        super().__init__(label_list)
        self.scores = scores

    def compute_logits(self, neighbors: np.ndarray) -> np.ndarray:
        empirical_p = np.zeros((neighbors.shape[0], len(self.label_list))) 
        for i, label in enumerate(self.label_list):
            label_id = self.label_to_id[label]
            nonconform_scores = compute_nonconformity_score(neighbors, label_id)
            for j, score in enumerate(nonconform_scores):
                empirical_p[j, i] = (self.scores >= score).sum() / len(self.scores)
        return empirical_p # (self.args.eval_batch_size, len(self.label_list))

class AbstractLogitsFactory(abc.ABC):
    """
    Abstract Factory to produce different logits
    """
    @abc.abstractmethod
    def createLogitsFunction(self) -> AbstractLogits:
        raise NotImplementedError

class LogProbabilityLogitsFactory(AbstractLogitsFactory):
    def __init__(self, label_list: List[int]):
        self.label_list = label_list

    def createLogitsFunction(self) -> AbstractLogits:
        return LogProbabilityLogits(self.label_list)

class ConformalLogitsFactory(AbstractLogitsFactory):
    def __init__(self, label_list: List[int], scores: np.ndarray):
        self.label_list = label_list
        self.scores = scores

    def createLogitsFunction(self) -> AbstractLogits:
        return ConformalLogits(self.label_list, self.scores)
