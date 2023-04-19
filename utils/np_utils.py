from typing import List, Tuple
import numpy as np
import re

def flip_bits(arr: np.ndarray) -> np.ndarray:
    """
    Given an numpy array of all either 0 or 1s, flip 0 to 1 and 1 to 0.
    """

    return (~arr.astype(bool)).astype(int)

def randargmax(logits: np.ndarray) -> np.ndarray:
    """
    Given a batch of logits with shape (batch_size, # of classes), return the 
    predicted indices (class ids) for each example, with ties broken arbitrarily
    (whereas np.argmax only returns the first [smallest] index)

    Args:
        logits (np.ndarray): the logits of shape (batch_size, # of classes)

    Returns:
        np.ndarray: the predicted class ids for each example as the indices in the original logits array
    """
    return np.argmax(np.random.random(logits.shape) * (logits==np.amax(logits, axis=1, keepdims=True)), axis=1)

def l2norm(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x-y)

def flip_bits(arr: np.ndarray) -> np.ndarray:
    """
    Given an numpy array of all either 0 or 1s, flip 0 to 1 and 1 to 0.
    """
    return (~arr.astype(bool)).astype(int)

def l2norm(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x-y)

def convert_boolean_array_to_str(arr: np.ndarray) -> Tuple[str, List[str]]:
    string_rep = re.sub("[\[\] ]", "", np.array2string(arr.astype(int), 
                                                       separator="", 
                                                       max_line_width=np.inf, 
                                                       threshold=np.inf))
    if len(arr.shape) > 1:
        # more than one dimensional, then split by line
        string_rep = string_rep.split("\n")
    return string_rep
