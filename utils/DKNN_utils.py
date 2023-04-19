import numpy as np

def compute_nonconformity_score(neighbors: np.ndarray, label_id: int, weights: np.ndarray) -> np.ndarray:
    """
    We compute the non-conformity score for each example in the evaluation dataset as:
            α(x, j) = ∑_{ λ ∈ 1..l} |i ∈ Ω_λ : i != j|, for all x ∈ caliberation set and j ∈ label_list
    where x is some input example, j is some label class. Lambda is a counter over all possible layers,
        and Ω_λ is the set of labels of nearest neighbors for example `x` at layer `lambda`. 

    Args:
        neighbors (np.ndarray): the set of nearest neighbors (col) for each example (row)
        label_id (int): the candidate label id

    Returns:
        np.ndarray: the non-conformity score for each example in the batch
    """
    return ((neighbors != label_id) * weights).sum(axis=1)

def compute_confidence(empirical_p: np.ndarray) -> np.ndarray:
    """
    We compute the confidence of predictions as 1 minus the second largest empirical p-value. 
    This is the probability that any label other than the prediction is the true label. 

    Args:
        empirical_p (np.ndarray): (eval_batch_size, len(self.label_list))

    Returns:
        np.ndarray: the confidence scores for each example (row)
    """
    empirical_p.sort()
    return 1 - empirical_p[:, -2]

def compute_credibility(empirical_p: np.ndarray) -> np.ndarray:
    """
    We compute the credibility of a prediction as "the empirical p-value of the prediction: it bounds the
    nonconformity of any label assigned to the test input with the training data." 

    Args:
        empirical_p (np.ndarray): (eval_batch_size, len(self.label_list))

    Returns:
        np.ndarray: the credibility scores for each example (row)
    """
    return empirical_p.max(axis=1)

def compute_instance_level_brier(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.zeros(probs.shape)
    labels[:, 1] = 1
     # the element at the 0th and 1th column is the same
    return np.square(np.subtract(probs, labels))[:, 0]

def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    We compute the brier score of a set of predictions as the sum of squares 
    between prediction probability for the positive class and the actual (binary) label

    Args:
        probs (np.ndarray): (eval_batch_size, len(self.label_list))
        labels (np.ndarray): (eval_batch_size)
    Returns:
        float: the brier score in [0, 1]
    """
    return np.mean(compute_instance_level_brier(probs, labels))