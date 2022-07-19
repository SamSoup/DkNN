import torch
import numpy as np

def get_layer_representations(hidden_states: torch.tensor) -> torch.tensor:
    """
    Obtain the layer representations by averaging across all tokens to obtain the embedding
    at sentence level (for now). This method may be changed if necessary.
    
    Args:
        hidden_states (torch.tensor): the hidden states from the model for one layer, with shape
        (batch_size, max_seq_len, embedding_dim)

    Returns:
        torch.tensor: (batch_size, self.layer_dim) of layer representations in order
    """

    # average across all tokens to obtain embedding -> (batch_size, embedding_dim)
    return torch.mean(hidden_states, dim=1).squeeze().detach().cpu()

def compute_nonconformity_score(neighbors: np.ndarray, label_id: int) -> np.ndarray:
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
    return (neighbors != label_id).sum(axis=1)

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
    return np.max(empirical_p, axis=1)