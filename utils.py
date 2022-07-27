from typing import Union, Any, Mapping, Dict, List, Optional
from transformers import (
    PreTrainedModel,
    TrainingArguments
)
from transformers.utils import find_labels
from datasets import Dataset
from packaging import version
import datasets
import torch
import torch.nn as nn
import numpy as np
import inspect


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

def find_signature_columns(model: Union[PreTrainedModel, nn.Module], args: TrainingArguments):
    default_label_names = find_labels(model.__class__)
    label_names = default_label_names if args.label_names is None else args.label_names
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += list(set(["label", "label_ids"] + label_names))
    return signature_columns

def prepare_input(data: Union[torch.Tensor, Any], device: torch.device) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    Taken directly from Transformers.Trainer, with slight modifications without deepseed support
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data

def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], 
                   signature_columns: List[str],
                   device: torch.device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state. Taken directly from Transformers.Trainer, with slight modifications without deepseed support
    """
    inputs = prepare_input(inputs, device)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
            f"training dataset contains keys expected by the model: {','.join(signature_columns)}."
        )
    return inputs

def remove_unused_columns(dataset: Dataset, 
                           model: Union[PreTrainedModel, nn.Module],
                           remove_unused_columns: bool,
                           signature_columns: List[str], 
                           description: Optional[str] = None) -> Dataset:
    """
    Trainer's internal function that drops extraenous keys from the dataset, taken directly from
    Transformers.Trainer with slight modifications

    Args:
        dataset (Dataset): the input dataset to strip
        remove_unused_columns (bool): should we remove unused columns or not
        signature_columns (List[str]): the list of columns to keep
        description (Optional[str], optional): description of the dataset. Defaults to None.

    Returns:
        Dataset: the cleaned dataset without unexpected arguments for the model to call forward on
    """

    if not remove_unused_columns:
        return dataset
    if signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]

    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set "
        print(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            f" If {', '.join(ignored_columns)} are not expected by `{model.__class__.__name__}.forward`, "
            f" you can safely ignore this message."
        )

    columns = [k for k in signature_columns if k in dataset.column_names]

    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        )
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)
