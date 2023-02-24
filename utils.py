from typing import Iterable, Union, Any, Mapping, Dict, List, Optional, Tuple
from transformers import (
    PreTrainedModel,
    TrainingArguments
)
import random
from tqdm.auto import tqdm
from transformers.utils import find_labels, ModelOutput
from packaging import version
import datasets
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import inspect
import re
import os
import json

def get_neighbor_text_and_label(
    train_inputs: pd.DataFrame, test_inputs: pd.DataFrame, 
    neighbor_indices: np.ndarray
):
    neighbors = pd.DataFrame(
        columns=[f"Neighbor{i+1}{postfix}" 
                 for i in range(neighbor_indices.shape[1]) 
                 for postfix in ("","_label") 
        ]
    )
    for ind in neighbor_indices:
        subset = train_inputs.iloc[ind]
        neighbors.loc[len(neighbors)] = subset[['text', 'label']].to_numpy().flatten()
    neighbors['input'] = test_inputs['text']
    neighbors['input_label'] = test_inputs['label']
    return neighbors

def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def parse_json(path: str):
    with open(path, 'r') as fr:
        results = json.load(fr)
    return results

def random_dropOut(x: Iterable[any], probability: float) -> Iterable[any]:
    return [ item for item in x if random.random() <= probability ]

def get_actual_layers_to_save(layer_config: str, num_layers: int):
    switcher = {
        "All": list(range(num_layers)),
        "Embedding Only": [0],
        "Embedding + Last": [0, num_layers-1],
        "Last Only": [num_layers-1], 
        "Random 25%": random.sample(list(range(num_layers)), math.ceil(0.25*num_layers)), 
        "Random 50%": random.sample(list(range(num_layers)), math.ceil(0.5*num_layers)), 
        "Random 75%": random.sample(list(range(num_layers)), math.ceil(0.75*num_layers)),
    }
    return switcher[layer_config]

def get_actual_poolers_to_save(layer_config: str, pooler_config: str, layers_to_save: List[int]):
    switcher = {
        "mean_with_attention": ["mean_with_attention"] * len(layers_to_save), # for all layers regardless
        # # use mean with attention for the embedding layer (first), and the rest cls
        "mean_with_attention_and_cls": (
            ["cls"] if layer_config == "Last Only"
            else ["mean_with_attention"] + ["cls"] * (len(layers_to_save) - 1)
        )
    }
    return switcher[pooler_config]

def get_train_representations_from_file(filename: str):
    train_representations = np.loadtxt(filename, delimiter=",")
    train_representations = train_representations[
        train_representations[:, -2].argsort() # sort by idx to get original training example 
    ]
    train_representations = train_representations[:, :-2] # last two are tag and label
    return train_representations

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

def save_matrix_with_tags_to_file(filename: str, tags: np.ndarray, mat: np.ndarray):
    with open(filename, "a") as f:
        for tag, row in zip(tags, mat):
            row_str = np.array2string(row, separator='\t', max_line_width=np.inf, 
                                      threshold=np.inf).removeprefix('[').removesuffix(']')
            f.write(f"{tag}\t{row_str}\n")

def remove_file_if_already_exists(path_to_file: str):
    if os.path.exists(path_to_file):
        os.remove(path_to_file)

def hidden_states_to_cpu(hidden_states: Tuple[torch.tensor]) -> List[torch.tensor]:
    ret = []
    for state in hidden_states:
        ret.append(state.detach().cpu())
        del state
    return ret

def get_hidden_states(is_encoder_decoder: bool, outputs: ModelOutput) -> Tuple[torch.tensor]:
    """
    Check if model is encoder-decoder, so that we get hidden states of both the encoder and decoder, 
    otherwise, return just the hidden states

    Args:
        is_encoder_decoder (bool): is this model an encoder-decoder model?
        outputs (ModelOutput): the outputs from the model

    Returns:
        Tuple[torch.tensor]: (batch_size, seq_length, hidden_dim) of len = embedding layer + model hidden layers
    """
    if is_encoder_decoder:
        return hidden_states_to_cpu(outputs["encoder_hidden_states"] + outputs["decoder_hidden_states"])
    return hidden_states_to_cpu(outputs["hidden_states"])

def save_database_after_stacking(layers_to_save, save_reps_path, database):
    for layer in tqdm(layers_to_save):
        database[layer] = np.vstack(database[layer])
        np.savetxt(os.path.join(save_reps_path, f"layer_{layer}.csv"), 
                    database[layer], delimiter=",")

def compute_layer_representations(is_encoder_decoder: bool, hidden_states,
                                  attention_mask, layers_to_save, poolers_to_use,
                                  database):
    if attention_mask is None:
        attention_mask = torch.ones(hidden_states[0].shape).detach().cpu()
    for layer, pooler in zip(layers_to_save, poolers_to_use):
        if is_encoder_decoder and layer >= len(hidden_states) / 2:
            layer_rep_np = pooler(hidden_states[layer], torch.ones(hidden_states[layer].shape[:2]))
        else:
            layer_rep_np = pooler(hidden_states[layer], attention_mask)
        if layer in database:
            database[layer].append(layer_rep_np)
        else:
            database[layer] = [layer_rep_np]
    return database

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

def get_pooled_layer_representations(hidden_states: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    """
    Obtain the layer representations by averaging across all tokens with respect to the attention mask 
    to obtain the embedding at sentence level (for now).
    Args:
        hidden_states (torch.tensor): the hidden states from the model for one layer, with shape
        (batch_size, max_seq_len, embedding_dim)
        attention_mask (torch.tensor): the binary attention mask of shape
        (batch_size, num_heads, sequence_length, sequence_length)

    Returns:
        torch.tensor: (batch_size, self.layer_dim) of layer representations in order
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return F.normalize(torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9), dim=0).squeeze().detach().cpu()

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
    # return np.sum(np.square(probs[:, 1] - labels)) / labels.shape[0]

def convert_boolean_array_to_str(arr: np.ndarray) -> Tuple[str, List[str]]:
    string_rep = re.sub("[\[\] ]", "", np.array2string(arr.astype(int), 
                                                       separator="", 
                                                       max_line_width=np.inf, 
                                                       threshold=np.inf))
    if len(arr.shape) > 1:
        # more than one dimensional, then split by line
        string_rep = string_rep.split("\n")
    return string_rep

def binToHexa(n):
    bnum = int(n)
    temp = 0
    mul = 1
      
    # counter to check group of 4
    count = 1
      
    # char array to store hexadecimal number
    hexaDeciNum = ['0'] * 100
      
    # counter for hexadecimal number array
    i = 0
    while bnum != 0:
        rem = bnum % 10
        temp = temp + (rem*mul)
          
        # check if group of 4 completed
        if count % 4 == 0:
            
            # check if temp < 10
            if temp < 10:
                hexaDeciNum[i] = chr(temp+48)
            else:
                hexaDeciNum[i] = chr(temp+55)
            mul = 1
            temp = 0
            count = 1
            i = i+1
              
        # group of 4 is not completed
        else:
            mul = mul*2
            count = count+1
        bnum = int(bnum/10)
          
    # check if at end the group of 4 is not
    # completed
    if count != 1:
        hexaDeciNum[i] = chr(temp+48)
          
    # check at end the group of 4 is completed
    if count == 1:
        i = i-1
          
    # printing hexadecimal number
    # array in reverse order
    print("\n Hexadecimal equivalent of {}:  ".format(n), end="")
    while i >= 0:
        print(end=hexaDeciNum[i])
        i = i-1

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

def remove_unused_columns(dataset: datasets.Dataset, 
                           model: Union[PreTrainedModel, nn.Module],
                           remove_unused_columns: bool,
                           signature_columns: List[str], 
                           description: Optional[str] = None) -> datasets.Dataset:
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
