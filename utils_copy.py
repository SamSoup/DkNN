from typing import List, Dict
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy import stats
import numpy as np
import pandas as pd
import random
import math
import os

def add_to_dataframe(df, results: Dict[str, float], **kwargs):
    results.update(kwargs)
    # small helper to add to the current dataframe
    return pd.concat(
        [df, pd.DataFrame(results, index=[0])], ignore_index = True
    )
    
def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

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


def get_actual_layers_to_save(layer_config: str, num_layers: int):
    switcher = {
        "All": list(range(num_layers)),
        "Embedding Only": [0],
        "Encoder Last": [num_layers / 2 - 1],
        "Embedding + Last": [0, num_layers-1],
        "Last Only": [num_layers-1], 
        "Random 25%": random.sample(list(range(num_layers)), math.ceil(0.25*num_layers)), 
        "Random 50%": random.sample(list(range(num_layers)), math.ceil(0.5*num_layers)), 
        "Random 75%": random.sample(list(range(num_layers)), math.ceil(0.75*num_layers)),
    }
    return switcher[layer_config]

def get_train_representations_from_file(filename: str):
    train_representations = np.loadtxt(filename, delimiter=",")
    train_representations = train_representations[
        train_representations[:, -2].argsort() # sort by idx to get original training example 
    ]
    train_representations = train_representations[:, :-2] # last two are tag and label
    return train_representations

def get_actual_poolers_to_save(layer_config: str, pooler_config: str, layers_to_save: List[int]):
    switcher = {
        "mean_with_attention": ["mean_with_attention"] * len(layers_to_save), # for all layers regardless
        # # use mean with attention for the embedding layer (first), and the rest cls
        "mean_with_attention_and_cls": (
            ["cls"] if layer_config == "Last Only"
            else ["mean_with_attention"] + ["cls"] * (len(layers_to_save) - 1)
        ),
        "mean_with_attention_and_eos": (
            ["eos"] if layer_config == "Last Only"
            else ["mean_with_attention"] + ["eos"] * (len(layers_to_save) - 1)
        ),
        "encoder_mean_with_attention_and_decoder_flatten": (
            ["flatten"] if layer_config == "Last Only" else
            ["mean_with_attention"] * len(layers_to_save)/2 + ["flatten"] * len(layers_to_save)/2
        )
    }
    return switcher[pooler_config]

def find_majority(votes):
    vote_count = Counter(votes)
    tops = vote_count.most_common(1)
    if len(tops) > 1:
        # break ties randomly
        idx = randint(0, len(tops)-1)
        return tops[idx][0]
    return tops[0][0]

def find_majority_batched(y_preds: np.ndarray):
    """(n_samples, n_predictors)"""
    return stats.mode(y_preds.transpose())[0].squeeze()

def compute_metrics(y_true, y_pred, prefix: str, is_multiclass: bool=False):
    if is_multiclass:
        results = {}
        results[f'{prefix}_accuracy'] = accuracy_score(y_true, y_pred)
        agg = ['micro', 'macro', 'weighted']
        for avg in agg:
            results[f'{prefix}_{avg}_f1'] = f1_score(y_true, y_pred, average=avg)
            results[f'{prefix}_{avg}_precision'] = precision_score(
                y_true, y_pred, average=avg
            )
            results[f'{prefix}_{avg}_recall'] = recall_score(
                y_true, y_pred, average=avg
            )
        # also get per class f1, precision, recall
        fl_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        for i, f1, precision, recall in zip(
            range(len(fl_per_class)), fl_per_class, 
                  precision_per_class, recall_per_class):
            results[f'{prefix}_{i}_f1'] = f1
            results[f'{prefix}_{i}_precision'] = precision
            results[f'{prefix}_{i}_recall'] = recall
        return results
    # compute f1, accraucy, precision, recall for binary case
    return {
        f'{prefix}_f1': f1_score(y_true, y_pred),
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_precision': precision_score(y_true, y_pred),
        f'{prefix}_recall': recall_score(y_true, y_pred)
    }

