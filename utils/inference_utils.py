from typing import List, Dict
from collections import Counter
from random import randint
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy import stats
import numpy as np

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
        # simply set accuracy, precision, recall, f1 = macro-averaged
        results[f'{prefix}_f1'] = results[f'{prefix}_macro_f1']
        results[f'{prefix}_precision'] = results[f'{prefix}_macro_precision']
        results[f'{prefix}_recall'] = results[f'{prefix}_macro_recall']
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