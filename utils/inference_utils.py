from typing import List, Dict
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy import stats
import numpy as np
import pandas as pd
import random
import math
import os

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

