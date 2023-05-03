from typing import List, Callable, Dict
from utils import compute_metrics
import numpy as np
import math
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator

def sample_with_replacement(X: np.ndarray, y: np.ndarray, size=1000):
    # X: (samples, features), y: (samples)
    assert X.shape[0] == y.shape[0]
    indices = np.random.choice(np.arange(X.shape[0]), size, replace=True)
    return X[indices], y[indices], indices

def compute_p_value(classifier_A_predictions: np.ndarray,
                    classifier_B_predictions: np.ndarray, 
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    size=1000, iterations=1e5, seed=42):
    np.random.seed(seed) # for reproducability
    assert classifier_A_predictions.size == classifier_B_predictions.size
    assert classifier_A_predictions.size == y_test.size
    deltas = {
        # metric name -> {delta:<>, count:<>, p-value:<>}
    }
    # compute initial deltas
    A_metrics = compute_metrics(
        y_test, classifier_A_predictions, 
        prefix="test", is_multiclass=False
    )
    B_metrics = compute_metrics(
        y_test, classifier_B_predictions, 
        prefix="test", is_multiclass=False
    )
    metric_names = A_metrics.keys()
    for m in metric_names:
        # gain should always be positive, so take absolute value
        # original delta, number of times where δ(x(i)) > 2δ(x)
        deltas[m] = {
            'delta': math.abs(A_metrics[m] - B_metrics[m]),
            'count': 0
        }
    for _ in iterations:
        X_boot, y_boot, indices = sample_with_replacement(X_test, y_test, size=size)
        A_metrics = compute_metrics(
            y_boot, classifier_A_predictions[indices], 
            prefix="test", is_multiclass=False
        )
        B_metrics = compute_metrics(
            y_boot, classifier_B_predictions[indices], 
            prefix="test", is_multiclass=False
        )
        for m in metric_names:
            boot_delta = math.abs(A_metrics[m] - B_metrics[m])
            if boot_delta > 2*deltas[m]['delta']:
                deltas[m]['count'] += 1
    # estimate the p-values
    for m in metric_names:
        deltas[m]['p-value'] = deltas[m]['count'] / iterations

    return deltas
