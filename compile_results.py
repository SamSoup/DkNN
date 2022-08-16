"""
This script is for personal data collection use and compiles results into an
user specified file
"""
from scipy.special import softmax
from utils import compute_confidence
from typing import Dict, List, Union
from datasets import load_metric
from collections import ChainMap
from tqdm.auto import tqdm
import pandas as pd
import json
import os
import numpy as np
import sys

def compute_confidence_from_matrix(labels: np.ndarray, logits: np.ndarray, is_DKNN: bool, is_conformal: bool):
    if is_DKNN:
        if is_conformal:
            # probability that any label other than the prediction is the true label
            # confidence is 1 - second max, using p-values computed from conformal scores
            confidence = compute_confidence(logits).reshape(-1, 1)
            probs = np.append(confidence, 1 - confidence, 1)
        else:
            # log probability, conver to probability
            probs = np.exp(logits)
    else:
        # transform scores to probabilities
        probs = softmax(logits, axis=1)
    predicts = np.argmax(logits, axis=1)
    # 75% confidence in predicting the positive class on a give
    # example gets a reward of 0.75 if gold is positive, and 0.25 if gold is negative ??
    # note that confidence (from probability will never be below 0.5)
    overall_reward = ((predicts == labels).astype(int) * probs[range(len(probs)), predicts] + 
                      (predicts != labels).astype(int) * probs[range(len(probs)), labels])
    overall_reward = sum(overall_reward) / len(overall_reward)
    return overall_reward

def compute_results_for_test_set(prefix: str, labels: np.ndarray, predictions: np.ndarray, metric_descs: List[str], *args):
    computed_scores = {}
    for metric_desc in metric_descs:
        if metric_desc == "confidence":
            computed_scores[prefix + metric_desc] = compute_confidence_from_matrix(labels, *args)
        else:
            metric_func = load_metric(metric_desc)
            computed_scores[prefix + metric_desc] = metric_func.compute(predictions=predictions, references=labels)[metric_desc]
    # get per-class f1 too
    f1_scores = load_metric("f1").compute(predictions=predictions, references=labels, average=None)["f1"]
    computed_scores[prefix + "f1-negative"] = f1_scores[0]
    computed_scores[prefix + "f1-positive"] = f1_scores[1]
    return computed_scores

def add_to_dataframe(df: pd.DataFrame, results: Dict[str, Union[str, float]]):
    new_row = { col: results[col] for col in df.columns }
    return pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index = True)

result_files = sys.argv[1]
output_file = sys.argv[2]

with open(result_files, 'r') as f:
     res_locations = json.load(f)
metric_descs = ["accuracy", "f1", "precision", "recall", "confidence"]
columns = ["Dataset", "Model", "Logits", "DKNN_method"]
inference_modes = ["eval_", "predict_"]
# add to dataset columns the metrics for each mode
for metric_desc in metric_descs:
    for mode in inference_modes:
        columns.append(mode + metric_desc)
        if metric_desc == "f1":
            columns.append(mode + "f1-positive")
            columns.append(mode + "f1-negative")
res_df = pd.DataFrame(columns=columns)

train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"
approaches = [
    {"Logits": "Scores", "DKNN_method": "None", "dir_extension": "eval", "get_neighbors": False},
    {"Logits": "Conformal P-value", "DKNN_method": "Conformal", "dir_extension": "DKNN/KD-Conformal", "get_neighbors": True},
    {"Logits": "log-probability", "DKNN_method": "Normal", "dir_extension": "DKNN/KD-Normal", "get_neighbors": True}
]

progress_bar = tqdm(range(len(res_locations) * len(approaches) * len(inference_modes)))
for loc_data in res_locations:
    train_data = pd.read_csv(os.path.join(loc_data['data_dir'], train_data_file))
    eval_labels = pd.read_csv(os.path.join(loc_data['data_dir'], eval_data_file))['label'].to_numpy()
    test_labels = pd.read_csv(os.path.join(loc_data['data_dir'], test_data_file))['label'].to_numpy()
    for app_data in approaches:
        all_results = ChainMap(app_data, loc_data)
        for mode in inference_modes:
            logits = np.loadtxt(os.path.join(all_results['output_dir'], all_results['dir_extension'], mode + "logits.txt"))
            results_path = os.path.join(all_results['output_dir'], all_results['dir_extension'], mode + "results")
            if mode == "eval_":
                with open(results_path + ".json", 'r') as fr:
                    results = json.load(fr) # evaluation results is a json dataframe
                results[mode + "confidence"] = compute_confidence_from_matrix(
                    eval_labels, logits[:, 1:], all_results["get_neighbors"], all_results["DKNN_method"] == "Conformal"
                )
            else:
                # test results must be computed
                predict_labels = pd.read_csv(results_path + ".txt", sep="\t")['prediction']
                results = compute_results_for_test_set(
                    mode, test_labels, predict_labels, metric_descs, logits[:, 1:], 
                    all_results["get_neighbors"], all_results["DKNN_method"] == "Conformal"
                )
            all_results = ChainMap(all_results, results)
            tag_indices = logits[:, 0]
            if all_results["get_neighbors"]:
                neighbor_indices = np.loadtxt(os.path.join(loc_data['output_dir'], app_data['dir_extension'], mode + "neighbors.txt"))
                indices = neighbor_indices[:, 0].repeat(neighbor_indices.shape[1] - 1) # total number of neighbors / example
                # use the indices to look up the nearest neighbors of each example and their labels
                neighbors_df = pd.concat(
                    map(lambda indices: train_data.iloc[indices], neighbor_indices[:, 1:]), 
                    ignore_index = True, copy=False
                )
                neighbors_df[mode + 'idx'] = indices.astype(int)
                neighbors_df['train_idx'] = neighbor_indices[:, 1:].flatten().astype(int)
                neighbors_df.to_csv(
                    os.path.join(loc_data['output_dir'], app_data['dir_extension'], mode + "neighbors_with_text_and_label.csv"),
                    header=True, index=False
                )
            progress_bar.update(1)
        res_df = add_to_dataframe(res_df, all_results)

# finally - save results
res_df.to_csv(output_file, header=True, index=False)
