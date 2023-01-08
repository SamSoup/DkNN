"""
USAGE: python3 compile_results_new.py ./result_locations_toxigen.json compiled_results_toxigen.csv
"""
from transformers import TrainingArguments
from args import DKNNArguments, DataArguments, ModelArguments
from scipy.special import softmax
from sklearn.metrics import brier_score_loss
from utils import compute_credibility, compute_brier_score, compute_instance_level_brier, parse_json
from typing import Dict, Any
from collections import ChainMap
from tqdm.auto import tqdm
from pycalib import metrics
import dataclasses
import pandas as pd
import json
import os
import torch
import numpy as np
import sys
import pickle

def compute_confidence_from_matrix(labels: np.ndarray, logits: np.ndarray, is_DKNN: bool, is_conformal: bool):
    """
    Compute confidence
    
    The reason this is NOT done when running the script is because we do not have access to the logits at
    evaluation time; hence why this is done after the main script executes as a post-processing step 
    """
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
    indices = range(len(probs))
    overall_reward = ((predicts == labels).astype(int) * probs[indices, predicts] + 
                      (predicts != labels).astype(int) * probs[indices, labels])
    overall_reward = sum(overall_reward) / len(overall_reward)
    return overall_reward

def evaluate_confidence_from_matrix(labels: np.ndarray, probs: np.ndarray):
    """
    Evaluate confidence using a simple scoring scheme
    
    The reason this is NOT done when running the script is because we do not have access to the logits at
    evaluation time; hence why this is done after the main script executes as a post-processing step 
    """
    predicts = np.argmax(probs, axis=1)
    # 75% confidence in predicting the positive class on a give
    # example gets a reward of 0.75 if gold is positive, and 0.25 if gold is negative ??
    # note that confidence (from probability will never be below 0.5)
    overall_reward = ((predicts == labels).astype(int) * probs[range(len(probs)), predicts] + 
                      (predicts != labels).astype(int) * probs[range(len(probs)), labels])
    overall_reward = sum(overall_reward) / len(overall_reward)
    return overall_reward

def convert_logits_into_probability(logits: np.ndarray, method: str):
    if method == "normal":
        normalized_logits = softmax(logits, axis=1)
    elif method == "nonconformal":
        normalized_logits = logits / np.tile(logits.sum(axis=1), (2, 1)).T
    else:
        normalized_logits = np.exp(logits) # convert log probability to probability 
    return normalized_logits

def compute_confidence(logits: np.ndarray, method: str):
    if method == "nonconformal":
        return compute_credibility(logits) # credibility uses empirical_p
    else:
        # conformal and normal
        probs = convert_logits_into_probability(logits, method)
        return np.max(probs, axis=1)

def add_group_results_to_all_results(group_results: Dict[str, float], mode: str, metric: str, all_results: Dict[str, Any]):
    for group, res in group_results.items():
        all_results[f"{group}_{mode}{metric_desc}"] = res
    return all_results

def add_to_dataframe(df: pd.DataFrame, results: Dict[str, Any]):
    new_row = { col: (results[col] if type(results[col]) is not list 
                      else ', '.join(map(str, results[col]))) 
               for col in df.columns if col in results }
    # if 'svm_accuracy' in results:
    #     print(results['svm_accuracy'])
    #     input()
    return pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index = True)

result_files = sys.argv[1]
output_file = sys.argv[2]

toxigen_eval = pd.read_csv("./data/toxigen/eval_data.csv")
toxigen_test = pd.read_csv("./data/toxigen/test_data.csv")

res_locations = parse_json(result_files)
# there are four places to look for argument files to put in our dataset
# note we that we leave model arguments (just names)
all_arguments = [field.name for field in dataclasses.fields(DKNNArguments) + 
                 dataclasses.fields(DataArguments) + dataclasses.fields(TrainingArguments)]
metric_descs = [
    "accuracy", "f1", "precision", "recall", # "confidence", 
    "samples_per_second", "samples", "steps_per_second", "runtime", 
    "probs", "brier", "ECE", 'all_brier' # all_brier is the instance level brier 
]
group_metric_descs = ["accuracy", "f1", "precision", "recall"]
simpler_models = ["svm", "knn"]
unique_sorted_groups = np.sort(toxigen_eval.target_group.unique())

# all_arguments
columns = ["Dataset", "Model", "DKNN_unique_train_examples_for_validation"] + all_arguments
# columns = ["Dataset", "Model", "DKNN_method", "Logits"]
dir_extensions = ["eval", "DKNN"]
inference_modes = ["eval_", "predict_"]
# add to dataset columns the metrics for each mode
for metric_desc in metric_descs:
    for mode in inference_modes:
        columns.append(mode + metric_desc)
        if metric_desc == "f1":
            columns.append(mode + "f1-positive")
            columns.append(mode + "f1-negative")
# as well as the group metrics
for metric_desc in group_metric_descs:
    for group in unique_sorted_groups:
        for mode in inference_modes:
            columns.append(f"{group}_{mode}{metric_desc}")
# as well as the simpler model's metric
for model in simpler_models:
    for metric in group_metric_descs:
        columns.append(f"{model}_{metric}")

res_df = pd.DataFrame(columns=columns)
bins = 10
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"

# first compile baseline results
## bart
res_df = add_to_dataframe(res_df, parse_json(
    os.path.join(os.getcwd(), 
                 "./baseline_results/baseline_deberta_results.json")
    )
)
## deberta
res_df = add_to_dataframe(res_df, parse_json(
    os.path.join(os.getcwd(), 
                 "./baseline_results/baseline_sentencebert_results.json")
    )
)
## sentence bert
res_df = add_to_dataframe(res_df, parse_json(
    os.path.join(os.getcwd(), 
                 "./baseline_results/baseline_bart_results.json")
    )
)
## word2vec
res_df = add_to_dataframe(res_df, parse_json(
    os.path.join(os.getcwd(), 
                 "./baseline_results/baseline_word2vec_results.json")
    )
)

progress_bar = tqdm(range(len(res_locations)))
for loc_data in tqdm(res_locations):
    train_data = pd.read_csv(os.path.join(loc_data['data_dir'], train_data_file))
    eval_labels = pd.read_csv(os.path.join(loc_data['data_dir'], eval_data_file))['label'].to_numpy()
    test_labels = pd.read_csv(os.path.join(loc_data['data_dir'], test_data_file))['label'].to_numpy()
    # skipping first entry in os.walk since that's always the current directory
    result_paths = ([os.path.join(loc_data['output_dir'], "eval")] + 
                    [subdir[0] for subdir in list(os.walk(os.path.join(loc_data['output_dir'], "DKNN")))[1:]])
    # parse args
    training_args = torch.load(os.path.join(loc_data['output_dir'], "training_args.bin")).to_dict()
    for path in tqdm(result_paths):
        if "OLD" in path:
            continue # ignore old results
        data_args = torch.load(os.path.join(path, "data_args.bin")).__dict__
        DKNN_args = torch.load(os.path.join(path, "DKNN_args.bin")).__dict__
        all_results = loc_data
        for mode in inference_modes:
            # parse logits, results, data_args, and DKNN_args
            logits = np.loadtxt(os.path.join(path, f"{mode}logits.txt"))
            results = parse_json(os.path.join(path, f"{mode}results.json"))
            for metric_desc in group_metric_descs:
                group_results = parse_json(os.path.join(path, f"{mode}group_{metric_desc}.json"))
                all_results = add_group_results_to_all_results(group_results, mode, metric_desc, all_results)
            if DKNN_args['layers_to_save_desc'] == 'Last Only':
                reduced_model_results = parse_json(os.path.join(path, "simpler_models.json"))
            else:
                reduced_model_results = {}
            all_results = ChainMap(all_results, training_args, results, data_args, DKNN_args, reduced_model_results)
            # compute confidence (turned off for now)
            labels = eval_labels if mode == "eval_" else test_labels
            probs = convert_logits_into_probability(logits[:, 1:], all_results["prediction_method"])
            # all_results[f"{mode}confidence"] = compute_confidence_from_matrix(
            #     labels, logits[:, 1:], all_results["do_DKNN"], all_results["prediction_method"] == "conformal")
            all_results[f"{mode}probs"] = pickle.dumps(probs)
            confidence = compute_confidence(logits[:, 1:], all_results["prediction_method"])
            all_results[f"{mode}confidence"] = evaluate_confidence_from_matrix(labels, probs)
            all_results[f"{mode}all_brier"] = pickle.dumps(compute_instance_level_brier(probs, labels))
            # all_results[f"{mode}brier"] = compute_brier_score(probs, labels)
            all_results[f"{mode}brier"] = brier_score_loss(y_true=labels, y_prob=probs[:, 1], pos_label=1) # pos class prob only
            all_results[f"{mode}ECE"] = metrics.ECE(labels, probs, bins=bins, ece_full=True)
            # for each inference example, compute a dataframe of neighbors by retreiving their actual value
#             if all_results["do_DKNN"]:
#                 neighbor_indices = np.loadtxt(os.path.join(path, f"{mode}neighbors.txt"))
#                 # use the indices to look up the nearest neighbors of each example and their labels
#                 neighbors_df = pd.concat(
#                     map(lambda indices: train_data.iloc[indices], neighbor_indices[:, 1:]), 
#                     ignore_index = True, copy=False
#                 )
#                 neighbors_df[f'{mode}idx'] = neighbor_indices[:, 0].repeat(neighbor_indices.shape[1] - 1) # total number of neighbors / example
#                 neighbors_df['train_idx'] = neighbor_indices[:, 1:].flatten()
#                 neighbors_df.to_csv(
#                     os.path.join(path, f"{mode}neighbors_with_text_and_label.csv"),
#                     header=True, index=False
#                 )
        # print(set(res_df.columns).difference(set(all_results.keys())))
        # assert(set(res_df.columns).issubset(set(all_results.keys())))
        res_df = add_to_dataframe(res_df, all_results)
    progress_bar.update(1)
res_df.to_csv(output_file, header=True, index=False)
