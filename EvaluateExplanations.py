"""
USAGE: python3 EvaluateExplanations.py ./result_locations_toxigen.json
This script is intended to compute automated metrics to evaluate the example-based
explanations (essentially the nearest neighbors themselves)

The metrics are:

1. BERT Score 
"""
from evaluate import load
from typing import Dict, Any
from tqdm.auto import tqdm
import pandas as pd
import json
import os
import numpy as np
import sys

def parse_json(path):
    with open(path, 'r') as fr:
        results = json.load(fr)
    return results

result_files = sys.argv[1]
toxigen_data_dir = "./data/toxigen"
twitter_hate_data_dir = "./data/twitter-hate/anubrata/"
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"
toxigen_data = {
    "train": pd.read_csv(os.path.join(toxigen_data_dir, train_data_file)),
    "eval_": pd.read_csv(os.path.join(toxigen_data_dir, eval_data_file)),
    "predict_": pd.read_csv(os.path.join(toxigen_data_dir, test_data_file))
}
twitter_hate_data = {
    "train": pd.read_csv(os.path.join(twitter_hate_data_dir, train_data_file)),
    "eval_": pd.read_csv(os.path.join(twitter_hate_data_dir, eval_data_file)),
    "predict_": pd.read_csv(os.path.join(twitter_hate_data_dir, test_data_file))
}
data = {
    "toxigen": toxigen_data,
    "twitter-hate": twitter_hate_data
}

bertscore = load("bertscore")
bleurt = load("bleurt", 'bleurt-large-512', module_type="metric")
inference_modes = ["eval_", "predict_"]
res_locations = parse_json(result_files)
progress_bar = tqdm(range(len(res_locations)))

for loc_data in tqdm(res_locations):
    dfs = data[loc_data['Dataset']]
    train_data = dfs['train']
    # skipping first entry in os.walk since that's always the current directory
    # not including /eval since no nearest neighbor there
    result_paths = [subdir[0] for subdir in list(os.walk(os.path.join(loc_data['output_dir'], "DKNN")))[1:]]
    result_paths.sort()
    for path in tqdm(result_paths):
        if "OLD" in path:
            continue # ignore old results
        for mode in inference_modes:
            # for each inference example, compute a dataframe of neighbors by retreiving their actual value
            neighbor_indices = np.loadtxt(os.path.join(path, f"{mode}neighbors.txt")).astype(int)
            neighbors_df = pd.DataFrame({
                'train_idx': neighbor_indices[:, 1:].flatten(),
                f'{mode}idx': neighbor_indices[:, 0].repeat(neighbor_indices.shape[1] - 1)
            })
            # use the indices to look up the nearest neighbors of each example and their labels
            for my_col, target_col in zip(['text', 'label'], [loc_data['input_colname'], 'label']):
                neighbors_df[f"{mode}{my_col}"] = neighbors_df[f'{mode}idx'].map(
                    lambda idx: dfs[mode].iloc[idx][target_col]
                )
                neighbors_df[f"train_{my_col}"] = neighbors_df['train_idx'].map(
                    lambda idx: train_data.iloc[idx][target_col]
                )
            # compute similarity metrics between the inference example and the training neighbor
            bert_score_results = bertscore.compute(predictions=neighbors_df[f'{mode}text'], 
                                                   references=neighbors_df[f'train_text'], lang="en")
            # make sure that values > 1 are made equal to 1 (since approximately ~[0, 1])
            # neighbors_df['bleurt_score'] = bleurt.compute(predictions=neighbors_df[f'{mode}text'],
            #                                               references=neighbors_df[f'train_text'])['scores']
            neighbors_df['bert_score_precision'] = bert_score_results['precision']
            neighbors_df['bert_score_recall'] = bert_score_results['recall']
            neighbors_df['bert_score_f1'] = bert_score_results['f1']
            # print(path, neighbor_indices.shape)
            # print(neighbors_df)
            neighbors_df.to_csv(
                os.path.join(path, f"{mode}neighbors_with_text_and_label.csv"),
                header=True, index=False
            )
            # input()
            # neighbors_df.to_csv(
            #     "test_neighbors.csv",
            #     header=True, index=False
            # )
            # print("done")
            # input()
    progress_bar.update(1)
