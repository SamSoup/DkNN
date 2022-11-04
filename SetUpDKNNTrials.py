"""
The purpose of this trial is to take a base trial configuration for a model, and 
then generate all possible trials that we want to run for that model.

We assume that the base trial configuration contains all the necessary 
information needed as specified below:

- all directories should be the base output directory without specific names

Usage:

python3 SetUpDKNNTrials.py <path to base json> <output directory> <number_of_layers>
"""

from tqdm.auto import tqdm
from itertools import product
from pathlib import Path
from typing import Dict, List
import sys
import os
import random
import copy
import math
import json
import pandas as pd

# base json file
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    base_config = json.load(f)
output_dir = sys.argv[2]
num_layers = int(sys.argv[3])

def expand_grid(trials: Dict[str, List[str]]):
   return pd.DataFrame([row for row in product(*trials.values())], 
                       columns=trials.keys())

possible_layer_configs = ["All", "Embedding Only", "Embedding + Last", "Last Only"]
# possible_layer_configs = ["All", "Embedding Only", "Embedding + Last", "Last Only", "Random 25%", "Random 50%", "Random 75%"]
possible_pooler_configs = ["mean_with_attention", "mean_with_attention_and_cls"]

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

def get_actual_poolers_to_save(pooler_config: str, layers_to_save: List[int]):
    switcher = {
        "mean_with_attention": ["mean_with_attention"] * len(layers_to_save), # for all layers regardless
        # # use mean with attention for the embedding layer (first), and the rest cls
        "mean_with_attention_and_cls": ["mean_with_attention"] + ["cls"] * (len(layers_to_save) - 1)
    }
    return switcher[pooler_config]

trials = {
    'prediction_method': ["nonconformal", "normal"], # always run conformal before normal
    'layers_to_save_desc': possible_layer_configs,
    'poolers_to_use_desc': possible_pooler_configs,
    'K': list(range(1, 10, 2)) + [99, 999], 
    'dist_to_weight_fct': ["uniform", "inverse", "inverse_squared"]
}

directories = ["output_dir", "save_database_path", "save_nonconform_scores_path"]
trials = expand_grid(trials)
trials = trials.reset_index()

# Filter down trials

# All trials are repeated across NonConformal vs. Conformal schemes, 
# organized by four major sections
# 1. Different Layer Choices
# 2. Different Layer Pooling Choices
# 3. Different K Choices
# 4. Different Weighted K Choices
for i, trial in tqdm(trials.iterrows()):
    id = f"trial-{i}"
    curr_config = copy.deepcopy(base_config)
    for key in trials.keys():
        curr_config[key] = trial[key]
    # based on the descriptions for layers and poolers, act accordingly
    curr_config['layers_to_save'] = get_actual_layers_to_save(trial["layers_to_save_desc"], num_layers)
    curr_config['poolers_to_use'] = get_actual_poolers_to_save(trial["poolers_to_use_desc"], curr_config['layers_to_save'])
    for dir in directories:
        curr_config[dir] = os.path.join(base_config[dir], id)
    # set reading from database and scores 
    curr_config['read_from_database_path'] = True if trial['prediction_method'] == "conformal" else False
    curr_config['read_from_scores_path'] = False
    with open(os.path.join(output_dir, f"{id}.json"), 'w') as f:
        json.dump(curr_config, f)

# Save trial metadata
trials.to_csv(os.path.join(output_dir, "trial_metadata.csv"), header=True, index=False)
