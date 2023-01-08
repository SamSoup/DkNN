"""
For all validation examples, then
- For the correct neighbors (Hs), find nearest NH
- For the wrong neighbors (NHs), find nearest H
"""
from transformers import (
    AutoConfig
)
from ModelForSentenceLevelRepresentation import get_model_for_representation
from EmbeddingPooler import EmbeddingPooler
from utils import parse_json
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
import pandas as pd
import os
import numpy as np
import torch

# The current script uses a trial that only uses 1 neighbor based
# on representations from the last layer only
# In this way, the NNs of the neighbor already obtained MUST be 
# not present in the set (no need to think about aggregation)
# that becomes an issue for K > 1 (and or multiple layers)

trial_config_path =  "./trial_configurations/toxigen/deberta-large-1-new/trial-126.json"
trial_config = parse_json(trial_config_path)
trial_path = "./output/toxigen/deberta-v3-large-1/DKNN/trial-126"

# prepare data
toxigen_data_dir = "./data/toxigen"
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"
toxigen_data = {
    "train": pd.read_csv(os.path.join(toxigen_data_dir, train_data_file)),
    "eval": pd.read_csv(os.path.join(toxigen_data_dir, eval_data_file)),
    "predict": pd.read_csv(os.path.join(toxigen_data_dir, test_data_file))
}
## split data into hate and nonhate examples based on labels
HATE = 1
NONHATE = 0
hate_mask = toxigen_data['train']['label'] == HATE
nonhate_mask = toxigen_data['train']['label'] == NONHATE
hate_examples = toxigen_data['train'][hate_mask]
nonhate_examples = toxigen_data['train'][nonhate_mask]
## read layer database
datapath = "./data/toxigen/deberta-large-1/trial-126"
train_representations = np.loadtxt(os.path.join(datapath, "layer_24.csv"), delimiter=",")
train_representations = train_representations[
    train_representations[:, -2].argsort() # sort by idx to get original training example 
]
train_representations = train_representations[:, :-2] # last two are tag and label
## read in neighbors id and distance for a particular trial
neighbor_indices = np.loadtxt(os.path.join(trial_path, "eval_neighbors.txt")).astype(int)[:, 1:]
neighbor_dists = np.loadtxt(os.path.join(trial_path, "eval_neighbor_dists.txt"))[:, 1:]

# set up KNN
config = AutoConfig.from_pretrained(
    trial_config['model_name_or_path'],
    num_labels=2,
    cache_dir=trial_config['cache_dir']
)
layer_dim=config.hidden_size
DKNN_args = torch.load(os.path.join(trial_path, "DKNN_args.bin")).__dict__
dist_funct = DistanceMetric.get_metric('minkowski', p=int(DKNN_args['minkowski_power']))
nonhate_tree = KDTree(
    train_reps[hate_mask][:, :layer_dim],
    leaf_size=DKNN_args['leaf_size'], 
    metric=dist_funct
)
hate_tree = KDTree(
    train_reps[nonhate_mask][:, :layer_dim],
    leaf_size=DKNN_args['leaf_size'],
    metric=dist_funct
)

# get evaluation 
batch_size = 8
layers_to_save = [-1]
poolers = [EmbeddingPooler().get("cls")]
model = get_model_for_representation(trial_config['model_name_or_path'])
eval_reps = model.encode_dataset(toxigen_data['eval'], batch_size,
                                 layers_to_save, poolers)[layers_to_save[-1]]

hate_distances, hate_neighbor_ids = hate_tree.query(eval_reps)
nonhate_distances, nonhate_neighbor_ids = nonhate_tree.query(eval_reps)

opp_neighbor_ids = []
opp_neighbor_dists = []
# for each neighbor, identify its nearest neighbor with the opposite label
for i, idx in tqdm(enumerate(neighbor_indices)):
    neighbor_label = toxigen_data['train'].iloc[i]['label']
    
    opp_neighbor_ids.append(
        hate_neighbor_ids[i, 0] if neighbor_label == HATE else nonhate_neighbor_ids[i, 0]
    )
    opp_neighbor_dists.append(
        hate_distances[i, 0] if neighbor_label == HATE else nonhate_distances[i, 0]
    )
opp_examples = toxigen_data['train'].iloc[opp_neighbor_ids, :]
ratios = neighbor_dists / np.array(opp_neighbor_dists)
