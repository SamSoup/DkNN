import sys
WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
sys.path.append(WORK_DIR)
from utils_copy import compute_metrics
from pprint import pprint
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial.distance import cdist 
from trial_configurations.configuration_constants import (
    toxigen_data,
    esnli_data,
    DATASETS,
    DATA_PATH,
    LABELS,
    LAYER_CONFIGS,
    MODELS,
    MODEL_CONFIGS,
    MODEL_METADATAS,
    WORK_DIR,
    SPLITS
)
NUM_EXPLANATIONS = 5

for mode in esnli_data:
    esnli_data[mode]['text'] = [ x + y for x, y in zip(esnli_data[mode]['premise'], esnli_data[mode]['hypothesis'])] 
    
dataset = "esnli"
is_multilclass = dataset == "esnli"
data_path = os.path.join(WORK_DIR, "data/{dataset}/{model_name}/{mode}/{pooler_config}/layer_{layer}.csv")
model_config = "deberta-large-seed-42"
pooler_config = "mean_with_attention"
layer = 24
X_train = np.loadtxt(
    data_path.format(mode='train', dataset=dataset, model_name=model_config, 
                     pooler_config=pooler_config, layer=layer), delimiter=','
)
X_eval = np.loadtxt(
    data_path.format(mode='eval', dataset=dataset, model_name=model_config,
                     pooler_config=pooler_config, layer=layer), delimiter=','
)
X_test = np.loadtxt(
    data_path.format(mode='test', dataset=dataset, model_name=model_config,
                     pooler_config=pooler_config, layer=layer), delimiter=','
)
y_train = LABELS[dataset]['y_train']
y_eval = LABELS[dataset]['y_eval']
y_test = LABELS[dataset]['y_test']

from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

with open(
    os.path.join(WORK_DIR, 'results', 'esnli', 'deberta-large', 'mean_with_attention',
                 f'deberta_large_seed_42_best_svm_layer24'),
    'wb'
) as f:
    pickle.dump(svm, f)

