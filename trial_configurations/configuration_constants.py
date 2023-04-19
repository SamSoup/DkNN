import sys
import os
import pandas as pd
import numpy as np

CACHE_DIR = "/work/06782/ysu707/ls6/.cache"
WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
sys.path.append(WORK_DIR)

SEEDS = [
    42, # 365, 469, 4399, 3012023
]

SPLITS = ['train', 'eval', 'test']

# dataset -> layers computed
DATASETS = {
    # 'toxigen': 'All',
    'esnli': 'Last Only'
}

LAYER_CONFIGS = [
    # "All", 
    # "Embedding Only", 
    # "Embedding + Last", 
    "Last Only"
]

BASELINES = [
    'Glove-Twitter-200',
    'FastText-300',
    'Google-news-300',
    'SentenceBert'
]

MODELS = [
    # "bart-large",
    # "deberta-large",
    "flan-t5-large"
    # "t5-large"
]

# defines where models outputs are stored
MODEL_CONFIGS = {
    dataset: {
        model: {
            seed: os.path.join(WORK_DIR, "output", dataset, f"{model}-seed-{seed}")
            for seed in SEEDS
        }
        for model in MODELS 
    }
    for dataset in DATASETS
}

# defines metadata specific to each model
MODEL_METADATAS = {
    "bart-large": {
        'num_layers': 26, 
        'available_poolers': [
            "mean_with_attention", 
            # "mean_with_attention_and_eos"
            ]
        },
    "deberta-large": {
        'num_layers': 25, 
        'available_poolers': [
            "mean_with_attention", 
            # "mean_with_attention_and_cls"
            ]
        },
    "flan-t5-large": {
        'num_layers': 50, 
        'available_poolers': [
            "mean_with_attention", 
            # "encoder_mean_with_attention_and_decoder_flatten"
            ]
        },
    "t5-large": {
        'num_layers': 50, 
        'available_poolers': [
            "mean_with_attention", 
            # "encoder_mean_with_attention_and_decoder_flatten"
            ]
        }
}

# dataset name -> split -> labels
LABELS = {}
for dataset in DATASETS:
    data_dir = os.path.join(WORK_DIR, "data", dataset)
    LABELS[dataset] = {
        f'y_{split}': pd.read_csv(
                os.path.join(data_dir, f"{split}_data.csv")
            )['label'].to_numpy()
        for split in SPLITS
    }

DATA_PATH = os.path.join(
    WORK_DIR,
    "data/{dataset}/{model}/{mode}/{pooler_config}/layer_{layer}.csv"
)

train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"

toxigen_data_dir = f"{WORK_DIR}/data/toxigen"
toxigen_data = {
    "train": pd.read_csv(os.path.join(toxigen_data_dir, train_data_file)),
    "eval": pd.read_csv(os.path.join(toxigen_data_dir, eval_data_file)),
    "test": pd.read_csv(os.path.join(toxigen_data_dir, test_data_file))
}

esnli_data_dir = f"{WORK_DIR}/data/esnli"
esnli_data = {
    "train": pd.read_csv(os.path.join(esnli_data_dir, train_data_file)),
    "eval": pd.read_csv(os.path.join(esnli_data_dir, eval_data_file)),
    "test": pd.read_csv(os.path.join(esnli_data_dir, test_data_file))
}
