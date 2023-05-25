from datasets import load_dataset
import sys
import os
import pandas as pd
import numpy as np

CACHE_DIR = "/work/06782/ysu707/ls6/.cache"
WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
sys.path.append(WORK_DIR)

# SEEDS = [
#     42, 365, 469, 4399, 3012023
# ]

SEEDS = [42]

SPLITS = ["train", "eval", "test"]

# dataset -> layers computed
DATASETS = {"toxigen": ["Last Only"], "esnli": ["Last Only"]}

LAYER_CONFIGS = ["All", "Embedding Only", "Embedding + Last", "Last Only"]

BASELINES = [
    "Glove-Twitter-200",
    "FastText-300",
    "Google-news-300",
    "SentenceBert",
]

METRICS = ["accuracy", "precision", "recall", "f1"]

MODELS = [
    "bart-large",
    "deberta-large",
    "flan-t5-large",
    # "t5-large"
    "llama7B"
]

WRAPPER_BOXES = ["KNN", "SVM", "Decision_Tree", "L_Means"]

# defines where models outputs are stored
MODEL_CONFIGS = {
    dataset: {
        model: {
            seed: os.path.join(
                WORK_DIR, "output", dataset, f"{model}-seed-{seed}"
            )
            for seed in SEEDS
        }
        for model in MODELS
    }
    for dataset in DATASETS
}

# defines metadata specific to each model
MODEL_METADATAS = {
    "bart-large": {
        "num_layers": 26,
        "available_poolers": [
            "mean_with_attention",
            # "mean_with_attention_and_eos"
        ],
    },
    "deberta-large": {
        "num_layers": 25,
        "available_poolers": [
            "mean_with_attention",
            # "mean_with_attention_and_cls"
        ],
    },
    "flan-t5-large": {
        "num_layers": 50,
        "available_poolers": [
            "mean_with_attention",
            # "encoder_mean_with_attention_and_decoder_flatten"
        ],
    },
    "t5-large": {
        "num_layers": 50,
        "available_poolers": [
            "mean_with_attention",
            # "encoder_mean_with_attention_and_decoder_flatten"
        ],
    },
    "llama7B": {
        "num_layers": 32,
        "available_poolers": ["mean_with_attention"],
    },
}

DATA = {
    dataset: load_dataset(f"Samsoup/{dataset}", use_auth_token=True)
    for dataset in DATASETS
}

# dataset name -> split -> labels
LABELS = {
    dataset: {
        split: np.array(DATA[dataset][split]["label"]) for split in SPLITS
    }
    for dataset in DATASETS
}

DATA_PATH = os.path.join(
    WORK_DIR, "data/{dataset}/{model}/{mode}/{pooler_config}/layer_{layer}.csv"
)
