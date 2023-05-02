"""
The purpose of this file is to grab the predictions from saved whiteboxes, and save them in a file
"""

from utils import load_whitebox, load_representation
from tqdm.auto import tqdm
from constants import DATASETS, WORK_DIR, MODELS, SEEDS, MODEL_METADATAS, WRAPPER_BOXES
import pandas as pd
import numpy as np

# for now, only do seed 42, last layer, mean_with_attention
SEEDS = list(filter(lambda x: x == 42, SEEDS))
pooler_config = "mean_with_attention"

# load representations from datasets
for dataset in tqdm(DATASETS, desc="datasets"):
    # initiate pandas dataframe for storage
    predictions = pd.DataFrame(np.nan, index=MODELS, columns=WRAPPER_BOXES).astype(object)
    for model in tqdm(MODELS, desc="Models"):
        for seed in SEEDS:
            model_full = f"{model}-seed-{seed}"
            layer = MODEL_METADATAS[model]['num_layers']-1
            X_test = load_representation(
                WORK_DIR, dataset, model_full, "test", pooler_config, layer
            )
            for whitebox in tqdm(WRAPPER_BOXES, "whiteboxes"):
                clf = load_whitebox(
                    WORK_DIR, dataset, model_full, pooler_config, whitebox, layer
                )
                preds = clf.predict(X_test)
                predictions.loc[model][whitebox] = preds
    predictions.to_csv(f"{dataset}_predictions.csv", index=True, header=True)
