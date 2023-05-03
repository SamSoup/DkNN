from bootstrap import compute_p_value
from tqdm.auto import tqdm
from datasets import load_dataset
from constants import DATASETS, MODELS, WORK_DIR, SEEDS, METRICS, MODEL_METADATAS, WRAPPER_BOXES
from utils import load_predictions, load_representation
import itertools
import pandas as pd
import numpy as np

# for now, only do seed 42, last layer, mean_with_attention
SEEDS = list(filter(lambda x: x == 42, SEEDS))
pooler_config = "mean_with_attention"

def create_result_df(models, metrics, whiteboxes):
    """
    Creates a empty dataframe with index = whitebox classifier names,
    and a multilevel column index of models * metrics
    """
    df = pd.DataFrame(np.nan, index=whiteboxes)
    df.columns = pd.MultiIndex.from_product([models, metrics], names=['models', 'metrics'])
    return df

for dataset in tqdm(DATASETS, desc="datasets"):
    data = load_dataset(f"Samsoup/{dataset}", use_auth_token=True)
    y_test = np.array(data['label'])
    # result file layout: 
    results = create_result_df(MODELS, METRICS, WRAPPER_BOXES)
    for model in tqdm(MODELS, desc="models"):
        for seed in SEEDS:
            X_test = load_representation(
                WORK_DIR, dataset, model_full, "test", pooler_config, layer
            )
            model_full = f"{model}-seed-{seed}"
            layer = MODEL_METADATAS[model]['num_layers']-1
            # load model's original predictions
            original_preds = np.array(load_predictions(WORK_DIR, dataset, model_full))
            # load model's whitebox predictions
            whitebox_preds = pd.read_pickle(f'{dataset}_predictions.pkl')
            # for each metric, compute sig test for the model to each wrapper box
            for whitebox in tqdm(WRAPPER_BOXES, desc="whiteboxes"):
                deltas = compute_p_value(
                    original_preds,
                    whitebox_preds[model][whitebox],
                    X_test,
                    y_test,
                    size=10000, iterations=1e5, seed=42
                )
                for metric in deltas:
                    results.loc[whitebox][model][metric] = deltas[metric]['p-value']
    results.to_pickle(f"{dataset}_significance_tests.pkl")
