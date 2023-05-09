from binomial import compute_binomial_p_value
from tqdm.auto import tqdm
from datasets import load_dataset
from constants import DATASETS, MODELS, WORK_DIR, SEEDS, METRICS, MODEL_METADATAS, WRAPPER_BOXES
from utils import load_predictions
import pandas as pd
import numpy as np
import os

# for now, only do seed 42, last layer, mean_with_attention
SEEDS = list(filter(lambda x: x == 42, SEEDS))
pooler_config = "mean_with_attention"

def create_result_df(models, metrics, whiteboxes):
    """
    Creates a empty dataframe with index = whitebox classifier names,
    and a multilevel column index of models * metrics
    """
    return pd.DataFrame(
        np.nan, index=whiteboxes, 
        columns=pd.MultiIndex.from_product([models, metrics], names=['models', 'metrics'])
    )

for dataset in tqdm(DATASETS, desc="datasets"):
    data = load_dataset(f"Samsoup/{dataset}", use_auth_token=True)
    y_test = np.array(data['test']['label'])
    is_multiclass = np.unique(y_test).size > 2
    whitebox_preds = pd.read_pickle(
        os.path.join(WORK_DIR, 'data', dataset, f'{dataset}_wrapperbox_predictions.pkl')
    )
    # result file layout: 
    results = create_result_df(MODELS, METRICS, WRAPPER_BOXES)
    for model in tqdm(MODELS, desc="models"):
        for seed in SEEDS:
            model_full = f"{model}-seed-{seed}"
            layer = MODEL_METADATAS[model]['num_layers']-1
            # load model's original predictions
            original_preds = np.array(load_predictions(WORK_DIR, dataset, model_full))
            # for each metric, compute sig test for the model to each wrapper box
            for whitebox in tqdm(WRAPPER_BOXES, desc="whiteboxes"):
                p_values = compute_binomial_p_value(
                    original_preds,
                    whitebox_preds.loc[model][whitebox],
                    y_test,
                    is_multiclass
                )
                for metric in METRICS:
                    results.loc[whitebox][model][metric] = p_values[f'test_{metric}']
    results.to_pickle(f"{dataset}_binomial_significance_tests.pkl")