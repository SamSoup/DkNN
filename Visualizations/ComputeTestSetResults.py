from tqdm.auto import tqdm
from datasets import load_dataset
from constants import DATASETS, MODELS, WORK_DIR, SEEDS, METRICS, MODEL_METADATAS, WRAPPER_BOXES
from utils import load_predictions, compute_metrics
import pandas as pd
import numpy as np
import os

# for now, only do seed 42, last layer, mean_with_attention
SEEDS = list(filter(lambda x: x == 42, SEEDS))
pooler_config = "mean_with_attention"

def create_result_df(models, metrics, classifiers):
    """
    Creates a empty dataframe with index = baseline + whitebox classifier names,
    and a multilevel column index of models * metrics
    """
    return pd.DataFrame(
        np.nan, index=classifiers, 
        columns=pd.MultiIndex.from_product([models, metrics], names=['models', 'metrics'])
    )

classifiers = ['original'] + WRAPPER_BOXES
for dataset in tqdm(DATASETS, desc="datasets"):
    data = load_dataset(f"Samsoup/{dataset}", use_auth_token=True)
    y_test = np.array(data['test']['label'])
    is_multiclass = np.unique(y_test).size > 2
    whitebox_preds = pd.read_pickle(
        os.path.join(WORK_DIR, 'data', dataset, f'{dataset}_wrapperbox_predictions.pkl')
    )
    # result file layout: 
    results = create_result_df(MODELS, METRICS, classifiers)
    for model in tqdm(MODELS, desc="models"):
        for seed in SEEDS:
            model_full = f"{model}-seed-{seed}"
            layer = MODEL_METADATAS[model]['num_layers']-1
            # for each metric, compute metrics for the model to each wrapper box
            for clf in tqdm(classifiers, desc="original+whiteboxes"):
                if clf == "original":
                    preds = np.array(load_predictions(WORK_DIR, dataset, model_full))
                else:
                    preds = whitebox_preds.loc[model][clf]
                metrics = compute_metrics(
                    y_test, preds,
                    prefix="test", is_multiclass=is_multiclass
                )
                for metric in METRICS:
                    results.loc[clf][model][metric] = metrics[f'test_{metric}']
    results.to_pickle(
        os.path.join(WORK_DIR, 'data', dataset, f"{dataset}_test_set_results.pkl")
    )
