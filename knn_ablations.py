import sys
WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
sys.path.append(WORK_DIR)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsClassifier
from trial_configurations.configuration_constants import (
    DATASETS,
    DATA_PATH,
    LABELS,
    MODELS,
    MODEL_CONFIGS,
    MODEL_METADATAS,
    WORK_DIR,
    SPLITS
)

from utils_copy import (
    add_to_dataframe,
    compute_metrics,
    get_train_representations_from_file,
    get_actual_layers_to_save,
    get_actual_poolers_to_save,
    find_majority_batched,
    randargmax,
    mkdir_if_not_exists
)

from itertools import product

def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())], columns=dictionary.keys())

trials = expand_grid({
    'n_neighbors': [1, 3, 5, 7, 9, 11] + list(range(101, 1002, 100)),
    'weights': ["uniform", "distance"],
    'metric': ["minkowski", "cosine"]
})

pooler_config = "mean_with_attention"
# only ablate the last layer, mean_with_attention for now
for dataset in tqdm(DATASETS, desc="Computing Results Per Dataset"):
    Y = LABELS[dataset]
    is_multiclass = np.unique(Y['y_train']).size > 2
    if is_multiclass:
        results_df = pd.DataFrame(
            columns = [
                'Dataset', 
                'Model', 
                'n_neighbors',
                'weights',
                'metric',
                'predict_accuracy', 
                'predict_micro_f1', 
                'predict_micro_precision', 
                'predict_micro_recall', 
                'predict_macro_f1', 
                'predict_macro_precision', 
                'predict_macro_recall', 
                'predict_weighted_f1', 
                'predict_weighted_precision', 
                'predict_weighted_recall', 
                'predict_0_f1', 
                'predict_0_precision', 
                'predict_0_recall', 
                'predict_1_f1', 
                'predict_1_precision', 
                'predict_1_recall', 
                'predict_2_f1', 
                'predict_2_precision',
                'predict_2_recall'
            ]
        )
    else:
        results_df = pd.DataFrame(
            columns = [
                'Dataset', 
                'Model', 
                'n_neighbors',
                'weights',
                'metric',
                'predict_accuracy', 'predict_f1', 
                'predict_precision', 'predict_recall',
            ]
        )

    for model in tqdm(MODELS, desc="Computing Results Per Model"):
        layer = get_actual_layers_to_save(
            'Last Only', 
            MODEL_METADATAS[model]['num_layers']
        )[0]
        X = {
            f'X_{split}': np.loadtxt(
                DATA_PATH.format(
                    dataset=dataset,
                    mode=split, 
                    model=model, 
                    pooler_config=pooler_config, 
                    layer=layer
                ), 
                delimiter=','
            )
            for split in SPLITS
        }
        for _, row in tqdm(trials.iterrows()):
            params = row.to_dict()
            knn = KNeighborsClassifier(**params)
            knn.fit(np.vstack([X['X_train'], X['X_eval']]), 
                    np.concatenate((Y['y_train'], Y['y_eval'])))
            y_pred = knn.predict(X['X_test'])
            results = compute_metrics(y_pred, Y['y_test'], "predict", is_multiclass)
            results_df = add_to_dataframe(
                results_df, 
                results,
                Dataset=dataset, 
                Model=model,
                **params
            )
# results_df
results_df.to_csv("esnli_knn_ablations", header=True, index=False)
