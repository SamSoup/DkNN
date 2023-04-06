"""
The purpose of this script is to run various wrapper boxes based on pre-computed
encodings from various sources
"""
from typing import Dict
from utils_copy import (
    compute_metrics,
    get_train_representations_from_file,
    get_actual_layers_to_save,
    get_actual_poolers_to_save,
    find_majority_batched,
    randargmax,
    mkdir_if_not_exists
)
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from classifiers.AgglomerativeClusteringClassifier import AgglomerativeClusteringClassifier
from sklearn.neighbors import KNeighborsClassifier
from classifiers.KMeansClassifier import KMeansClassifier
from classifiers.KMedoidsClassifier import KMedoidsClassifier
from classifiers.AgglomerativeClusteringClassifier import (
    AgglomerativeClusteringClassifier
)
from classifiers.LabelBiasedClusteringClassifier import (
    LabelBiasedClusteringClassifier
)
from classifiers.DecisionTreeProxyClassifier import (
    DecisionTreeProxyClassifier
)
from sklearn.model_selection import GridSearchCV
from ModelForSentenceLevelRepresentation import (
    ModelForSentenceLevelRepresentation, get_model_for_representation
)
from trial_configurations.configuration_constants import (
    DATASETS,
    DATA_PATH,
    LABELS,
    LAYER_CONFIGS,
    MODELS,
    MODEL_CONFIGS,
    MODEL_METADATAS,
    WORK_DIR,
    SEEDS,
    SPLITS
)

from sklearn import tree
from pprint import pprint
from tqdm.auto import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
import sys

WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
sys.path.append(WORK_DIR)


# these take several hyperparameters and honestly I do not know how to 
# parse all of them from commandline, so I just put some I think are reasonable
# ones for now
whiteboxes = {
    # 'SVM': (SVC(gamma='auto', class_weight='balanced', kernel="linear"), {}),
    'Decision_Tree': (DecisionTreeClassifier(max_depth=3, min_samples_leaf=10), {}),
    'KNN': (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), {}),
    # 'Decision Tree Proxy': (DecisionTreeProxyClassifier(max_depth=3), {}),
    # 'Label Biased Hierarchical Clustering': (
    #     LabelBiasedClusteringClassifier(max_depth=3), {}
    # ),
    # 'Agglomerative Hierarchical Clustering': (AgglomerativeClusteringClassifier(
    #     max_depth=1, linkage="ward", n_clusters=3), {'n_clusters': [2, 3], 
    #                                                  'max_depth': [1, 2, 3]}),
    'L_Means': (KMeansClassifier(), {}),
    'SVM': (LinearSVC(tol=1e-4, dual=False), {}),
    # 'K-medoids': (KMedoidsClassifier(), {'n_clusters': [2, 3, 10, 20, 30]})
}

results_df = pd.DataFrame(
    columns = [
        'Dataset', 'Model', 'Whitebox', 'Layers_Used', 'Poolers_Used', 'Seed',
        'predict_accuracy', 'predict_f1', 
        'predict_precision', 'predict_recall',
    ]
)

def save_model_metadata(
    clf, clf_name, model_name, save_path, layer_postfix
):
    mkdir_if_not_exists(save_path)
    with open(
        os.path.join(save_path, f'{model_name}_best_{clf_name}_{layer_postfix}'),
        'wb'
    ) as f:
        pickle.dump(clf, f)

# I have wrote this function but decided this is probably not used much
def save_knn_metadata(
    clf, X_test, model_name, save_path, layer_postfix=""
):
    """Helper to save the neighbor distances and indices"""
    
    neigh_dist, neigh_ind = clf.kneighbors(X_test)
    mkdir_if_not_exists(save_path)
    np.savetxt(os.path.join(save_path, f'{model_name}_knn_neighbors_dists_{layer_postfix}.txt'), neigh_dist, delimiter=",")
    np.savetxt(os.path.join(save_path, f'{model_name}_knn_neighbors_indices_{layer_postfix}.txt'), neigh_ind, delimiter=",")

def run_whiteboxes(X_train, X_eval, X_test, y_train, y_eval, whiteboxes, **kwargs):
    """
    This method runs all white box methods for representations retrieved from a particular layer
    """
    preds = {}
    is_multiclass = np.unique(y_train).size > 2
    scoring = "f1" if not is_multiclass else 'accuracy'
    for name, clf_set in tqdm(whiteboxes.items()):
        classifier, parameters = clf_set
        if parameters:
            clf = GridSearchCV(classifier, parameters, n_jobs=-1, scoring=scoring)
        else:
            if isinstance(classifier, KMeansClassifier):
                # manually set K-means # of clusters equal to the number of
                # unique label classes
                n_clusters = np.unique(y_train).size
                clf = KMeansClassifier(n_clusters=n_clusters)
            else:
                clf = classifier
        # for whiteboxes, use both training and validation data
        clf.fit(X_train, y_train)
        save_model_metadata(clf, name, **kwargs)
        # save neighbor distanes for KNN, for now just ignore
        # if name == 'KNN':
        #     if parameters:
        #         clf = KNeighborsClassifier(**clf.best_params_)
        #         clf.fit(X_train, y_train)
        #     save_knn_metadata(clf, X_test, **kwargs)
        y_pred = clf.predict(X_test)
        preds[name] = y_pred
    return preds

def add_to_dataframe(df, results: Dict[str, float], **kwargs):
    results.update(kwargs)
    # small helper to add to the current dataframe
    return pd.concat(
        [df, pd.DataFrame(results, index=[0])], ignore_index = True
    )

# compute the white box results for the different transformers
# knn results will be retrieved later from main results
output_filename = sys.argv[1]
data_path = os.path.join(
    WORK_DIR,
    "data/{dataset}/{model_name}/{mode}/{pooler_config}/layer_{layer}.csv"
)
for dataset, layer_available in tqdm(DATASETS.items()):
    save_whitebox_path = os.path.join(WORK_DIR, "results", dataset)
    model_configs = MODEL_CONFIGS[dataset]
    Y = LABELS[dataset]
    is_multiclass = np.unique(Y['y_train']).size > 2
    for model_name in tqdm(MODELS):
        num_layers = MODEL_METADATAS[model_name]['num_layers']
        pooler_configs = MODEL_METADATAS[model_name]['available_poolers']
        layers_to_save = get_actual_layers_to_save(layer_available, num_layers)
        models = model_configs[model_name] # same model but diff seeds
        for seed in SEEDS:
            model_name_with_seed = f"{model_name}-seed-{seed}"
            for pooler_config in tqdm(pooler_configs):
                save_whitebox_path_model = os.path.join(
                    save_whitebox_path, model_name, pooler_config
                )
                all_y_preds = {wrapper: {} for wrapper in whiteboxes}
                for layer in tqdm(layers_to_save):
                    X  = {
                        f'X_{split}': np.loadtxt(
                            data_path.format(
                                dataset=dataset,
                                mode=split, 
                                model_name=model_name_with_seed, 
                                pooler_config=pooler_config, 
                                layer=layer
                            ), 
                            delimiter=','
                        ) for split in SPLITS
                    }
                    y_preds = run_whiteboxes(
                        **X, y_train=Y['y_train'], y_eval=Y['y_eval'],
                        whiteboxes=whiteboxes,
                        model_name=model_name_with_seed,
                        save_path=save_whitebox_path_model, 
                        layer_postfix=f"layer{layer}"
                    )
                    for wrapper in whiteboxes:
                        all_y_preds[wrapper][layer] = y_preds[wrapper]
                # layer_config -> wrapper_name -> predictions_stacked
                layer_to_wrapper_to_preds = {}
                for layer_config in tqdm(LAYER_CONFIGS):
                    # subset the predictions based on actual layers used
                    layers_to_compute = get_actual_layers_to_save(layer_config, num_layers)
                    layer_to_wrapper_to_preds[layer_config] = {}
                    for wrapper in tqdm(whiteboxes):
                        y_preds_stacked = np.concatenate([
                            all_y_preds[wrapper][l].reshape(-1,1) 
                            for l in layers_to_compute
                        ], axis=1)
                        layer_to_wrapper_to_preds[layer_config][wrapper] = y_preds_stacked
                        y_pred = find_majority_batched(y_preds_stacked)
                        results = compute_metrics(Y['y_test'], y_pred, 'predict', is_multiclass)
                        results_df = add_to_dataframe(
                            results_df,
                            results,
                            Dataset=dataset,
                            Model=model_name,
                            Whitebox=wrapper,
                            Layers_Used=layer_config,
                            Seed=seed,
                            Poolers_Used=pooler_config
                        )
                    # also, run stacked predictions combining KNN, SVM, and DT
                    # ensemble = np.hstack([
                    #     layer_to_wrapper_to_preds[layer_config]['KNN'], 
                    #     layer_to_wrapper_to_preds[layer_config]['Decision Tree'], 
                    #     layer_to_wrapper_to_preds[layer_config]['SVM']
                    # ])
                    # y_pred = find_majority_batched(ensemble)
                    # results = compute_metrics(Y['y_test'], y_pred, 'predict', is_multiclass)
                    # results_df = add_to_dataframe(
                    #     results_df,
                    #     results,
                    #     Dataset=dataset,
                    #     Model=model_name,
                    #     Whitebox="Stacked",
                    #     Layers_Used=layer_config,
                    #     Poolers_Used=pooler_config
                    # )

# Compute the results for different baseline ('one layer')
# baseline_configs = {
#     'Glove-Twitter-200',
#     'FastText-300',
#     'Google-news-300',
#     'SentenceBert'
# }

# for dataset in tqdm(DATASETS):
#     Y = LABELS[dataset]
#     is_multiclass = np.unique(Y['y_train']).size > 2
#     for baseline in tqdm(baseline_configs):
#         data_path = os.path.join(WORK_DIR, "data", dataset, baseline)
#         # read in train, predict
#         save_whitebox_path = os.path.join(WORK_DIR, "results", dataset)
#         X = {
#             f'X_{split}': np.loadtxt(
#                 os.path.join(data_path, f"{split}_sentence_representations.txt"), 
#                 delimiter=","
#             )
#             for split in SPLITS
#         }
#         y_preds = run_whiteboxes(
#             **X, y_train=Y['y_train'], y_eval=Y['y_eval'],
#             whiteboxes=whiteboxes, 
#             model_name=baseline,
#             save_path=save_whitebox_path, 
#             layer_postfix=""
#         )
#         for wrapper, y_pred in tqdm(y_preds.items()):
#             results = compute_metrics(Y['y_test'], y_pred, 'predict', is_multiclass)
#             results_df = add_to_dataframe(
#                 results_df,
#                 results,
#                 Dataset=dataset,
#                 Model=baseline,
#                 Whitebox=wrapper,
#                 Layers_Used=None,
#                 Poolers_Used=None
#             )
#         ensemble = np.hstack([
#             y_preds['KNN'].reshape(-1, 1), 
#             y_preds['Decision Tree'].reshape(-1, 1), 
#             y_preds['SVM'].reshape(-1, 1)
#         ])
#         y_pred = find_majority_batched(ensemble)
#         results = compute_metrics(Y['y_test'], y_pred, 'predict', is_multiclass)
#         results_df = add_to_dataframe(
#             results_df,
#             results,
#             Dataset=dataset,
#             Model=baseline,
#             Whitebox="Stacked",
#             Layers_Used=None,
#             Poolers_Used=None
#         )

results_df.to_csv(
    path_or_buf=os.path.join(WORK_DIR, output_filename), 
    index=False, index_label=False, header=True
)
