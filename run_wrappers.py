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
from sklearn.tree import DecisionTreeClassifier
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

# boiler plate code to set up 
# can make this a cli tool, but oh well
datasets = [
    'toxigen', 
    # 'esnli'
]
splits = ['train', 'eval', 'test']

# dataset name -> split -> labels
labels = {}
for dataset in datasets:
    data_dir = os.path.join(WORK_DIR, "data", dataset)
    labels[dataset] = {
        split: pd.read_csv(
                os.path.join(data_dir, f"{split}_data.csv")
            )['label'].to_numpy()
        for split in splits
    }

layer_configs = [
    "All", 
    "Embedding Only", 
    "Embedding + Last", 
    "Last Only"
]

# these take several hyperparameters and honestly I do not know how to 
# parse all of them from commandline, so I just put some I think are reasonable
# ones for now
whiteboxes = {
    'SVM': (SVC(gamma='auto', class_weight='balanced', kernel="linear"), {}),
    'Decision Tree': (DecisionTreeClassifier(max_depth=3), {}),
    'KNN': (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), {}),
    'Decision Tree Proxy': (DecisionTreeProxyClassifier(max_depth=3), {}),
    'Label Biased Hierarchical Clustering': (
        LabelBiasedClusteringClassifier(max_depth=3), {}
    ),
    'K-means': (KMeansClassifier(), {'n_clusters': [2, 3]}),
    'K-medoids': (KMedoidsClassifier(), {'n_clusters': [1, 10, 20, 30]})
}

results_df = pd.DataFrame(
    columns = [
        'Model', 'Whitebox', 'Layers_Used', 
        'Poolers_Used',
        'predict_accuracy', 'predict_f1', 
        'predict_precision', 'predict_recall',
    ]
)

# unfortunately hard codes here
model_configs = {
    "bart-large": ("./output/toxigen/bart-large", 26, 
                     ["mean_with_attention", #"mean_with_attention_and_eos"
                      ]),
    "deberta-large": ("./output/toxigen/deberta-large", 25, 
                        ["mean_with_attention", #"mean_with_attention_and_cls"
                         ]),
    "flan-t5-large": ("./output/toxigen/flan-t5-large", 50, 
                        ["mean_with_attention", # "encoder_mean_with_attention_and_decoder_flatten"
                         ])
}

def save_model_metadata(
    clf, clf_name, model_config, save_path, layer_postfix
):
    mkdir_if_not_exists(save_path)
    with open(
        os.path.join(save_path, f'{model_config}_best_{clf_name}_{layer_postfix}'),
        'wb'
    ) as f:
        pickle.dump(clf, f)

# I have wrote this function but decided this is probably not used much
def save_knn_metadata(
    clf, X_test, model_config, save_path, layer_postfix=""
):
    """Helper to save the neighbor distances and indices"""
    
    neigh_dist, neigh_ind = clf.kneighbors(X_test)
    mkdir_if_not_exists(save_path)
    np.savetxt(os.path.join(save_path, f'{model_config}_knn_neighbors_dists_{layer_postfix}.txt'), neigh_dist, delimiter=",")
    np.savetxt(os.path.join(save_path, f'{model_config}_knn_neighbors_indices_{layer_postfix}.txt'), neigh_ind, delimiter=",")

def run_whiteboxes(X_train, X_eval, X_test, y_train, y_eval, whiteboxes, **kwargs):
    """
    This method runs all white box methods for representations retrieved from a particular layer
    """
    preds = {}
    for name, clf_set in whiteboxes.items():
        classifier, parameters = clf_set
        if parameters:
            clf = GridSearchCV(classifier, parameters, n_jobs=-1, scoring="f1")
        else:
            clf = classifier
        # for whiteboxes, use both training and validation data
        clf.fit(np.vstack([X_train, X_eval]), np.concatenate((y_train, y_eval)))
        # save_model_metadata(clf, name, **kwargs)
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
output_path = sys.argv[1]
data_path = os.path.join(
    WORK_DIR,
    "data/{dataset}/{model_name}/{mode}/{pooler_config}/layer_{layer}.csv"
)
for dataset in tqdm(datasets):
    save_whitebox_path = os.path.join(WORK_DIR, "results", dataset)
    for model_config in tqdm(model_configs):
        model_name_or_path, num_layers, pooler_configs = model_configs[model_config]
        # layers_to_save = get_actual_layers_to_save('Last Only', num_layers)
        # just do all layers
        layers_to_save = get_actual_layers_to_save('All', num_layers)
        for pooler_config in tqdm(pooler_configs):
            save_whitebox_path = os.path.join(
                save_whitebox_path, model_config, pooler_config
            )
            all_y_preds = {wrapper: [] for wrapper in whiteboxes}
            for layer in tqdm(layers_to_save):
                X , Y = {}, {}
                for split in splits:
                    X[f'X_{split}'] = np.loadtxt(
                        data_path.format(
                            dataset=dataset,
                            mode=split, 
                            model_name=model_config, 
                            pooler_config=pooler_config, 
                            layer=layer
                        ), 
                        delimiter=','
                    )
                    if split != "test":
                        # do not add y_test
                        Y[f'y_{split}'] = labels[dataset][split]
                y_preds = run_whiteboxes(
                    **X, **Y,
                    whiteboxes=whiteboxes,
                    model_config=model_config,save_path=save_whitebox_path, 
                    layer_postfix=f"layer{layer}"
                )
                for wrapper in whiteboxes:
                    all_y_preds[wrapper].append(y_preds[wrapper])
            # subset the predictions based on actual layers used
            for layer_config in tqdm(layer_configs):
                layers_to_save = get_actual_layers_to_save(layer_config, num_layers)
                for wrapper in tqdm(whiteboxes):
                    # y_pred = all_y_preds[wrapper][-1]
                    y_preds_stacked = np.concatenate([
                        all_y_preds[wrapper][l].reshape(-1,1) 
                        for l in layers_to_save
                    ], axis=1)
                    all_y_preds[wrapper]['all'] = y_preds_stacked
                    y_pred = find_majority_batched(y_preds_stacked)
                    results = compute_metrics(y_test, y_pred, 'predict')
                    results_df = add_to_dataframe(
                        results_df, 
                        Model=model_config,
                        Whitebox=wrapper,
                        Layers_Used=layer_config,
                        Poolers_Used=pooler_config
                    )
                # also, run stacked predictions combining KNN, SVM, and DT
                stacked = np.hstack([
                    all_y_preds['KNN']['all'], 
                    all_y_preds['Decision Tree']['all'], 
                    all_y_preds['SvM']['all']
                ])
                results_df = add_to_dataframe(
                    results_df,
                    results,
                    Model=model_config,
                    Whitebox="Stacked",
                    Layers_Used=layer_config,
                    Poolers_Used=pooler_config
                )

# Compute the results for different baseline ('one layer')
baseline_configs = {
    'Glove-Twitter-200',
    'FastText-300',
    'Google-news-300',
    'SentenceBert'
}

for dataset in tqdm(datasets):
    for baseline in tqdm(baseline_configs):
        data_path = os.path.join(WORK_DIR, "data", baseline)
        # read in train, predict
        save_whitebox_path = os.path.join(WORK_DIR, "results", dataset)
        X , Y = {}, {}
        for split in splits:
            X[f'X_{split}'] = np.loadtxt(
                os.path.join(data_path, "train_sentence_representations.txt"), 
                delimiter=","
            )
            if split != "test":
                # do not add y_test
                Y[f'y_{split}'] = labels[dataset][split]
        y_preds = run_whiteboxes(
            **X, **Y,
            whiteboxes=whiteboxes, 
            model_config=model_config,
            save_path=save_whitebox_path, 
            layer_postfix=f"layer{layer}"
        )
        for wrapper, y_pred in tqdm(y_preds.items()):
            results = compute_metrics(y_test, y_pred, 'predict')
            results_df = add_to_dataframe(
                results_df,
                results,
                Model=baseline,
                Whitebox=wrapper,
                Layers_Used=None,
                Poolers_Used=None
            )

results_df.to_csv(
    path_or_buf=os.path.join(output_path, "whitebox_results.csv"), 
    index=False, index_label=False, header=True
)
