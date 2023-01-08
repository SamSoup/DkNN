"""
Support vectors should be minimal / sparse / parsimonious

Given test set of N examples, and the set of all neighbors used to explain those N predictions, cardinality C = |neighbors| (smaller is better) 

Like SVM where support vectors are minimal training points defining decision hyperplane, these minimal neighbors are enough to fully reproduce what full training data set yields (same performance)
* We train neural model on full dataset to learn embedding (fine-tuning), so we needed other training points to get embedding, but not after that

More compact models generally better, less for user to have to remember maybe to internalize model
* If we used only this minimal set of nearest neighbors to train simple SVM or KNN model, how accurate would it be 
* Spirit of Cynthia Rudin, can we get a simpler (glassbox, inherently interpretable) model with the same performance

We could also look at loss curve as we progressively drop more and more neighbors, making model more compact/sparse
Similar to influence in how removing a given training point reduces loss over entire dataset (LOO analysis)

Usage:

python3 SimplifyModel.py ./result_locations_toxigen.json 16
"""

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig
)
from utils import parse_json, get_model_for_representation
from EmbeddingPooler import EmbeddingPooler
from ModelForSentenceLevelRepresentation import ModelForSentenceLevelRepresentation
from datasets import Dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
import sys
import os
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# toxigen_data_dir = "./data/toxigen"
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"
# toxigen_data = {
#     "train": pd.read_csv(os.path.join(toxigen_data_dir, train_data_file)),
#     "eval_": pd.read_csv(os.path.join(toxigen_data_dir, eval_data_file)),
#     "predict_": pd.read_csv(os.path.join(toxigen_data_dir, test_data_file))
# }
# model_name_or_path = "./output/toxigen/deberta-v3-large-1/checkpoint-200"

def get_unique_neighbor_indices(path_to_neighbors: str):
    neighbor_indices = np.loadtxt(path_to_neighbors).astype(int)
    return np.unique(neighbor_indices[:, 1])

def get_train_embeddings_given_indices(path, indices):
    train_representations = np.loadtxt(path, delimiter=",")
    train_representations = train_representations[
        train_representations[:, -2].argsort() # sort by idx to get original training example 
    ]
    train_representations = train_representations[:, :-2] # last two are tag and label
    return train_representations[indices] # get the hidden rep of examples

def compute_metrics(y_true, y_pred, prefix: str):
    # compute f1, accraucy, precision, recall
    return {
        f'{prefix}_f1': f1_score(y_true, y_pred),
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_precision': precision_score(y_true, y_pred),
        f'{prefix}_recall': recall_score(y_true, y_pred)
    }

result_files = sys.argv[1]
batch_size = int(sys.argv[2])
res_locations = parse_json(result_files)
p = EmbeddingPooler()
whiteboxes = [
    (GridSearchCV(Pipeline(steps=[("scaler", StandardScaler()), 
                                  ("svm", SVC(gamma='auto'))]),
                  {"svm__kernel": ['linear', 'poly', 'rbf', 'sigmoid']},
                  n_jobs=16), 'svm'),
    (GridSearchCV(Pipeline(steps=[("scaler", StandardScaler()), 
                                  ("knn", KNeighborsClassifier())]),
                  {"knn__n_neighbors": range(1, 9, 2)},
                  n_jobs=16),'knn'),
]

for loc_data in tqdm(res_locations):
    train_data = pd.read_csv(os.path.join(loc_data['data_dir'], train_data_file))
    # eval_labels = pd.read_csv(os.path.join(loc_data['data_dir'], eval_data_file))
    test_data = pd.read_csv(os.path.join(loc_data['data_dir'], test_data_file))
    # skipping first entry in os.walk since that's always the current directory
    result_paths = [subdir[0] for subdir in list(
        os.walk(os.path.join(loc_data['output_dir'], "DKNN"))
    )[1:]]
    model = get_model_for_representation(loc_data['model_path'])
    input_colname = loc_data['input_colname']
    for path in tqdm(result_paths):
        if "OLD" in path:
            continue # ignore old results
        # load the neighbors
        unique_train_neighbor_indices = get_unique_neighbor_indices(
            os.path.join(path, "eval_neighbors.txt")
        )
        train_subset = train_data.iloc[unique_train_neighbor_indices].reset_index(drop=True)
        # compute the unique subset of neighbors as identified by the validation set
        DKNN_args = torch.load(os.path.join(path, "DKNN_args.bin")).__dict__
        if DKNN_args['layers_to_save_desc'] != 'Last Only':
            continue # ignore non last layer only schemes
        database_path = DKNN_args['save_database_path']
        layers_to_save = DKNN_args['layers_to_save']
        poolers = list(map(p.get, DKNN_args['poolers_to_use']))
        ## obtain the minimal set of training + full test representations from LM
        test_embeddings_per_layer = model.encode_dataset(test_data, batch_size, 
                                                         layers_to_save, poolers)
        train_embeddings_per_layer = {}
        for layer in layers_to_save:
            embeddings_path = os.path.join(database_path, f"layer_{layer}.csv")
            train_embeddings = get_train_embeddings_given_indices(
                embeddings_path, unique_train_neighbor_indices
            )
            train_embeddings_per_layer[layer] = train_embeddings
        # train simpler model and observe results
        X_train, y_train = train_embeddings_per_layer[layers_to_save[-1]], train_subset['label'].to_numpy()
        X_test, y_test = test_embeddings_per_layer[layers_to_save[-1]], test_data['label'].to_numpy()
        results = {
            'DKNN_unique_train_examples_for_validation': unique_train_neighbor_indices.size 
        }
        for clf, prefix in whiteboxes:
            clf.fit(X_train, y_train)
            results.update(compute_metrics(y_test, clf.predict(X_test), prefix))
        with open(os.path.join(path, "simpler_models.json"), "w") as outfile:
            json.dump(results, outfile, indent = 4)
        # input()

# trial = "output/toxigen/deberta-v3-large-1/DKNN/trial-126"
# train_embeddings_path = "data/toxigen/deberta-large-1/trial-126/layer_24.csv"
# # Get all unique neighbors trial-126, Last Layer Only with K=1 and F1=0.6767486
# neighbor_indices = np.loadtxt(os.path.join(os.getcwd(), 
#                                            trial, 
#                                            "eval_neighbors.txt")).astype(int)
# unique_train_neighbor_indices = np.unique(neighbor_indices[:, 1])
# train_subset = toxigen_data['train'].iloc[unique_train_neighbor_indices].reset_index(drop=True)

# # load the layer representations

# ## take from original model - get from saved training examples
# train_representations = np.loadtxt(os.path.join(os.getcwd(), train_embeddings_path), delimiter=",")
# train_representations = train_representations[
#     train_representations[:, -2].argsort() # sort by idx to get original training example 
# ]
# train_representations = train_representations[:, :-2] # last two are tag and label
# train_embeddings = train_representations[unique_train_neighbor_indices] # get the hidden rep of examples
# test_embeddings = m.encode_dataset(toxigen_data['predict_'], 8)[24] # last layer

# ## take from sentence bert 
# model = SentenceTransformer('all-mpnet-base-v2')
# strain_embeddings = model.encode(train_subset['text'])
# stest_embeddings = model.encode(toxigen_data['predict_']['text'])

# # baseline with everything
# all_train_embeddings = model.encode(toxigen_data['train']['text'])

# def train_simpler_model(X_train, y_train, X_test, y_test):
#     for clf in [
#         make_pipeline(StandardScaler(), SVC(gamma='auto')), 
#         make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
#     ]:
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)

#         print("F1: ", f1_score(y_test, y_pred))
#         print("Accuracy: ", accuracy_score(y_test, y_pred))

# train_simpler_model(all_train_embeddings, toxigen_data['train']['label'], stest_embeddings, toxigen_data['predict_']['label'])
# train_simpler_model(train_embeddings, train_subset['label'], test_embeddings, toxigen_data['predict_']['label'])
# train_simpler_model(strain_embeddings, train_subset['label'], stest_embeddings, toxigen_data['predict_']['label'])
