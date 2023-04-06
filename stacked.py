from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import sys

WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
sys.path.append(WORK_DIR)

from trial_configurations.configuration_constants import (
    DATASETS,
    DATA_PATH,
    LABELS,
    LAYER_CONFIGS,
    MODELS,
    MODEL_CONFIGS,
    MODEL_METADATAS,
    WORK_DIR,
    SPLITS
)
from utils_copy import compute_metrics

K = 5
dataset = "esnli"
data_path = os.path.join(WORK_DIR, "data/{dataset}/{model_name}/{mode}/{pooler_config}/layer_{layer}.csv")
# data_path = os.path.join(work_dir, "output/toxigen/{model_name}/{mode}/layer_reps/{pooler_config}/layer{layer}.csv")
model_config = "flan-t5-large"
pooler_config = "mean_with_attention"
# pooler_config = "encoder_mean_with_attention_and_decoder_flatten"
# layer = 24
layer = 49
X_train = np.loadtxt(
    data_path.format(mode='train', dataset=dataset, model_name=model_config, 
                     pooler_config=pooler_config, layer=layer), delimiter=','
)
X_eval = np.loadtxt(
    data_path.format(mode='eval', dataset=dataset, model_name=model_config, 
                     pooler_config=pooler_config, layer=layer), delimiter=','
)
X_test = np.loadtxt(
    data_path.format(mode='test', dataset=dataset, model_name=model_config,
                     pooler_config=pooler_config, layer=layer), delimiter=','
)
y_train = LABELS[dataset]['y_train']
y_eval = LABELS[dataset]['y_eval']
y_test = LABELS[dataset]['y_test']

K = 5
knn = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
knn.fit(np.vstack([X_train, X_eval]), np.concatenate((y_train, y_eval)))
knn_preds = knn.predict(X_test)
knn_preds_proba = knn.predict_proba(X_test)
print(compute_metrics(knn_preds, y_test, 'predict', True))

from sklearn.model_selection import GridSearchCV

# parameters = {
#     'max_depth': [1, 3, 5, 10, 100, None],
#     'min_samples_split':[50, 100], 
#     'min_samples_leaf':[10, 50]
# }
# dtree = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=16)
dtree = DecisionTreeClassifier(max_depth=3)
# dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(np.vstack([X_train, X_eval]), np.concatenate((y_train, y_eval)))
dtree_preds = dtree.predict(X_test)
dtree_preds_proba = dtree.predict_proba(X_test)
print(compute_metrics(dtree_preds, y_test, 'predict', True))

svm = LinearSVC(tol=1e-4)
# svm = SVC(random_state=42, gamma='auto', kernel='linear', class_weight='balanced', probability=True)
svm.fit(np.vstack([X_train, X_eval]), np.concatenate((y_train, y_eval)))
svm_preds = svm.predict(X_test)
# svm_preds_proba = svm.predict_proba(X_test)
print(compute_metrics(svm_preds, y_test, 'predict', True))

with open(
    os.path.join(WORK_DIR, f'{model_config}_best_SVM_layer_{layer}.pkl'),
    'wb'
) as f:
    pickle.dump(svm, f)

#y_preds_proba = knn_preds_proba.reshape(-1,2) + dtree_preds_proba.reshape(-1,2) + rf_preds_proba.reshape(-1,2)
# y_preds_proba = np.hstack([knn_preds_proba.reshape(-1,1), ])
# y_preds = np.hstack([knn_pred.reshape(-1,1), dtree_pred.reshape(-1,1), svm_pred.reshape(-1,1), rf_preds.reshape(-1, 1)])
#y_preds_proba.shape

# from np_utils import randargmax

# y_preds = randargmax(y_preds_proba)
# print(compute_metrics(y_preds, y_test, 'predict', True))

from scipy import stats
# stack the prediction from the three classifer
def majority_vote(y_preds):
    return stats.mode(y_preds.transpose(), keepdims=False)[0].squeeze()
final_preds = majority_vote(y_preds)
print(compute_metrics(final_preds, y_test, 'predict', True))