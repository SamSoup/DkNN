"""
The only difference between DKNN and KNN is that DKNN leverages 
the learned sentence-level representations from a LM that's fine 
tuned to the classification task at hand. What if said fine tuning is 
not necessary? Can we take some pre-trained embeddings 
(e.g. word2vec, SentenceBert) and just directly use that 
encoding followed by DKNN? 
export TOKENIZERS_PARALLELISM=false
"""
from ModelForSentenceLevelRepresentation import get_model_for_representation
from EmbeddingPooler import EmbeddingPooler
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV
import gensim
import gensim.downloader as api
import nltk
import json
import pandas as pd
import numpy as np
import os

# get data
toxigen_data_dir = "./data/toxigen"
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"
toxigen_data = {
    "train": pd.read_csv(os.path.join(toxigen_data_dir, train_data_file)),
    "eval": pd.read_csv(os.path.join(toxigen_data_dir, eval_data_file)),
    "predict": pd.read_csv(os.path.join(toxigen_data_dir, test_data_file))
}

knn = GridSearchCV(
    Pipeline(steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
    {"knn__n_neighbors": range(1, 9, 2)},
    n_jobs=16
)

def compute_metrics(y_true, y_pred, prefix: str):
    # compute f1, accraucy, precision, recall
    return {
        f'{prefix}_f1': f1_score(y_true, y_pred),
        f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}_precision': precision_score(y_true, y_pred),
        f'{prefix}_recall': recall_score(y_true, y_pred)
    }

all_configs = {
    'do_DKNN': False,
    'neighbor_method': None,
    'prediction_method': None
}

configurations = []

# large language model, without fine tunining
batch_size = 8
layers_to_save = [-1]
poolers = [EmbeddingPooler().get("mean_with_attention")]

bart = get_model_for_representation("facebook/bart-large")
deberta = get_model_for_representation("microsoft/deberta-v3-large")

configurations.append({
    'Dataset': 'toxigen',
    'Model': 'baseline_bart',
    'X_train': bart.encode_dataset(toxigen_data['train'], batch_size, 
                                   layers_to_save, poolers)[layers_to_save[-1]],
    'X_test': bart.encode_dataset(toxigen_data['predict'], batch_size, 
                                   layers_to_save, poolers)[layers_to_save[-1]]
})

configurations.append({
    'Dataset': 'toxigen',
    'Model': 'baseline_deberta',
    'X_train': deberta.encode_dataset(toxigen_data['train'], batch_size, 
                                        layers_to_save, poolers)[layers_to_save[-1]],
    'X_test': deberta.encode_dataset(toxigen_data['predict'], batch_size, 
                                        layers_to_save, poolers)[layers_to_save[-1]]
})

# SentenceBert 
sBert = SentenceTransformer('all-mpnet-base-v2')
configurations.append({
    'Dataset': 'toxigen',
    'Model': 'baseline_sentencebert',
    'X_train': sBert.encode(toxigen_data['train']['text']),
    'X_test': sBert.encode(toxigen_data['predict']['text'])
})

# X_train, y_train = sBert.encode(toxigen_data['train']['text']), toxigen_data['train']['label']
# X_test, y_test = sBert.encode(toxigen_data['predict']['text']), toxigen_data['predict']['label']

# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# sentencebert_results = compute_metrics(y_test, y_pred, "predict")
# sentencebert_results['Dataset'] = 'toxigen'
# sentencebert_results['Model'] = 'baseline_sentencebert'
# sentencebert_results['do_DKNN'] = False
# sentencebert_results['neighbor_method'] = None
# sentencebert_results['prediction_method'] = None
# with open('baseline_sentencebert_results.json', 'w') as outfile:
#     json.dump(sentencebert_results, outfile, indent = 4)

# Word2vec
# CBOW (Continuous Bag of Words): CBOW model predicts the current word given 
# context words within a specific window. The input layer contains the context 
# words and the output layer contains the current word. The hidden layer 
# contains the number of dimensions in which we want to represent the 
# current word present at the output layer. 

# for each sentence, tokenize it
nltk.download('punkt')

# https://radimrehurek.com/gensim/models/word2vec.html
cbow = api.load('glove-twitter-200')

# compute the sentence level representation by mean pooling across 
# the vectors for each token
    
def get_sentence_representations_and_label_from_cbow(cbow, sentences, labels):
    processed_sentences = list(
        map(lambda string: string.replace("\n", " ").lower(), sentences)
    )
    processed_sentences = list(map(word_tokenize, processed_sentences))
    sentence_representations = [ 
        np.mean(
            [ np.array(cbow[token]) for token in sentence if token in cbow ], axis=0
        ) for sentence in processed_sentences
    ]
    return np.stack([ np.append(rep, labels[i])
        for i, rep in enumerate(sentence_representations) 
        if len(rep.shape) >= 1 
    ])

train_embeddings_and_label = get_sentence_representations_and_label_from_cbow(
    cbow, toxigen_data['train']['text'], toxigen_data['train']['label']
)
test_embeddings_and_label = get_sentence_representations_and_label_from_cbow(
    cbow, toxigen_data['predict']['text'], toxigen_data['predict']['label']
)
configurations.append({
    'Dataset': 'toxigen',
    'Model': 'baseline_word2vec',
    'X_train': train_embeddings_and_label[:, :-1],
    'y_train': train_embeddings_and_label[:, -1],
    'X_test': test_embeddings_and_label[:, :-1],
    'y_test': test_embeddings_and_label[:, -1]
})

# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# word2vec_results = compute_metrics(y_test, y_pred, "predict")
# word2vec_results['Dataset'] = 'toxigen'
# word2vec_results['Model'] = 'baseline_word2vec'
# word2vec_results['do_DKNN'] = False
# word2vec_results['neighbor_method'] = None
# word2vec_results['prediction_method'] = None
# with open('baseline_word2vec_results.json', 'w') as outfile:
#     json.dump(word2vec_results, outfile, indent = 4)

for config in tqdm(configurations):
    X_train, X_test = config['X_train'], config['X_test']
    if 'y_train' in config and 'y_test' in config:
        delete_y = True
        y_train, y_test = config['y_train'], config['y_test']
    else:
        delete_y = False
        y_train, y_test = toxigen_data['train']['label'], toxigen_data['predict']['label']
    # fit simple KNN
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    base_results = compute_metrics(y_test, y_pred, "predict")
    base_results.update(all_configs)
    base_results.update(config)
    # delete extra keys before saving
    del base_results['X_train']
    del base_results['X_test']
    if delete_y:
        del base_results['y_test']
        del base_results['y_train']
    # write results to local
    with open(f'./baseline_results/{config["Model"]}_results.json', 'w') as outfile:
        json.dump(base_results, outfile, indent = 4)
