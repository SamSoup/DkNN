"""
The purpose of this notebook is to compute the sentence level
representations for the train, validation and test set of datasets
one in all place

Models included thus far:

1. Glove Embeddings (from twitter data, https://github.com/stanfordnlp/GloVe)
2. FastText 
3. SentenceBert
3. Pretrained Bart-large
4. Pretrained Deberta-large
"""
from tqdm.auto import tqdm
import pandas as pd
import os
import numpy as np

WORK_DIR = "/work/06782/ysu707/ls6/DkNN"
import sys

sys.path.append(WORK_DIR)

# obtain the datasets to compute representations for
toxigen_data_dir = f"{WORK_DIR}/data/toxigen"
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"
toxigen_data = {
    "train": pd.read_csv(os.path.join(toxigen_data_dir, train_data_file)),
    "eval": pd.read_csv(os.path.join(toxigen_data_dir, eval_data_file)),
    "test": pd.read_csv(os.path.join(toxigen_data_dir, test_data_file))
}

esnli_data_dir = f"{WORK_DIR}/data/esnli"
esnli_data = {
    "train": pd.read_csv(os.path.join(esnli_data_dir, train_data_file)),
    "eval": pd.read_csv(os.path.join(esnli_data_dir, eval_data_file)),
    "test": pd.read_csv(os.path.join(esnli_data_dir, test_data_file))
}

DATASET = toxigen_data
dataset_name = "toxigen"
# DATASET = esnli_data
# dataset_name = "esnli"
from utils_copy import get_train_representations_from_file, get_actual_layers_to_save, get_actual_poolers_to_save
from ModelForSentenceLevelRepresentation import get_model_for_representation
from EmbeddingPooler import EmbeddingPooler

model_configs = {
    # "bart-large": (f"{WORK_DIR}/output/{dataset_name}/bart-large", 26),
    # "deberta-large": (f"{WORK_DIR}/output/{dataset_name}/deberta-large", 25),
    # "flan-t5-large": (f"{WORK_DIR}/output/{dataset_name}/flan-t5-large/", 50)
    "t5-large": (f"{WORK_DIR}/output/{dataset_name}/t5-large/", 50)
}
p = EmbeddingPooler()
pooler_configs = ["mean_with_attention", 
                  # "mean_with_attention_and_cls"
                  # "mean_with_attention_and_eos"
                 ]
save_path_stub = os.path.join(
    WORK_DIR,
    "data/{dataset_name}/{model_config}/{split}/{pooler_config}/layer_{layer}.csv"
)
batch_size = 32
for model_config in model_configs:
    model_name_or_path, num_layers = model_configs[model_config]
    model = get_model_for_representation(
        model_name_or_path,
        num_labels=2
    )
    # model = get_model_for_representation(
    #     model_name_or_path,
    #     num_labels=3,
    #     sentence1_key="premise",
    #     sentence2_key="hypothesis"
    # )
    # layers_to_save = get_actual_layers_to_save('All', num_layers)
    layers_to_save = get_actual_layers_to_save('Last Only', num_layers)
    for pooler_config in pooler_configs:
        # poolers_to_use = get_actual_poolers_to_save('All', pooler_config, layers_to_save)
        poolers_to_use = get_actual_poolers_to_save('Last Only', pooler_config, layers_to_save)
        poolers_to_use = [p.get(pooler) for pooler in poolers_to_use]
        for split in DATASET:
            database = model.encode_dataset(DATASET[split],
                                            # ["nonhate", "hate"],
                                            ["entailment", "neutral", "contradiction"],
                                            batch_size, 
                                            layers_to_save, 
                                            poolers_to_use)
            for layer in database:
                # save the representations at
                save_path = save_path_stub.format(
                    dataset_name=dataset_name,
                    model_config=model_config,
                    split=split,
                    pooler_config=pooler_config, 
                    layer=layer
                )
                np.savetxt(save_path, database[layer], delimiter=",")