"""
Usage: python3 ComputeGroupResults.py ./result_locations_toxigen.json
"""
from utils import randargmax
from datasets import load_metric
from typing import List
from tqdm.auto import tqdm
import json
import os
import sys
import numpy as np
import pandas as pd

toxigen_eval = pd.read_csv("./data/toxigen/eval_data.csv")
toxigen_test = pd.read_csv("./data/toxigen/test_data.csv")
# evaluation set distribution 
# print(toxigen_eval.target_group.value_counts())
# test set distribution
# print(toxigen_test.target_group.value_counts())

convert_test_set_names = {
    "folks with physical disabilities" : "physical_dis",
    "jewish folks": "jewish",
    "muslim folks": "muslim",
    "chinese folks": "chinese",
    "mexican folks": "mexican",
    "middle eastern folks": "middle_east",
    "folks with mental disabilities": "mental_dis",
    "lgbtq+ folks": "lgbtq",
    "women": "women",
    "latino/hispanic folks": "latino",
    "black/african-american folks": "black",
    "asian folks": "asian",
    "native american/indigenous folks": "native_american",
    "black folks / african-americans": "black",
    "native american folks": "native_american"
}
def preprocess_toxigen_test(data: pd.DataFrame):
    data = data.rename(columns={"target_group": "target_group_orig"}, errors="raise")
    data["target_group"] = list(map(lambda n: convert_test_set_names[n], data["target_group_orig"]))
    return data

toxigen_test = pd.read_csv("./data/toxigen/test_data.csv")
# test set group names has some redundancy - clean up a bit
toxigen_test = preprocess_toxigen_test(toxigen_test)
# print(toxigen_test.target_group.value_counts())

assert all(np.sort(toxigen_test.target_group.unique()) == np.sort(toxigen_eval.target_group.unique()))

def compute_group_level_metric(logits: np.ndarray, data: pd.DataFrame, metric_descs: List[str], save_path_header: str):
    predictions = randargmax(logits)
    data["predictions"] = predictions
    for metric_desc in metric_descs:
        metric_func = load_metric(metric_desc)
        res = data.groupby("target_group").apply(
            lambda group: metric_func.compute(predictions=group["predictions"], references=group["label"])[metric_desc]
        ).to_dict()
        # save the computed group level result
        save_path = save_path_header + f"group_{metric_desc}.json"
        with open(save_path, "w") as outfile:
            json.dump(res, outfile, indent = 4)


# Read logits, then compute group-wise accuracy, F1
input_file = sys.argv[1]
with open(input_file, 'r') as fr:
    res_locations = json.load(fr)

inference_modes = ["eval_", "predict_"]
metric_descs = ["accuracy", "f1", "precision", "recall"]
dir_extensions = ["eval", "DKNN"]
train_data_file = "train_data.csv"
eval_data_file = "eval_data.csv"
test_data_file = "test_data.csv"

for loc_data in tqdm(res_locations):
    eval_data = pd.read_csv(os.path.join(loc_data['data_dir'], eval_data_file))
    test_data = preprocess_toxigen_test(pd.read_csv(os.path.join(loc_data['data_dir'], test_data_file)))
    result_paths = ([os.path.join(loc_data['output_dir'], "eval")] + 
                    [subdir[0] for subdir in list(os.walk(os.path.join(loc_data['output_dir'], "DKNN")))[1:]])
    for path in tqdm(result_paths):
        if "OLD" in path:
            continue
        for mode in inference_modes:
            logits = np.loadtxt(os.path.join(path, f"{mode}logits.txt"))[:, 1:] # trim off index
            data = eval_data if mode == "eval_" else test_data
            save_path_header = os.path.join(path, f"{mode}")
            compute_group_level_metric(logits, data, metric_descs, save_path_header)
