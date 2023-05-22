from typing import Dict
from tqdm.auto import tqdm
import pandas as pd
import os, json
import numpy as np
import pickle


def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def remove_file_if_already_exists(path_to_file: str):
    if os.path.exists(path_to_file):
        os.remove(path_to_file)


def parse_json(path: str):
    with open(path, "r") as fr:
        results = json.load(fr)
    return results


def add_to_dataframe(df, results: Dict[str, float], **kwargs):
    results.update(kwargs)
    # small helper to add to the current dataframe
    return pd.concat([df, pd.DataFrame(results, index=[0])], ignore_index=True)


def save_database_after_stacking(layers_to_save, save_reps_path, database):
    for layer in tqdm(layers_to_save):
        database[layer] = np.vstack(database[layer])
        np.savetxt(
            os.path.join(save_reps_path, f"layer_{layer}.csv"),
            database[layer],
            delimiter=",",
        )
        del database[layer]  # save memory after saving


def save_matrix_with_tags_to_file(
    filename: str, tags: np.ndarray, mat: np.ndarray
):
    with open(filename, "a") as f:
        for tag, row in zip(tags, mat):
            row_str = (
                np.array2string(
                    row, separator="\t", max_line_width=np.inf, threshold=np.inf
                )
                .removeprefix("[")
                .removesuffix("]")
            )
            f.write(f"{tag}\t{row_str}\n")


def load_predictions(work_dir: str, dataset: str, classifier: str):
    return pd.read_csv(
        f"{work_dir}/output/{dataset}/{classifier}/predict_results.txt",
        sep="\t",
    )["prediction"].to_list()


def load_whitebox(
    work_dir: str,
    dataset: str,
    classifier: str,
    pooler_config: str,
    whitebox: str,
    layer: int,
):
    classifier_category = classifier[: classifier.index("seed") - 1]
    with open(
        f"{work_dir}/results/{dataset}/{classifier_category}/{pooler_config}/{classifier}_best_{whitebox}_layer{layer}",
        "rb",
    ) as f:
        return pickle.load(f)


def load_representation(
    work_dir: str,
    dataset: str,
    classifier: str,
    mode: str,
    pooler_config: str,
    layer: int,
):
    return np.loadtxt(
        f"{work_dir}/data/{dataset}/{classifier}/{mode}/{pooler_config}/layer_{layer}.csv",
        delimiter=",",
    )
