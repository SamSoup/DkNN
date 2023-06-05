from typing import Dict, List
from tqdm.auto import tqdm
from typing import Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
import pandas as pd
import os, json
import numpy as np
import pickle

PROMPT_DICT = {
    "prompt_cls": (
        "Generate a classification label indicating if the following text is"
        " {categories}.\n\nText:\n{text}\n\nLabel:"
    ),
    "prompt_nli": (
        "Given the premise and hypothesis, classify the relationship as one of"
        " entailment, neutral, or contradiction.\n\nPremise:\n{premise}\n\n"
        " Hypothesis:\n{hypothesis}\n\nRelation:"
    ),
}


def combine_cls_categories(categories: List[str]):
    stub = ""
    for i in range(len(categories) - 1):
        stub += f"{categories[i]}, "
    stub = stub[:-2] + f" or {categories[-1]}"
    return stub


def load_datasets(
    name: str, input_key: str, intToText: List[str] = None
) -> Tuple[Dataset]:
    datasets = load_dataset(
        name, cache_dir=os.environ["TRANSFORMERS_CACHE"], use_auth_token=True
    )

    # for generation models, convert ids to actually text labels
    # and in addition, create a column combining prompts and input texts
    if intToText is not None:

        def map_ids_to_text(example):
            example["label"] = intToText[example["label"]]
            return example

        for split in datasets:
            datasets[split] = datasets[split].map(map_ids_to_text)
            dataset = datasets[split]

            examples, prompts = [], []
            for index in range(len(dataset)):
                if "nli" in name:
                    premise = dataset["premise"][index]
                    hypothesis = dataset["hypothesis"][index]
                    label = dataset["label"][index]
                    prompt = PROMPT_DICT["prompt_nli"].format(
                        premise=premise, hypothesis=hypothesis
                    )
                else:
                    text = dataset[input_key][index]
                    label = dataset["label"][index]
                    prompt = PROMPT_DICT["prompt_cls"].format(
                        categories=combine_cls_categories(intToText), text=text
                    )
                example = prompt + label
                prompts.append(prompt)
                examples.append(example)
            datasets[split] = datasets[split].add_column(
                "prompt_with_label", examples
            )
            datasets[split] = datasets[split].add_column("prompt_only", prompts)
    return datasets["train"], datasets["eval"], datasets["test"]


def train_val_test_split(
    data: pd.DataFrame,
    train_pct: float,
    eval_pct: float,
    test_pct: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Here we perform train-validation-test split by calling the `train_test_split`
    function from scikit-learn twice. Data is stratified to maintain class
    distribution.

    Args:
        data_file (str): a path to the entire dataset, note that the labels (y) column
                         must be named `label`
        train_pct (float): % of data to keep for the training set
        eval_pct (float): % of data to keep for the validation set
        test_pct (float): % of data to keep for the test set
        seed (int): the seed for random shuffling of the dataFrame, for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: the train, eval, and test split
    """

    eval_and_test_pct = eval_pct + test_pct
    data_train, data_eval_and_test = train_test_split(
        data,
        train_size=train_pct,
        random_state=seed,
        shuffle=True,
        stratify=data["label"],
    )
    data_eval, data_test = train_test_split(
        data_eval_and_test,
        test_size=test_pct / eval_and_test_pct,
        random_state=seed,
        shuffle=True,
        stratify=data_eval_and_test["label"],
    )

    return data_train, data_eval, data_test


def read_json_or_df(file: str) -> pd.DataFrame:
    """
    Read the data specified by `file` as a pandas Dataframe

    Args:
        file (str): a path to the dataset, ending in ".csv" or ".json"

    Returns:
        pd.DataFrame: the data in tabular format
    """

    if file.endswith(".json"):
        data = pd.read_json(file)
    else:
        # let pandas auto detect the input separator
        data = pd.read_csv(file, sep=None, engine="python")

    return data


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
