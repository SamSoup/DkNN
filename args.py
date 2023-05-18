"""
Specify the Command line arguments in the form of @dataclass (Huggingface)

See python3 main.py -help for all possible arugments
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import os


@dataclass
class ComputeEncodingsArguments:
    do_compute_encodings: Optional[bool] = field(
        default=False, metadata={"help": "Should compute representation?"}
    )
    save_train_encodings_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If not none, save the training representations at "
                "specified path"
            )
        },
    )
    save_eval_encodings_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If not none, save the validation set representations at "
                "specified path"
            )
        },
    )
    save_test_encodings_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "If not none, save the test set representations at "
                "specified path"
            )
        },
    )
    layers_to_save: Optional[List[int]] = field(
        default_factory=list,
        metadata={
            "help": (
                "A list of layers to save its representation for (DkNN only) in"
                " python list format;"
            )
        },
    )
    layers_to_save_desc: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "A simplified word description of the meaning of the layers to"
                " save"
            )
        },
    )
    poolers_to_use: Optional[List[str]] = field(
        default_factory=list,
        metadata={
            "help": (
                "A list of pooler methods to use to save its representation for"
                " (DkNN only) in python list format;Must have matching length"
                " with layers_to_save"
            )
        },
    )
    poolers_to_use_desc: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "A simplified word description of the meaning of the poolers"
                " to use"
            )
        },
    )


@dataclass
class DKNNArguments:
    """
    Arguments pertaining specifically for Deek K Nearest Neighbor Configurations
    """

    do_DKNN: Optional[bool] = field(
        default=False,
        metadata={"help": "Should we do Deep K Nearest Neighbor Inference?"},
    )
    neighbor_method: Optional[str] = field(
        default="KD-Tree",
        metadata={
            "help": (
                "Which Nearest Neighbor method should we do; one of {KD-Tree,"
                " LSH}"
            )
        },
    )
    prediction_method: str = field(
        default="normal",
        metadata={
            "help": (
                "Specify which method to use for predictions, if `normal` then"
                " do argmax over log probabilities, if `conformal` do conformal"
                " predictions based on p-values overscores computed over"
                " caliberation set (DkNN). One of {normal, conformal} for now."
            )
        },
    )
    K: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "If DkNN_method is not None, how many neighbors to retrieve per"
                " layer per example"
            )
        },
    )
    read_from_database_path: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, read from `save_database_path` instead of writing"
                " to it."
            )
        },
    )
    save_database_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory path to save the training data representation, by"
                " default means overwriting what's in the directory"
            )
        },
    )
    read_from_scores_path: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, read from `save_nonconform_scores_path` instead of"
                " writing to it."
            )
        },
    )
    save_nonconform_scores_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save the non-conformity scores for the validation"
                " data, must end in .csv"
            )
        },
    )
    leaf_size: Optional[int] = field(
        default=40,
        metadata={
            "help": (
                "If neighbor_method is KD-Tree, then this is the number of"
                " points at which to switchto brute force nearest neighbors"
                " (hyper-parameter)"
            )
        },
    )
    num_hash_funct: Optional[int] = field(
        default=16,
        metadata={
            "help": (
                "If neighbor_method is KSH, then this is the number of random"
                " hash functions to use"
            )
        },
    )
    output_and_save_neighbors: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Should we output and save the nearest neighbors of each"
                " inference example? If so, will save neighbor ids and"
                " distances to the retrieved nearest neighbors"
            )
        },
    )
    dist_metric: Optional[str] = field(
        default="minkowski",
        metadata={
            "help": (
                "The distance function to use for computing nearest neighbors,"
                " must be one of minkowski (with power specified) or cosine"
            )
        },
    )
    minkowski_power: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "The power p to be used in the minkowski distance function, if"
                " we should use it"
            )
        },
    )
    dist_to_weight_fct: Optional[str] = field(
        default="uniform",
        metadata={
            "help": (
                "Which function of distances to weights conversion should we"
                " use; may be one of <>"
            )
        },
    )

    def __post_init__(self):
        self.neighbor_methods = {"KD-Tree", "LSH"}
        self.prediction_methods = {"nonconformal", "conformal"}
        self.dist_metrics = {"minkowski", "cosine"}
        if self.do_DKNN:
            assert (
                self.neighbor_method in self.neighbor_methods
            ), f"Nearest Neighbor method be one of {self.neighbor_methods}"
            assert self.dist_metric in self.dist_metrics, (
                "Nearest Neighbor distance method be one of"
                f" {self.neighbor_methods}"
            )
            assert (
                self.prediction_method in self.prediction_methods
            ), f"Prediction method must be one of {self.prediction_methods}"
            if self.neighbor_method == "KD-Tree":
                assert (
                    self.dist_metric != "cosine"
                ), f"For KD-Tree, the distance function cosine is not supported"
        if self.read_from_database_path:
            assert os.path.exists(self.save_database_path) and os.path.isdir(
                self.save_database_path
            ), (
                "If reading from database path, then"
                f" {self.save_database_path} must already exist and be a valid"
                " directory"
            )
        if self.read_from_scores_path:
            assert os.path.exists(
                self.save_nonconform_scores_path
            ) and os.path.isfile(self.save_nonconform_scores_path), (
                "If reading from database path, then"
                f" {self.save_nonconform_scores_path}must already exist and be"
                " a valid path"
            )
        if self.save_nonconform_scores_path is not None:
            assert self.save_nonconform_scores_path.endswith(
                ".csv"
            ), f"{self.save_nonconform_scores_path} must end in a .csv"


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training, eval and test.
    Using `HfArgumentParser` we can turn this class into argparse arguments to
    be able to specify them on the command line.
    Structure follows from run_glue.py from Transformers.
    """

    dataset_name: str = field(
        metadata={"help": "Name of the dataset to train/predict on"}
    )
    int_to_text: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "For T5 only: convert classication id to text"},
    )
    compute_predict_results: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Should we also compute metrics for the prediction (test) set;"
                " if yes, then thetest set must contain valid labels and"
                " do_predict must be true"
            )
        },
    )
    evaluation_metrics: Optional[List[str]] = field(
        default_factory=list, metadata={"help": "A list of evaluation to use."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization."
                " Sequences longer than this will be truncated, sequences"
                " shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will"
                " pad the samples dynamically when batching to the maximum"
                " length."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the"
                " number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the"
                " number of evaluation examples to this value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the"
                " number of test set examples to this value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to file containing the training data; must be either .csv"
                " or .json."
            )
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to file containing the validation data; must be either"
                " .csv or .json."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to file containing the test set data; must be either .csv"
                " or .json."
            )
        },
    )
    do_train_val_test_split: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "If True, then split the train_file into train-eval-test"
                " splits."
            )
        },
    )
    train_data_pct: Optional[float] = field(
        default=0.7,
        metadata={
            "help": (
                "The amount of data to use for training, only used if"
                " do_train_val_test_split is True."
            )
        },
    )
    eval_data_pct: Optional[float] = field(
        default=0.2,
        metadata={
            "help": (
                "The amount of data to use for validation, only used if"
                " do_train_val_test_split is True."
            )
        },
    )
    test_data_pct: Optional[float] = field(
        default=0.1,
        metadata={
            "help": (
                "The amount of data to use for test set, only used if"
                " do_train_val_test_split is True."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    sentence1_key: str = field(
        default="input",
        metadata={
            "help": (
                "The name of the column that contains the first input sequences"
                " to the model."
            )
        },
    )
    sentence2_key: str = field(
        default="input",
        metadata={
            "help": (
                "The name of the column that contains the second input"
                " sequences to the model."
            )
        },
    )
    do_early_stopping: bool = field(
        default=True,
        metadata={
            "help": (
                "Should we do early stopping? If so, then"
                " load_best_model_at_end must be trueevaluation_strategy must"
                ' be "steps", and that metric_for_best_model must be specified'
            )
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "If we are using early stopping callback, then this is the"
                " input to transformers.EarlyStoppingCallback"
            )
        },
    )
    do_weighted_cross_entropy_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Should we use weighted cross entropy loss? If true, then the"
                " weightsfor each class must be speciifed in weights_per_class"
            )
        },
    )
    weights_per_class: Optional[List[float]] = field(
        default_factory=list,
        metadata={
            "help": (
                "The weights per class for use in computing the weighted loss"
            )
        },
    )
    save_logits: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Should we save the logits of each inference example?"
        },
    )

    def __post_init__(self):
        to_check = {
            "Train": self.train_file,
            "Eval": self.validation_file,
            "Test": self.test_file,
        }
        for k, v in to_check.items():
            if v is not None:
                assert v.endswith(".csv") or v.endswith(
                    ".json"
                ), f"{k} path must be a file ending in .tsv or .txt"
        if self.do_train_val_test_split:
            assert np.isclose(
                self.train_data_pct + self.eval_data_pct + self.test_data_pct,
                1.0,
            ), "train, eval, test split % must add up to 1"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune from.

    Structure follows from run_glue.py from Transformers.
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    is_generative: Optional[bool] = field(
        default=False, metadata={"help": "Is this model generative?"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pretrained config name or path if not the same as model_name"
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pretrained tokenizer name or path if not the same as"
                " model_name"
            )
        },
    )
    freeze_base_model_params: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Should we freeze the base model parameters and only train"
                " classification layer?"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded"
                " from huggingface.co"
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer (backed by the"
                " tokenizers library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag"
                " name or commit id)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli"
                " login` (necessary to use this script with private models)."
            )
        },
    )
