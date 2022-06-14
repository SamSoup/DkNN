from dataclasses import dataclass, field
from typing import Optional

import os

@dataclass
class DataTrainingArguments:
    """
    Structure follows from run_glue.py from Transformers
    Arguments pertaining to what data we are going to input our model for training, eval and test.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify 
    them on the command line.
    """

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training "
                    "examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation "
                    "examples to this value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test set "
                    "examples to this value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "Path to file containing the training data; must be either .csv or .json."
        }
    )
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "Path to file containing the validation data; must be either .csv or .json."
        }
    )
    test_file: Optional[str] = field(
        default=None, metadata={
            "help": "Path to file containing the test set data; must be either .csv or .json."
        }
    )
    do_train_val_test_split: Optional[bool] = field(
        default=False, metadata = {
            "help": "If True, then split the train_file into train-eval-test splits."
        }
    )
    train_data_pct: Optional[float] = field (
        default = 0.7, metadata = {
            "help": "The amount of data to use for training, only used if do_train_val_test_split is True."
        }
    )
    eval_data_pct: Optional[float] = field (
        default = 0.2, metadata = {
            "help": "The amount of data to use for validation, only used if do_train_val_test_split is True."
        }
    )
    test_data_pct: Optional[float] = field (
        default = 0.1, metadata = {
            "help": "The amount of data to use for test set, only used if do_train_val_test_split is True."
        }
    )
    seed: Optional[int] = field (
        default = 42, metadata = {
            "help": "The random seed for initializing weights, shuffling rows for train-val-test split, etc."
        }
    )

    def __post_init__(self):
        to_check = {
            "Train": self.train_file,
            "Eval": self.validation_file,
            "Test": self.test_file
        }
        for k, v in to_check.items():
            if v is not None:
                assert (
                    v.endswith('.csv') or v.endswith('.json')
                ), f"{k} path must be a file ending in .tsv or .txt"
        if self.do_train_val_test_split:
            assert(self.train_data_pct + self.test_data_pct + self.test_data_pct == 1.0)
