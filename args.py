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
            "help": "Directory path containing the training data as pairs of .tsv and .txt files."
        }
    )
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "Directory path containing the validation data as pairs of .tsv and .txt files."
        }
    )
    test_file: Optional[str] = field(
        default=None, metadata={
            "help": "Directory path containing the test data as pairs of .tsv and .txt files."
        }
    )
    labels_file: Optional[str] = field(
        default=None, metadata={
            "help": "Path to file containing all possible labels."
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
                    os.path.isdir(v) and
                    any(fname.endswith('.tsv') or fname.endswith('.txt') for fname in os.listdir(v))
                ), f"{k} path must be a directory containings .tsv and .txt files"
        if self.labels_file is not None:
            assert (
                    os.path.isfile(self.labels_file) and self.labels_file.endswith(".txt")
                ), f"{k} path must be a directory containings .tsv and .txt files"
