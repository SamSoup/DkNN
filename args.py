from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training, eval and test.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify 
    them on the command line. Structure follows from run_glue.py from Transformers.
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
    shuffle_seed : Optional[int] = field (
        default = 42, metadata = {
            "help": "The random seed for shuffling rows for train-val-test split and other data related operations."
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
            assert(np.isclose(self.train_data_pct + self.eval_data_pct + self.test_data_pct, 1.0))

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    Structure follows from run_glue.py from Transformers.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )