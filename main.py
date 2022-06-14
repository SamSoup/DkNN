from transformers import (
    HfArgumentParser,
)
from data import train_val_test_split
from args import DataTrainingArguments
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    # See all possible arguments by passing the --help flag to this script.

    parser = HfArgumentParser((DataTrainingArguments))
    data_args = parser.parse_args_into_dataclasses()

    if data_args.do_train_val_test_split:
        train_data, eval_data, test_data = train_val_test_split(
            data_args.train_file, data_args.train_data_pct,
            data_args.eval_data_pct, data_args.test_data_pct
        )
