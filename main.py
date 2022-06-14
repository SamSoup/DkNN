from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from args import DataTrainingArguments
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    # See all possible arguments by passing the --help flag to this script.

    parser = HfArgumentParser((DataTrainingArguments))
    data_args = parser.parse_args_into_dataclasses()
