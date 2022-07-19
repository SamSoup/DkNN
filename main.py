"""
Main script for execution
"""

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed
)
from data import train_val_test_split, read_data
from args import DataArguments, ModelArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_metric, Dataset
from DkNNTrainer import DkNNTrainer
import numpy as np
import os
import torch
import logging
import sys
import datasets
import transformers
import random

logger = logging.getLogger(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def detect_last_checkpoint(output_dir: str, do_train: bool, 
                            overwrite_output_dir: bool,
                            resume_from_checkpoint: bool) -> str:
    """
    Detect the last checkpoint for a particular model to resume training

    Args:
        output_dir (str): the path where checkpoints are stored
        do_train (bool): True if we want to train the model
        overwrite_output_dir (bool): True if we want to override the checkpoint dir
        resume_from_checkpoint (bool): True if we wish to resume from the latest checkpt

    Raises:
        ValueError: When output directory already exists and we do not wish to override it

    Returns:
        str: the path to the last checkpoint of our model
    """

    last_checkpoint = None
    if os.path.isdir(output_dir) and do_train and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this "
                "behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint

def set_up_logging(training_args: TrainingArguments):
    """
    Logging set ups

    Args:
        training_args (TrainingArguments): various arguments for training,
        see huggingface documentation for specifics at
        https://huggingface.co/docs/transformers/v4.19.4/en/main_classes/trainer#transformers.TrainingArguments
    """

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

def main():
    # See all possible arguments by passing the --help flag to this script.
    # parse the arguments
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # logging
    set_up_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)

    # read in data
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    if data_args.do_train_val_test_split:
        train_data, eval_data, test_data = train_val_test_split(
            data_args.train_file, data_args.train_data_pct,
            data_args.eval_data_pct, data_args.test_data_pct,
            data_args.shuffle_seed
        )
    else:
        train_data = read_data(data_args.train_file)
        eval_data = read_data(data_args.validation_file)
        test_data = read_data(data_args.test_file)

    # convert data to what models expect
    train_data = Dataset.from_pandas(train_data)
    eval_data = Dataset.from_pandas(eval_data)
    test_data = Dataset.from_pandas(test_data)

    # note this assumes that the training data must have all possible labels
    # note also that this assumes a classification task
    label_list = train_data.unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer for training
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    ignore_mismatched_sizes = False
    # max_position_embeddings=512 by default, which may be less than the maximum sequence length
    # if that's case, then we randomly initialize the classification head by passing
    # ignore_mismatched_sizes = True
    if config.max_position_embeddings < data_args.max_seq_length:
        config.max_position_embeddings = data_args.max_seq_length
        ignore_mismatched_sizes = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    # check max_seq_length compared to model defaults
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = ((examples[data_args.input_key],))
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    with training_args.main_process_first(desc="dataset map pre-processing"):
        train_data = train_data.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        eval_data = eval_data.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        test_data = test_data.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # make sure that we trunk examples, if specified
    check = [
        (training_args.do_train, data_args.max_train_samples, train_data),
        (training_args.do_eval, data_args.max_eval_samples, eval_data),
        (training_args.do_predict, data_args.max_test_samples, test_data),
    ]
    for do, sample_limit, data in check:
        if do and sample_limit is not None:
            data = data.select(range(min(len(train_data), sample_limit)))

    # add a unique tag for the training examples - might be useful for retrieval later
    new_column = list(range(train_data.num_rows))
    train_data = train_data.add_column("tag", new_column)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_data)), 3):
            logger.info(f"Sample {index} of the training set: {train_data[index]}.")

    # Custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    scores = ["accuracy", "f1", "precision", "recall"]
    def compute_metrics(p: EvalPrediction):
        computed_scores = {}
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        # see datasets.list_metrics() for the complete list
        for score in scores:
            metric_func = load_metric(score)
            computed_scores[score] = metric_func.compute(predictions=preds, references=p.label_ids)[score]
        return computed_scores

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # torch.cuda.empty_cache()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=eval_data if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # detect last checkpt
    last_checkpoint = detect_last_checkpoint(training_args.output_dir, training_args.do_train, 
                                             training_args.overwrite_output_dir, training_args.resume_from_checkpoint)

    # Training
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))
        trainer.save_model()  # Saves the tokenizer too
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if data_args.DkNN_method is not None:
            trainer = DkNNTrainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=eval_data if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
                DkNN_method=data_args.DkNN_method,
                num_neighbors=data_args.K,
                label_list = label_list,
                layers_to_save=data_args.layers_to_save,
                save_database_path=data_args.save_database_path
            )
        metrics = trainer.evaluate(eval_dataset=eval_data)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_data))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test set Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        test_data = test_data.remove_columns("label")
        predictions = trainer.predict(test_data, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    item = label_list[item]
                    writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()
