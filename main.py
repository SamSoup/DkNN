"""
Main script for execution, this is the ONLY place where any processing
of command line arguments occurs
"""

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    EarlyStoppingCallback,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed
)
from typing import Dict
from data import train_val_test_split, read_data
from args import DKNNArguments, DataArguments, ModelArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_metric, Dataset, load_dataset
from NearestNeighborLogits import LogProbabilityLogitsFactory, ConformalLogitsFactory
from NearestNeighborFinder import KDTreeNearestNeighborFactory, LocalitySensitiveHashingNearestNeighborFactory
from DeepKNearestNeighborClassifier import DeepKNearestNeighborClassifier
from ComputeAndSaveTrainRepTrainer import ComputeAndSaveTrainRepTrainer
from ComputeAndSaveConformalScoresTrainer import ComputeAndSaveConformalScoresTrainer
from DeepKNearestNeighborTrainer import DeepKNearestNeighborTrainer
from CustomLossTrainer import CustomLossTrainer
import numpy as np
import torch.nn as nn
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
    parser = HfArgumentParser((DKNNArguments, DataArguments, ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        DKNN_args, data_args, model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        DKNN_args, data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    # logging
    set_up_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)

    # read in data
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # split dataset into train, val, test
    if data_args.do_train_val_test_split:
        data = read_data(data_args.train_file)
        train_data, eval_data, test_data = train_val_test_split(
            train_data, data_args.train_data_pct,
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
        # Tokenize the texts - input only
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
            desc="Running tokenizer on training dataset",
        )
        eval_data = eval_data.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on eval dataset",
        )
        test_data = test_data.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on test dataset",
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
    train_data = train_data.add_column("tag", list(range(train_data.num_rows)))
    if DKNN_args.save_logits or DKNN_args.output_and_save_neighbor_ids:
        eval_data = eval_data.add_column("tag", list(range(eval_data.num_rows)))
        test_data = test_data.add_column("tag", list(range(test_data.num_rows)))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_data)), 3):
            logger.info(f"Sample {index} of the training set: {train_data[index]}.")

    # Custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    metric_descs = ["accuracy", "f1", "precision", "recall"]
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        computed_scores = {}
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        # see datasets.list_metrics() for the complete list
        for metric_desc in metric_descs:
            metric_func = load_metric(metric_desc)
            computed_scores[metric_desc] = metric_func.compute(predictions=preds, references=p.label_ids)[metric_desc]
        # get per-class f1 too
        f1_scores = load_metric("f1").compute(predictions=preds, references=p.label_ids, average=None)["f1"]
        computed_scores["f1-negative"] = f1_scores[0]
        computed_scores["f1-positive"] = f1_scores[1]
        return computed_scores

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    # Additional: add early stopping callback if specified
    callbacks = None
    if data_args.do_early_stopping:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]

    # Initialize our Trainer
    if data_args.do_weighted_cross_entropy_loss:
        weights = torch.tensor(data_args.weights_per_class).to(device)
        trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=eval_data if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        loss_fct=nn.CrossEntropyLoss(weight=weights)
    )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=eval_data if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
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
        # TODO: hyper-parameter search
        # trainer.hyperparameter_search(
        #     direction="maximize", 
        #     backend="ray", 
        #     n_trials=10 # number of trials
        # )
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

    # DKNN
    if DKNN_args.do_DKNN:
        # first, we compute the representations for all training examples
        saveTrainRepTrainer = ComputeAndSaveTrainRepTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data,
            layers_to_save=DKNN_args.layers_to_save,
            read_from_database_path=DKNN_args.read_from_database_path,
            save_database_path=DKNN_args.save_database_path,
        )
        database = saveTrainRepTrainer.compute_and_save_training_points_representations()
        # create the NearestNeighborFinder
        if DKNN_args.neighbor_method == "KD-Tree":
            nearestNeighborFactory = KDTreeNearestNeighborFactory(
                K=DKNN_args.K, 
                layers_to_save=DKNN_args.layers_to_save, 
                database=database, 
                layer_dim=model.config.hidden_size,
                leaf_size=DKNN_args.leaf_size
            )
        elif DKNN_args.neighbor_method == "LSH":
            nearestNeighborFactory = LocalitySensitiveHashingNearestNeighborFactory(
                K=DKNN_args.K, 
                layers_to_save=DKNN_args.layers_to_save, 
                database=database, 
                layer_dim=model.config.hidden_size,
                num_hash_funct=DKNN_args.num_hash_funct
            )
        else:
            raise ValueError("Illegal Nearest Neighbor Finder method")
        nearestNeighborFinderFunction = nearestNeighborFactory.createNearestNeighborFunction()

        # create the NearestNeighborLogit Function
        if DKNN_args.prediction_method == "normal":
            logitsFactory = LogProbabilityLogitsFactory(label_list)
        elif DKNN_args.prediction_method == "conformal":
            # now, compute the conformal socres first
            computeAndSaveConformalScoresTrainer = ComputeAndSaveConformalScoresTrainer(
                model=model,
                args=training_args,
                caliberation_dataset=eval_data,
                data_collator=data_collator,
                nearestNeighborFunction=nearestNeighborFinderFunction,
                read_from_scores_path=DKNN_args.read_from_scores_path,
                save_nonconform_scores_path=DKNN_args.save_nonconform_scores_path
            )
            scores = computeAndSaveConformalScoresTrainer.compute_and_save_nonconformity_scores()
            logitsFactory = ConformalLogitsFactory(label_list, scores)
        else:
            raise ValueError("Illegal Nearest Neighbor Prediction method")
        nearestNeighborLogitFunction = logitsFactory.createLogitsFunction()

        # TODO: consider different loss functions?
        # finally, create the complete Nearest Neighbor Classifier
        classifier = DeepKNearestNeighborClassifier(
            NearestNeighborFunction=nearestNeighborFinderFunction,
            LogitsFunction=nearestNeighborLogitFunction,
            LossFunction=torch.nn.functional.nll_loss
        )

        # create custom trainer that utilizes our function
        trainer = DeepKNearestNeighborTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data if training_args.do_eval else None,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            classifier=classifier,
            save_logits=DKNN_args.save_logits,
            output_and_save_neighbor_ids=DKNN_args.output_and_save_neighbor_ids
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
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
        if type(predictions) is tuple:
            # assume the first thing in the tuple is predictions
            predictions = predictions[0]
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
