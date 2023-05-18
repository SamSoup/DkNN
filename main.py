"""
Main script for execution, this is the ONLY place where any processing
of command line arguments occurs

Note: before running this script, it is crucial that login through the 
huggingface terminal to be complete
"""
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    EarlyStoppingCallback,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from utils import randargmax
from typing import Dict, Tuple, List, Union
from SaveLogitsTrainer import SaveLogitsTrainer
from args import (
    ComputeEncodingsArguments,
    DKNNArguments,
    DataArguments,
    ModelArguments,
)
from transformers.trainer_utils import (
    get_last_checkpoint,
    denumpify_detensorize,
)
from datasets import Dataset, load_dataset
from NearestNeighborLogits import (
    LogProbabilityLogitsFactory,
    ConformalLogitsFactory,
)
from NearestNeighborDistancesToWeightsFuncts import (
    NearestNeighborDistancesToWeightsFuncts,
)
from NearestNeighborFinders import (
    KDTreeNearestNeighborFactory,
    LocalitySensitiveHashingNearestNeighborFactory,
)
from DeepKNearestNeighborClassifier import DeepKNearestNeighborClassifier
from ComputeEncodings import ComputeEncodings
from ComputeAndSaveTrainRepTrainer import ComputeAndSaveTrainRepTrainer
from ComputeAndSaveConformalScoresTrainer import (
    ComputeAndSaveConformalScoresTrainer,
)
from DeepKNearestNeighborTrainer import DeepKNearestNeighborTrainer
from CustomLossTrainer import CustomLossTrainer
from CustomSeqToSeqTrainer import CustomSeq2SeqTrainer
from EmbeddingPooler import EmbeddingPooler
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_distances
import evaluate
import numpy as np
import torch.nn as nn
import os
import torch
import pprint
import logging
import sys
import datasets
import transformers
import random

os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".cache")
logger = logging.getLogger(__name__)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def detect_last_checkpoint(
    output_dir: str,
    do_train: bool,
    overwrite_output_dir: bool,
    resume_from_checkpoint: bool,
) -> str:
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
                f"Output directory ({output_dir}) already exists and is not"
                " empty. Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
                " To avoid this behavior, change the `--output_dir` or add"
                " `--overwrite_output_dir` to train from scratch."
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
        f"Process rank: {training_args.local_rank}, device:"
        f" {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)},"
        f" 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def set_seed_for_reproducability(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_data(name: str, intToText: List[str] = None) -> Tuple[Dataset]:
    dataset = load_dataset(
        name, cache_dir=os.environ["TRANSFORMERS_CACHE"], use_auth_token=True
    )

    # for generation models, convert ids to actually text labels
    if intToText is not None:

        def map_ids_to_text(example):
            example["label"] = intToText[example["label"]]
            return example

        for split in dataset:
            dataset[split] = dataset[split].map(map_ids_to_text)
    return dataset["train"], dataset["eval"], dataset["test"]


def set_max_seq_length(config: AutoConfig, max_seq_length: int):
    """
    Max_position_embeddings may be less than the maximum sequence length
    if so, then we randomly initialize the classification head by passing
    ignore_mismatched_sizes = True

    Args:
        config (AutoConfig): _description_
        max_seq_length (int): _description_
    """
    ignore_mismatched_sizes = False
    if (
        hasattr(config, "max_position_embeddings")
        and config.max_position_embeddings < max_seq_length
    ):
        config.max_position_embeddings = max_seq_length
        ignore_mismatched_sizes = True
    if hasattr(config, "n_positions") and config.n_positions < max_seq_length:
        config.n_positions = max_seq_length
        ignore_mismatched_sizes = True
    return config, ignore_mismatched_sizes


def load_model(
    model_name_or_path: str,
    config: AutoConfig,
    model_revision: str,
    use_auth_token: bool,
    ignore_mismatched_sizes: bool,
    tokenizer: AutoTokenizer,
    freeze_base_model_params: bool,
):
    # select the appropriate LM
    if "t5" in model_name_or_path:
        LM = AutoModelForSeq2SeqLM
    else:
        LM = AutoModelForSequenceClassification
    model = LM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        revision=model_revision,
        use_auth_token=True if use_auth_token else None,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    model.resize_token_embeddings(len(tokenizer))
    if freeze_base_model_params:
        for name, param in model.named_parameters():
            if (
                "classification_head" not in name and "classifier" not in name
            ):  # freeze all besides classifier layer
                param.requires_grad = False
    return model


def update_label_to_id(
    model: Union[AutoModelForSeq2SeqLM, AutoModelForSequenceClassification],
    config: AutoConfig,
    label_list: List[Union[str, int]],
    num_labels: int,
):
    # Some models have set the order of the labels to use, so ensure we do so
    label_to_id = None
    if (
        model.config.label2id
        != PretrainedConfig(num_labels=num_labels).label2id
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k, v in model.config.label2id.items()
        }
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]])
                for i in range(num_labels)
            }
        else:
            logger.warning(
                (
                    "Your model seems to have been trained with labels, but"
                    " they don't match the dataset: "
                ),
                (
                    f"model labels: {list(sorted(label_name_to_id.keys()))},"
                    f" dataset labels: {list(sorted(label_list))}.\nIgnoring"
                    " the model labels as a result."
                ),
            )
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in config.label2id.items()
        }

    return label_to_id


def main():
    # See all possible arguments by passing the --help flag to this script.
    # parse the arguments
    parser = HfArgumentParser(
        (
            ComputeEncodingsArguments,
            DKNNArguments,
            DataArguments,
            ModelArguments,
            TrainingArguments,
        )
    )
    print(sys.argv)
    encoding_args, DKNN_args, data_args, model_args, training_args = (
        parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        if len(sys.argv) > 2 and sys.argv[-1].endswith(".json")
        else parser.parse_args_into_dataclasses()
    )
    sentence1_key, sentence2_key = (
        data_args.sentence1_key,
        data_args.sentence2_key,
    )

    # set up
    set_up_logging(training_args)
    set_seed_for_reproducability(training_args.seed)

    # read in data
    train_data, eval_data, test_data = (
        load_data(data_args.dataset_name, data_args.int_to_text)
        if model_args.is_generative
        else load_data(data_args.dataset_name)
    )
    # NOTE: assume that the training data must have all label classes
    # NOTE: assume a classification task
    label_list = train_data.unique("label")
    label_list.sort()  # for determinism
    num_labels = len(label_list)

    # Load config, pretrained model and tokenizer
    config_name = (
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path
    )
    config = AutoConfig.from_pretrained(
        config_name,
        num_labels=num_labels,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config, ignore_mismatched_sizes = set_max_seq_length(
        config, data_args.max_seq_length
    )

    tokenizer_name = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if (
        "gpt2" in model_args.model_name_or_path
        or "llama" in model_args.model_name_or_path
    ):
        # GPT-2 is a text generative model which its last token embedding to
        # predict subsequent tokens. Therefore unlike BERT which uses its first
        # token embedding, in the tokenization step of input text here,
        # we should use the last token as below.
        # Similarly, for llama this is also required
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    model = load_model(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        model_revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        tokenizer=tokenizer,
        freeze_base_model_params=model_args.freeze_base_model_params,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation,
        # to the max sequence length in each batch (faster, less memory)
        padding = False

    label_to_id = update_label_to_id(model, config, label_list, num_labels)

    # check max_seq_length compared to model defaults
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger"
            " than the maximum length for themodel"
            f" ({tokenizer.model_max_length}). Using"
            f" max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts - input only
        args = (
            (examples[sentence1_key],)
            if sentence2_key == "none"
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs, if not t5
        if "label" in examples:
            if "t5" in model_args.model_name_or_path:
                result["label"] = tokenizer(
                    examples["label"],
                    padding=True,
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids
            elif label_to_id is not None:
                result["label"] = [
                    (label_to_id[l] if l != -1 else -1)
                    for l in examples["label"]
                ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        data_dict = {
            "train": [
                training_args.do_train,
                data_args.max_train_samples,
                train_data,
            ],
            "eval": [
                training_args.do_eval,
                data_args.max_eval_samples,
                eval_data,
            ],
            "test": [
                training_args.do_predict,
                data_args.max_test_samples,
                test_data,
            ],
        }
        for split in data_dict:
            # preprocess each dataset
            data_dict[split][2] = data_dict[split][2].map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {split} dataset",
            )
            # truncate each dataset if specified
            do, sample_limit = data_dict[split][0], data_dict[split][1]
            if do and sample_limit is not None:
                data_dict[split][2] = data_dict[split][2].select(
                    range(min(len(train_data), sample_limit))
                )

    train_data, eval_data, test_data = (
        data_dict["train"][2],
        data_dict["eval"][2],
        data_dict["test"][2],
    )
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_data)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_data[index]}."
            )

    # Custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    metric_descs = data_args.evaluation_metrics
    agg = ["micro", "macro", "weighted"]

    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        computed_scores = {}
        preds = (
            p.predictions[0]
            if isinstance(p.predictions, tuple)
            else p.predictions
        )
        if preds.ndim > 1:
            preds = randargmax(preds)  # break ties arbitrarily
        # preds = np.argmax(preds, axis=1)
        # see datasets.list_metrics() for the complete list
        for metric_desc in metric_descs:
            metric_func = evaluate.load(metric_desc)
            if num_labels > 2:
                for avg in agg:
                    if "accuracy" not in metric_desc:
                        computed_scores[
                            f"{avg}_{metric_desc}"
                        ] = metric_func.compute(
                            predictions=preds,
                            references=p.label_ids,
                            average=avg,
                        )[
                            metric_desc
                        ]
                    else:
                        computed_scores[metric_desc] = metric_func.compute(
                            predictions=preds, references=p.label_ids
                        )[metric_desc]
            else:
                computed_scores[metric_desc] = metric_func.compute(
                    predictions=preds, references=p.label_ids
                )[metric_desc]
        # get per-class f1 too, if we want f1
        if "f1" in metric_descs:
            f1_scores = evaluate.load("f1").compute(
                predictions=preds, references=p.label_ids, average=None
            )["f1"]
            for i, l in enumerate(data_args.int_to_text):
                computed_scores[f"f1-{l}"] = f1_scores[i]
        return computed_scores

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    generative_label_to_id = {
        label: i for i, label in enumerate(data_args.int_to_text)
    }

    def compute_metrics_generative(eval_preds):
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels
        )
        # convert text to ids
        decoded_preds_ids = np.array(
            list(map(lambda x: generative_label_to_id[x], decoded_preds))
        )
        decoded_labels_ids = np.array(
            list(map(lambda x: generative_label_to_id[x], decoded_labels))
        )
        p = EvalPrediction(
            predictions=decoded_preds_ids, label_ids=decoded_labels_ids
        )
        result = compute_metrics(p)

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer
    # is passed to Trainer, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Additional: add early stopping callback if specified
    callbacks = None
    if data_args.do_early_stopping:
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=data_args.early_stopping_patience
            )
        ]

    p = EmbeddingPooler()
    poolers = list(map(p.get, encoding_args.poolers_to_use))

    # Compute Representations
    if encoding_args.do_compute_encodings:
        encoder = ComputeEncodings(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            test_dataset=test_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            layers_to_save=encoding_args.layers_to_save,
            poolers=poolers,
            save_train_encodings_path=encoding_args.save_train_encodings_path,
            save_eval_encodings_path=encoding_args.save_eval_encodings_path,
            save_test_encodings_path=encoding_args.save_test_encodings_path,
        )
        encoder.compute_and_save_encodings()

    # Initialize our Trainer
    if data_args.do_weighted_cross_entropy_loss:
        train_class_weights = torch.tensor(data_args.weights_per_class).to(
            device
        )
        trainer = CustomLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=eval_data if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            save_logits=data_args.save_logits,
            loss_fct=nn.CrossEntropyLoss(weight=train_class_weights),
        )
    elif "t5" in model_args.model_name_or_path:
        training_args.generation_max_length = config.max_length
        training_args.generation_num_beams = config.num_beams
        training_args.predict_with_generate = True
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=eval_data if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics_generative,
            save_reps_path=DKNN_args.save_database_path,
            layers_to_save=encoding_args.layers_to_save,
            poolers_to_use=poolers
            if DKNN_args.save_database_path is not None
            else None,
        )
    else:
        trainer = SaveLogitsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data if training_args.do_train else None,
            eval_dataset=eval_data if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            save_logits=data_args.save_logits,
            save_reps_path=DKNN_args.save_database_path,
            layers_to_save=encoding_args.layers_to_save,
            poolers_to_use=poolers
            if DKNN_args.save_database_path is not None
            else None,
        )

    # detect last checkpt
    last_checkpoint = detect_last_checkpoint(
        training_args.output_dir,
        training_args.do_train,
        training_args.overwrite_output_dir,
        training_args.resume_from_checkpoint,
    )

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
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))
        trainer.save_model()  # Saves the tokenizer too
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # DKNN
    if DKNN_args.do_DKNN:
        # get the poolers for the hidden states
        p = EmbeddingPooler()
        poolers = list(map(p.get, encoding_args.poolers_to_use))

        # first, we compute the representations for all training examples
        saveTrainRepTrainer = ComputeAndSaveTrainRepTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_dataset=train_data,
            layers_to_save=encoding_args.layers_to_save,
            poolers=poolers,
            read_from_database_path=DKNN_args.read_from_database_path,
            save_database_path=DKNN_args.save_database_path,
        )
        database = (
            saveTrainRepTrainer.compute_and_save_training_points_representations()
        )
        # create the NearestNeighborFinder
        # first, we need to obtain the distance function
        if DKNN_args.dist_metric == "minkowski":
            dist_funct = DistanceMetric.get_metric(
                "minkowski", p=int(DKNN_args.minkowski_power)
            )
        elif DKNN_args.dist_metric == "cosine":
            dist_funct = cosine_distances
        if DKNN_args.neighbor_method == "KD-Tree":
            nearestNeighborFactory = KDTreeNearestNeighborFactory(
                K=DKNN_args.K,
                layers_to_save=encoding_args.layers_to_save,
                database=database,
                layer_dim=model.config.hidden_size,
                dist_metric=dist_funct,
                leaf_size=DKNN_args.leaf_size,
            )
        elif DKNN_args.neighbor_method == "LSH":
            nearestNeighborFactory = (
                LocalitySensitiveHashingNearestNeighborFactory(
                    K=DKNN_args.K,
                    layers_to_save=encoding_args.layers_to_save,
                    database=database,
                    layer_dim=model.config.hidden_size,
                    dist_metric=dist_funct,
                    num_hash_funct=DKNN_args.num_hash_funct,
                )
            )
        else:
            raise ValueError("Illegal Nearest Neighbor Finder method")
        nearestNeighborFinderFunction = (
            nearestNeighborFactory.createNearestNeighborFunction()
        )

        # specify the conversion from distances to weight method
        dist_to_weight_fct = NearestNeighborDistancesToWeightsFuncts(
            DKNN_args.K
        )
        if DKNN_args.dist_to_weight_fct not in dist_to_weight_fct.name_to_fct:
            raise ValueError("Unexpected Distances to Weights Function")
        dist_to_weight_fct = dist_to_weight_fct.get(
            DKNN_args.dist_to_weight_fct
        )

        # create the NearestNeighborLogit Function
        if DKNN_args.prediction_method == "conformal":
            logitsFactory = LogProbabilityLogitsFactory(label_list)
        elif DKNN_args.prediction_method == "nonconformal":
            # compute the conformal socres first
            computeAndSaveConformalScoresTrainer = ComputeAndSaveConformalScoresTrainer(
                model=model,
                args=training_args,
                caliberation_dataset=eval_data,
                data_collator=data_collator,
                layers_to_save=encoding_args.layers_to_save,
                poolers=poolers,
                tokenizer=tokenizer,
                nearestNeighborFunction=nearestNeighborFinderFunction,
                dist_to_weight_fct=dist_to_weight_fct,
                read_from_scores_path=DKNN_args.read_from_scores_path,
                save_nonconform_scores_path=DKNN_args.save_nonconform_scores_path,
            )
            scores = (
                computeAndSaveConformalScoresTrainer.compute_and_save_nonconformity_scores()
            )
            logitsFactory = ConformalLogitsFactory(label_list, scores)
        else:
            raise ValueError("Illegal Nearest Neighbor Prediction method")
        nearestNeighborLogitFunction = logitsFactory.createLogitsFunction()

        # TODO: consider different loss functions?
        # finally, create the complete Nearest Neighbor Classifier
        classifier = DeepKNearestNeighborClassifier(
            NearestNeighborFunction=nearestNeighborFinderFunction,
            dist_to_weight_fct=dist_to_weight_fct,
            LogitsFunction=nearestNeighborLogitFunction,
            LossFunction=torch.nn.functional.nll_loss,
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
            layers_to_save=encoding_args.layers_to_save,
            poolers=poolers,
            classifier=classifier,
            save_logits=data_args.save_logits,
            output_and_save_neighbors=DKNN_args.output_and_save_neighbors,
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_data)
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_data)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_data))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Test set Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        if (
            "t5" in model_args.model_name_or_path
            or "xlnet" in model_args.model_name_or_path
        ):
            if data_args.compute_predict_results:
                test_labels = test_data["label"]
            test_data = test_data.remove_columns("label")

            predict_results = trainer.predict(
                test_data,
                metric_key_prefix="predict",
                max_length=training_args.generation_max_length,
                num_beams=training_args.generation_num_beams,
            )
            if trainer.is_world_process_zero():
                if training_args.predict_with_generate:
                    predictions = tokenizer.batch_decode(
                        predict_results.predictions,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    predictions = [pred.strip() for pred in predictions]
                    prediction_ids = np.array(
                        list(
                            map(
                                lambda x: generative_label_to_id[x], predictions
                            )
                        )
                    )
                    output_prediction_file = os.path.join(
                        training_args.output_dir, "predict_results.txt"
                    )
                    with open(
                        output_prediction_file, "w", encoding="utf-8"
                    ) as writer:
                        logger.info(
                            "***** Writing Predict results to"
                            f" {output_prediction_file} *****"
                        )
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(prediction_ids):
                            writer.write(f"{index}\t{item}\n")

                    if data_args.compute_predict_results:
                        p = EvalPrediction(
                            predictions=predict_results.predictions,
                            label_ids=test_labels,
                        )
                        predict_metrics = compute_metrics_generative(p)
                        # To be JSON-serializable, we need to remove numpy types
                        # or zero-d tensors
                        predict_metrics = denumpify_detensorize(predict_metrics)
                        predict_metrics.update(predict_results.metrics)
                        predict_metrics["predict_samples"] = len(test_data)
                        # Prefix all keys with metric_key_prefix + '_'
                        for key in list(predict_metrics.keys()):
                            if not key.startswith("predict_"):
                                predict_metrics[
                                    f"predict_{key}"
                                ] = predict_metrics.pop(key)
                        trainer.log_metrics("predict", predict_metrics)
                        trainer.save_metrics("predict", predict_metrics)
        else:
            # Removing the `label` columns because it contains -1 and Trainer
            # won't like that.
            if data_args.compute_predict_results:
                test_labels = test_data["label"]
            test_data = test_data.remove_columns("label")
            prediction_output = trainer.predict(
                test_data, metric_key_prefix="predict"
            )
            predictions = prediction_output.predictions
            if (
                isinstance(predictions, tuple)
                and len(predictions) != test_data.num_rows
            ):
                # assume the first thing in the tuple is predictions,
                # if it's multiple tensors
                predictions = predictions[0]

            # compute results for test set (NOTE: this assumes that the test
            # set also has labels)
            if data_args.compute_predict_results:
                p = EvalPrediction(
                    predictions=predictions, label_ids=test_labels
                )
                predict_metrics = compute_metrics(p)
                # To be JSON-serializable, we need to remove numpy types or
                # zero-d tensors
                predict_metrics = denumpify_detensorize(predict_metrics)
                predict_metrics.update(prediction_output.metrics)
                predict_metrics["predict_samples"] = len(test_data)
                # Prefix all keys with metric_key_prefix + '_'
                for key in list(predict_metrics.keys()):
                    if not key.startswith("predict_"):
                        predict_metrics[f"predict_{key}"] = predict_metrics.pop(
                            key
                        )
                trainer.log_metrics("predict", predict_metrics)
                trainer.save_metrics("predict", predict_metrics)

            # predictions = np.argmax(predictions, axis=1)
            predictions = randargmax(predictions)  # break ties arbitrarily
            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    # finally, save the data arguments and DKNN arguments
    torch.save(
        data_args, os.path.join(training_args.output_dir, "data_args.bin")
    )
    torch.save(
        DKNN_args, os.path.join(training_args.output_dir, "DKNN_args.bin")
    )
    torch.save(
        encoding_args,
        os.path.join(training_args.output_dir, "encoding_args.bin"),
    )


if __name__ == "__main__":
    main()
