"""
Custom Trainer that wraps around a Transformer model from Huggingface, and
overrides test-time behavior as specifed in Papernot, Nicolas, and Patrick McDaniel
"""
from inspect import signature
from datasets import Dataset
from transformers import (
    DataCollator,
    EvalPrediction, 
    PreTrainedModel, 
    Trainer, 
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments
)
from transformers.trainer_utils import PredictionOutput
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import nested_detach
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import Callable, Union, Optional, Dict, List, Tuple, Any
from utils import (
    compute_layer_representations,
    find_signature_columns,
    get_hidden_states,
    remove_file_if_already_exists,
    remove_unused_columns,
    save_database_after_stacking,
    save_matrix_with_tags_to_file
)
from DeepKNearestNeighborClassifier import DeepKNearestNeighborClassifier
from datasets import Dataset
from transformers.utils import logging
from packaging import version
import torch.nn as nn
import torch
import gc
import os

logger = logging.get_logger(__name__)

class SaveLogitsTrainer(Trainer):
    """Custom Trainer that uses allows user to specify whether logits should be saved

    Args:
        Trainer (Transformers.Trainer): Override Original Trainer
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        save_logits: bool = False,
        save_reps_path: str = None,
        layers_to_save: List[int] = None,
        poolers_to_use: List[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]=None
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        """
        super().__init__(model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        torch.cuda.empty_cache() # save memory before iterating through dataset
        self.save_logits = save_logits
        self.save_logits_path = None
        self.save_reps_path = save_reps_path # if not None, then save
        self.layers_to_save = layers_to_save
        self.poolers_to_use = poolers_to_use
        self._signature_columns = find_signature_columns(model, args)
        self._signature_columns += ["tag"]

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, 
                 metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._create_save_files(prefix=metric_key_prefix)
        self.database = {}
        
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if self.save_reps_path is not None:
            logger.info("*** Saving Layer Representations ***")
            save_database_after_stacking(
                self.layers_to_save, self.save_reps_path, self.database
            )
        return results

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # print("Before hidden states:")
        # print(torch.cuda.memory_summary())
        if self.save_logits_path is not None and self.save_logits:
            tags = inputs.pop("tag").cpu().detach().numpy()
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if model.training or self.save_reps_path is None:
            outputs = model(**inputs)
        else:
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=self.save_reps_path is not None
                )
        attention_mask = inputs.get("attention_mask", None).detach().cpu()
        for k, tensor in inputs.items():
            del tensor
        del inputs
        if self.save_reps_path is not None:
            hidden_states = get_hidden_states(self.model.config.is_encoder_decoder, outputs)
            self.database = compute_layer_representations(
                self.model.config.is_encoder_decoder, hidden_states, attention_mask, 
                self.layers_to_save, self.poolers_to_use, self.database
            )
            if self.model.config.is_encoder_decoder:
                del outputs.encoder_hidden_states
                del outputs.decoder_hidden_states
                outputs.encoder_hidden_states = None
                outputs.decoder_hidden_states = None
            else:
                del outputs.hidden_states
                outputs.hidden_states = None
            del hidden_states
            # print("here!\n\n")
            # print("After hidden states:")
            gc.collect()
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary())
            
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.save_logits_path is not None and self.save_logits:
            save_matrix_with_tags_to_file(self.save_logits_path, tags, outputs['logits'].cpu().detach().numpy())
        return (loss, outputs) if return_outputs else loss

    def predict(self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, 
                metric_key_prefix: str = "test") -> PredictionOutput:
        self._create_save_files(prefix=metric_key_prefix)
        self.database = {}
        results = super().predict(test_dataset, ignore_keys, metric_key_prefix)
        if self.save_reps_path is not None:
            logger.info("*** Saving Layer Representations ***")
            save_database_after_stacking(
                self.layers_to_save, self.save_reps_path, self.database
            )
        return results

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            # eliminated sagemaker stuff - deprecated anyways
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    if self.save_logits:
                        tags = inputs.pop("tag").cpu().detach().numpy()
                    if model.training or self.save_reps_path is None:
                        outputs = model(**inputs)
                    else:
                        outputs = model(
                            **inputs, 
                            output_hidden_states=self.save_reps_path is not None
                        )
                    if self.save_reps_path is not None:
                        hidden_states = get_hidden_states(self.model.config.is_encoder_decoder, outputs)
                        if self.model.config.is_encoder_decoder:
                            del outputs.encoder_hidden_states
                            del outputs.decoder_hidden_states
                            outputs.encoder_hidden_states = None
                            outputs.decoder_hidden_states = None
                        else:
                            del outputs.hidden_states
                            outputs.hidden_states = None
                        # del hidden_states
                        attention_mask = inputs.get("attention_mask", None).detach().cpu()
                        self.database = compute_layer_representations(
                            self.model.config.is_encoder_decoder, hidden_states, attention_mask, 
                            self.layers_to_save, self.poolers_to_use, self.database
                        )
                        del hidden_states
                        torch.cuda.empty_cache()
                    # save logits
                    logits = outputs["logits"]
                    if self.save_logits:
                        save_matrix_with_tags_to_file(self.save_logits_path, tags, outputs['logits'].cpu().detach().numpy())
                    # save layer representations
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _remove_unused_columns(self, dataset: Dataset, description: Optional[str] = None):
        return remove_unused_columns(
            dataset=dataset, 
            model=self.model, 
            remove_unused_columns=True, 
            signature_columns=self._signature_columns, 
            description=description
        )

    def _create_save_files(self, prefix: str):
        if self.save_logits:
            self.save_logits_path = os.path.join(self.args.output_dir, f"{prefix}_logits.txt")
            remove_file_if_already_exists(self.save_logits_path)
