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
from transformers.trainer_pt_utils import nested_detach
from typing import Callable, Union, Optional, Dict, List, Tuple, Any
from utils import get_hidden_states, hidden_states_to_cpu, find_signature_columns, remove_file_if_already_exists, remove_unused_columns, save_matrix_with_tags_to_file
from DeepKNearestNeighborClassifier import DeepKNearestNeighborClassifier
from datasets import Dataset
from packaging import version
import torch.nn as nn
import torch
import os
    
class DeepKNearestNeighborTrainer(Trainer):
    """Custom Trainer that uses a Deep K Nearest Neighbor Classifier

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
        classifier: DeepKNearestNeighborClassifier = None,
        save_logits: bool = False,
        output_and_save_neighbor_ids: bool = False,        
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        """
        super().__init__(model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        torch.cuda.empty_cache() # save memory before iterating through dataset
        self.classifier = classifier
        self.save_logits = save_logits
        self.output_and_save_neighbor_ids = output_and_save_neighbor_ids
        self._signature_columns = find_signature_columns(model, args)
        self._signature_columns += ["tag"]

    def DKNN_predict(self, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     labels: torch.Tensor, model: nn.Module) -> Dict[str, torch.Tensor]:
        if self.save_logits or self.output_and_save_neighbor_ids:
            tags = inputs.pop("tag").cpu().detach().numpy()
        outputs = model(**inputs, output_hidden_states=True)
        is_encoder_decoder = (model.module.config.is_encoder_decoder if type(model) == nn.parallel.DataParallel 
                            else model.config.is_encoder_decoder)
        hidden_states = get_hidden_states(is_encoder_decoder, outputs)
        # NOTE: hidden_states occpuy a LOT of cuda memory, so we need to 
        # delete them from the GPU immediately after retrieval
        hidden_states = hidden_states_to_cpu(hidden_states)
        torch.cuda.empty_cache()
        loss, logits, neighbor_ids = self.classifier.compute_loss_and_logits(
            hidden_states, labels, self.args.device, self.output_and_save_neighbor_ids
        )
        # save the metadatas, if we should - logits, neighbors
        if self.save_logits:
            save_matrix_with_tags_to_file(self.save_logits_path, tags, logits.cpu().detach().numpy())
        if self.output_and_save_neighbor_ids:
            save_matrix_with_tags_to_file(self.save_neighbor_ids_path, tags, neighbor_ids)
        outputs = { "loss": loss, "logits": logits }
        return outputs

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, 
                 metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._create_save_files(prefix=metric_key_prefix)
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        We override this function to utilize our custom classifer incorporated as well as detach
        the hidden states from GPU to save cuda memory

        Args:
            model (nn.Module): the model for which to compute logits from
            inputs (Dict[str, Union[torch.Tensor, Any]]): the input batch of example tensors
            return_outputs (bool, optional): should we return logits along with loss. Defaults to False.

        Returns:
            Either a tuple of (loss, logits) or just the loss when return_outputs if False
        """

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # NOTE: use DKNN at eval time
        outputs = self.DKNN_predict(inputs, labels, model)
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def predict(self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, 
                metric_key_prefix: str = "test") -> PredictionOutput:
        self._create_save_files(prefix=metric_key_prefix)
        return super().predict(test_dataset, ignore_keys, metric_key_prefix)

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
                    # NOTE: use DKNN at predict time
                    logits = tuple(self.DKNN_predict(inputs, labels, model)["logits"])
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
        if self.output_and_save_neighbor_ids:
            self.save_neighbor_ids_path = os.path.join(self.args.output_dir, f"{prefix}_neighbors.txt")
            remove_file_if_already_exists(self.save_neighbor_ids_path)
        if self.save_logits:
            self.save_logits_path = os.path.join(self.args.output_dir, f"{prefix}_logits.txt")
            remove_file_if_already_exists(self.save_logits_path)
