"""
Custom Trainer that wraps around a Transformer model from Huggingface, and
overrides test-time behavior as specifed in Papernot, Nicolas, and Patrick McDaniel
"""
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
from typing import Callable, Union, Optional, Dict, List, Tuple, Any
from DeepKNearestNeighborClassifier import DeepKNearestNeighborClassifier
from datasets import Dataset
import torch.nn as nn
import torch

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
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        """
        super().__init__(model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        torch.cuda.empty_cache() # save memory before iterating through dataset
        self.classifier = classifier

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     return_outputs: bool = False):
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
        # NOTE: hidden_states occpuy a LOT of cuda memory, so we need to 
        # delete them from the GPU immediately after retrieval
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = []
        for state in outputs["hidden_states"]:
            hidden_states.append(state.cpu())
            del state
        torch.cuda.empty_cache()     
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        loss, logits = self.classifier.compute_loss_and_logits(hidden_states, labels, self.args.device)
        outputs = { "loss": loss, "logits": logits }
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
