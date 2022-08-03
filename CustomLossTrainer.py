"""
Custom Trainer that wraps around a Transformer model from Huggingface, and
utilizes weighted cross-entropy loss to aid against imabalanced class datasets
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

class CustomLossTrainer(Trainer):
    """Custom Trainer that uses Custom Loss Function

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
        loss_fct: Optional[Callable[[torch.tensor, torch.tensor], torch.tensor]] = None,
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        """
        super().__init__(model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        self.loss_fct = loss_fct

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
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
