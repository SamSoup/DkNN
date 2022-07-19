"""
Custom Trainer that wraps around a Transformer model from Huggingface, and
overrides test-time behavior as specifed in Papernot, Nicolas, and Patrick McDaniel
"""

from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from packaging import version
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
from utils import save_training_points_representations
from DkNN import DkNN_KD_TREE, DkNN_LSH
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import datasets
import inspect

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DkNNTrainer(Trainer):
    """Custom Trainer for Deep K Nearest Neighbor Approach

    Args:
        Trainer (Transformers.Trainer): Override Original Trainer
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        DkNN_method: str = "KD-Tree",
        num_neighbors: int = 10,
        label_list: List[Any] = None,
        layers_to_save: List[int] = [],
        save_database_path: Optional[str] = None
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        
        Args:
            DkNN_method: (str): which DkNN method to apply; must be one of {KD-Tree, LSH}
            label_list (List[Any], optional): the list of all possible labels (y's) in training Dataset. Defaults to None.
            layers_to_save (List[int], optional): the list of layer (indices) to save. Defaults to [].
            num_neighbors (int, optional): how many neighbors to examine per inference example. Defaults to 10.
            save_database_path (Optional[str], optional): path to save layer representations. Defaults to None.
        """

        Trainer.__init__(self, model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        if DkNN_method == "KD-Tree":
            self.DkNNClassifier = DkNN_KD_TREE(num_neighbors, label_list, layers_to_save, self.model.config.hidden_size)
        elif DkNN_method == "LSH":
            self.DkNNClassifier = DkNN_LSH(num_neighbors, label_list, layers_to_save, self.model.config.hidden_size)
        torch.cuda.empty_cache() # save memory before iterating through dataset
        train_dataloader = DataLoader(self._remove_unused_columns(train_dataset), shuffle=True, 
                                      batch_size=self.args.train_batch_size, 
                                      collate_fn=self.data_collator)
        save_training_points_representations(train_dataloader, layers_to_save, save_database_path, self.model)

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     return_outputs: bool = False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        We override this function to utilize our custom loss computation
        """

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # NOTE: hidden_states occpuy a LOT of cuda memory, 
        # print("Before calling model on inputs")
        # print(torch.cuda.memory_summary())
        # input()
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = []
        for state in outputs["hidden_states"]:
            hidden_states.append(state.cpu())
            del state
        torch.cuda.empty_cache()
        # print("After calling model on inputs")
        # print(torch.cuda.memory_summary())
        # input()        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        neighbors = self.DkNNClassifier.nearest_neighbors(hidden_states)
        loss, logits = self.DkNNClassifier.compute_loss_and_logits_DkNN(neighbors, labels)
        outputs = {
            "loss": loss,
            "logits": logits
        }
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def _remove_unused_columns(self, dataset: Dataset, description: Optional[str] = None):
        """
        Trainer's internal function that drops extraenous keys from the dataset,
        I have modified this function to not drop the unique tag associated with each example
        s.t. I can know exactly which example was selected for labeling & to-train on
        """
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
        
        # CUSTOM BEHAVIOR: keep the tag field also
        self._signature_columns += ["tag"]

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            print(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                f" you can safely ignore this message."
            )

        columns = [k for k in self._signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)