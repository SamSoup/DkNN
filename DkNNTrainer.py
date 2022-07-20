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
from utils import get_layer_representations, compute_nonconformity_score
from DkNN import DkNN_KD_TREE, DkNN_LSH
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import datasets
import inspect
import shutil
import os

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
        read_from_database_path: bool = False,
        save_database_path: Optional[str] = None,
        read_from_scores_path: bool = False,
        save_nonconform_scores_path: Optional[str] = None
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        
        Args:
            DkNN_method: (str): which DkNN method to apply; must be one of {KD-Tree, LSH}
            label_list (List[Any], optional): the list of all possible labels (y's) in training Dataset. Defaults to None.
            layers_to_save (List[int], optional): the list of layer (indices) to save. Defaults to [].
            num_neighbors (int, optional): how many neighbors to examine per inference example. Defaults to 10.
            save_database_path (Optional[str], optional): path to save layer representations. Defaults to None.
            save_nonconform_scores_path (Optional[str], optional): path to save nonconformity scores for evaluation data. Defaults to None.
        """

        Trainer.__init__(self, model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        torch.cuda.empty_cache() # save memory before iterating through dataset
        if read_from_database_path:
            print("***** Loading database of layer representation from specified path *****")
            database = { layer: np.loadtxt(os.path.join(save_database_path, f"layer_{layer}.csv"), delimiter=",") for layer in layers_to_save }
        else:
            database = self.save_training_points_representations(train_dataset, layers_to_save, save_database_path)
        if DkNN_method == "KD-Tree":
            self.DkNNClassifier = DkNN_KD_TREE(num_neighbors, layers_to_save, database, self.model.config.hidden_size, label_list)
        elif DkNN_method == "LSH":
            self.DkNNClassifier = DkNN_LSH(num_neighbors, layers_to_save, database, self.model.config.hidden_size, label_list)
        if read_from_scores_path:
            print("***** Loading scores from specified path *****")
            self.DkNNClassifier.scores = np.loadtxt(save_nonconform_scores_path, delimiter=",")
        else:
            label_to_id = {label: i for i, label in enumerate(label_list)}
            self.DkNNClassifier.scores = self.compute_nonconformity_score_for_caliberation_set(eval_dataset, save_nonconform_scores_path, label_to_id)

    def save_training_points_representations(self, train_dataset: Dataset, layers_to_save: List[int], 
                                             save_database_path: Optional[str]) -> Dict[int, np.array]:
        """
        Following Antigoni Maria Founta et. al. - we make one more pass through the training set
        and save specified layers' representations in a database (for now simply a DataFrame, may
        migrate to Spark if necessary) stored in memory

        Args:
            train_dataloader (DataLoader): the dataset used to train the model wrapped by a dataloader
            layers_to_save (List[int]): the list of layers to save
            save_database_path (Optional[str]): directory to save the layer representations, if not None
        """

        print("***** Running DkNN - Computing Layer Representations for Training Examples *****")
        train_dataloader = DataLoader(self._remove_unused_columns(train_dataset), shuffle=True, 
                                        batch_size=self.args.train_batch_size, 
                                        collate_fn=self.data_collator)
        progress_bar = tqdm(range(len(train_dataloader)))
        self.model.eval()
        # Column metadata: [0-self.layer_dim = representation, label of this example, tag (index in train data)]
        # total dim when finished = (training size) x (self.layer_dim + 3) per array, a total of len(self.layers_to_save) np.arrays
        database = { layer: None for layer in layers_to_save }
        for batch in train_dataloader:
            inputs = self._prepare_inputs(batch)
            tags = inputs.pop("tag").cpu().numpy()
            labels = inputs.pop("labels").cpu().numpy()
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)                   # dict: {'loss', 'logits', 'hidden_states'}
            hidden_states = outputs['hidden_states']
            # Hidden-states of the model = the initial embedding outputs + the output of each layer                            
            # filter representations to what we need: (num_layers+1, batch_size, max_seq_len, embedding_dim)
            for layer in layers_to_save:
                layer_rep_np = get_layer_representations(hidden_states[layer])
                layer_rep_np = np.concatenate(
                    (layer_rep_np, tags.reshape(-1, 1), labels.reshape(-1, 1)), axis=1)     # (batch_size, embedding_dim + 2)
                database[layer] = (np.append(database[layer], layer_rep_np, axis=0) 
                                   if database[layer] is not None else layer_rep_np)
            progress_bar.update(1)
        if save_database_path is not None:
            print("***** Running DkNN - Saving Layer Representations for Training Examples *****")
            if os.path.exists(save_database_path) and os.path.isdir(save_database_path):
                shutil.rmtree(save_database_path)
            os.makedirs(save_database_path)
            progress_bar = tqdm(range(len(layers_to_save)))
            for layer in layers_to_save:
                save_file_name = os.path.join(save_database_path, f"layer_{layer}.csv")
                np.savetxt(save_file_name, database[layer], delimiter=",")
                progress_bar.update(1)
        return database

    def compute_nonconformity_score_for_caliberation_set(self, eval_data: Dataset, save_nonconform_scores_path: str, 
                                                         label_to_id: Dict[Any, int]):
        """
        We compute and store all non-conformity scores for the caliberation set
        as an numpy array of dimensions (# of example in caliberation, # of possible labels)
        
        Args:
            eval_data (Dataset): caliberation dataset
            save_nonconform_scores_path (Optional[str], optional): path to save nonconformity scores for evaluation data. Defaults to None.
            label_to_id (Dict[Any, int]): dictionary mapping from original format of label to some integer id
        """

        nonconformity_scores = np.zeros(len(eval_data))
        print("***** Running DkNN - Computing Nonconformity Scores for Caliberation Data *****")
        eval_dataloader = DataLoader(self._remove_unused_columns(eval_data), shuffle=True, 
                                        batch_size=self.args.eval_batch_size, 
                                        collate_fn=self.data_collator)
        progress_bar = tqdm(range(len(eval_dataloader)))
        self.model.eval()
        for i, batch in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(batch)
            labels = inputs.pop("labels").cpu().numpy()
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)                   # dict: {'loss', 'logits', 'hidden_states'}
            hidden_states = outputs['hidden_states']
            neighbors = self.DkNNClassifier.nearest_neighbors(hidden_states)
            for j, label in enumerate(labels):
                label_id = label_to_id[label]
                nonconform_score = compute_nonconformity_score(neighbors[i, :].reshape(1, -1), label_id)
                nonconformity_scores[i*self.args.eval_batch_size + j] = nonconform_score
            progress_bar.update(1)
        if save_nonconform_scores_path is not None:
            print("***** Running DkNN - Saving Layer Representations *****")
            np.savetxt(save_nonconform_scores_path, nonconformity_scores, delimiter=",")
        return nonconformity_scores

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     return_outputs: bool = False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        We override this function to utilize our custom loss computation

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
        # delete them from the GPU when necessary
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = []
        for state in outputs["hidden_states"]:
            hidden_states.append(state.cpu())
            del state
        torch.cuda.empty_cache()     
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        neighbors = self.DkNNClassifier.nearest_neighbors(hidden_states)
        loss, logits = self.DkNNClassifier.compute_loss_and_logits_DkNN(neighbors, self.scores,labels)
        outputs = { "loss": loss, "logits": logits }
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def _remove_unused_columns(self, dataset: Dataset, description: Optional[str] = None) -> Dataset:
        """
        Trainer's internal function that drops extraenous keys from the dataset,
        I have modified this function to not drop the unique tag associated with each example
        s.t. I can know exactly which example was selected for labeling & to-train on

        Args:
            dataset (Dataset): the input dataset to strip
            description (Optional[str], optional): description of the dataset. Defaults to None.

        Returns:
            Dataset: the cleaned dataset without unexpected arguments for the model to call forward on
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
