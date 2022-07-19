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
    TrainingArguments,
    default_data_collator
)
from typing import Callable, Union, Optional, Dict, List, Tuple, Any
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import inspect
import datasets
import numpy as np
import pandas as pd
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
        label_list: List[Any] = None,
        layers_to_save: List[int] = [],
        num_neighbors: int = 10,
        save_database_path: Optional[str] = None
    ):
        """
        Arguments follows directly from Transformers.Trainer except the following:
        
        Args:
            label_list (List[Any], optional): the list of all possible labels (y's) in training Dataset. Defaults to None.
            layers_to_save (List[int], optional): the list of layer (indices) to save. Defaults to [].
            num_neighbors (int, optional): how many neighbors to examine per inference example. Defaults to 10.
            save_database_path (Optional[str], optional): path to save layer representations. Defaults to None.
        """

        Trainer.__init__(self, model, args, data_collator, train_dataset, 
                            eval_dataset, tokenizer, model_init, compute_metrics,
                            callbacks, optimizers, preprocess_logits_for_metrics)
        self.label_list = label_list
        self.label_to_id = {label: i for i, label in enumerate(label_list)}
        self.layer_dim = self.model.config.hidden_size
        # extract the input signature of this model to know which keyword arguments it expects
        self.input_args = inspect.signature(self.model.forward).parameters.keys()
        self.layers_to_save = layers_to_save
        self.k = num_neighbors # number of nearest neighbors to obtain per sample per layer
        torch.cuda.empty_cache() # save memory before iterating through dataset
        self.save_training_points_representations(train_dataset, save_database_path)

    def get_layer_representations(self, hidden_states: torch.tensor) -> torch.tensor:
        """
        Obtain the layer representations by averaging across all tokens to obtain the embedding
        at sentence level
        
        Args:
            hidden_states (torch.tensor): the hidden states from the model, with shape
            (batch_size, max_seq_len, embedding_dim)

        Returns:
            torch.tensor: (batch_size, self.layer_dim) of layer representations in order
        """
        # average across all tokens to obtain embedding -> (batch_size, embedding_dim)
        return torch.mean(hidden_states, dim=1).squeeze().detach().cpu()

    def save_training_points_representations(self, train_dataset: Dataset, save_database_path: Optional[str]):
        """
        Following Antigoni Maria Founta et. al. - we make one more pass through the training set
        and save specified layers' representations in a database (for now simply a DataFrame, may
        migrate to Spark if necessary) stored in memory

        Args:
            train_data (Dataset): the dataset used to train the model
            save_database_path (Optional[str]): directory to save the layer representations, if not None
        """

        train_dataset = self._remove_unused_columns(train_dataset)
        train_dataloader = DataLoader(train_dataset, shuffle=True, 
                                      batch_size=self.args.train_batch_size, 
                                      collate_fn=self.data_collator)
        print("***** Running DkNN - Computing Layer Representations *****")
        progress_bar = tqdm(range(len(train_dataloader)))
        self.model.eval()
        # Column metadata: [0-self.layer_dim = representation, label of this example, tag (index in train data)]
        # total dim when finished = (training size) x (self.layer_dim + 3) per df, a total of len(self.layers_to_save) dataframes
        # dict { layer_num: pd.Dataframe of layer representations for all training examples}
        self.database = {
            layer: None for layer in self.layers_to_save
        }
        for batch in train_dataloader:
            tags = batch['tag'].cpu().numpy()
            del batch['tag']
            inputs = self._prepare_inputs(batch)            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)                   # dict: {'loss', 'logits', 'hidden_states'}
            hidden_states = outputs['hidden_states']
            # Hidden-states of the model = the initial embedding outputs + the output of each layer                            
            # filter representations to what we need: (num_layers+1, batch_size, max_seq_len, embedding_dim)
            for layer in self.layers_to_save:
                layer_rep_np = self.get_layer_representations(hidden_states[layer])
                layer_rep_np = np.concatenate(
                    (layer_rep_np, tags.reshape(-1, 1), 
                     np.array(train_dataset.select(tags)['label']).reshape(-1, 1)), axis=1) # (batch_size, embedding_dim + 2)
                self.database[layer] = (np.append(self.database[layer], layer_rep_np, axis=0) 
                                        if self.database[layer] is not None else layer_rep_np)
            progress_bar.update(1)
        if save_database_path is not None:
            print("***** Running DkNN - Saving Layer Representations *****")
            if os.path.exists(save_database_path) and os.path.isdir(save_database_path):
                shutil.rmtree(save_database_path)
            os.makedirs(save_database_path)
            progress_bar = tqdm(range(len(self.layers_to_save)))
            for layer in self.layers_to_save:
                save_file_name = os.path.join(save_database_path, f"layer_{layer}.csv")
                np.savetxt(save_file_name, self.database[layer], delimiter=",")
                progress_bar.update(1)
        # construct the pool of trees to search
        self.trees = {
            layer: KDTree(self.database[layer][:, :self.layer_dim], metric="l2") 
                for layer in self.layers_to_save
        }

    def nearest_neighbors(self, hidden_states: Tuple[torch.tensor]):
        """
        Andoni et. al. https://arxiv.org/pdf/1509.02897.pdf
        Possible algorithms:
        
        1. space partitioning with indexing
        2. dimension reduction 
        3. sketching 
        4. Locality-Sensitive Hashing (LSH): sub-linear query time and sub-quadratic space complexity
            - idea: bucket similar samples through a hash function so as to maximize collision for
            similar vectors (layer representations here)
            - locality-sensitive hash functions: probability of collision is higher for "nearby" points 
            than for points that are far apart
            - how sensitive depends on the rho = \frac{log 1/p_1}{log 1/p_2}, where p_1 and p_2 are the
            collision probability for near and far points (depending on some distance r)
            - cross-polytope vs. hyperplane LSH
        5. KDTree: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree
        """
        print("***** Running DkNN - Nearest Neighbor Search (One Batch) *****")
        # for each example in the batch, find their total list of neighbors
        # the batch size may be not equivalent to self.args.eval_batch_size, if the 
        # number of training examples is *not* an exact multiple of self.args.eval_batch_size
        # so we use the first layer hidden state's first dimension (presumed batch size)
        neighbors = np.zeros((hidden_states[0].shape[0], len(self.layers_to_save) * self.k))
        progress_bar = tqdm(range(len(self.layers_to_save)))
        for l, layer in enumerate(self.layers_to_save):
            tree = self.trees[layer]
            dist, batch_indices = tree.query(self.get_layer_representations(hidden_states[layer]), k=self.k) # (batch_size, k)
            # based on idx, look up tag in database per batch of example
            for i, neighbor_indices in enumerate(batch_indices):
                labels = self.database[layer][neighbor_indices][:, -1] # label is the last element of the 2d np array
                labels_ids = list(map(lambda l: self.label_to_id[l], labels))
                neighbors[i, l * self.k : (l+1) * self.k] = labels_ids
            progress_bar.update(1)
        return neighbors

    def compute_loss_and_logits_DkNN(self, neighbors: np.ndarray, labels: Optional[torch.tensor]):
        """
        Compute loss and logits using the nearest k neighbors

        Args:
            neighbors (np.ndarray): an array of (batch_size, num of layers * k)
        """
        # TODO
        # first find the log-probabilities for each class 
        probs = np.zeros((neighbors.shape[0], len(self.label_list)))
        for i, label in enumerate(self.label_list):
            label_id = self.label_to_id[label]
            prob = (neighbors == label_id).sum(axis=1) / neighbors.shape[1]
            probs[:, i] = prob
        logits = torch.log(torch.from_numpy(probs).to(device)) # (self.args.eval_batch_size, len(self.label_list))
        if labels is not None:
            # Negative log likelihood loss for C potential classes
            loss = F.nll_loss(logits, labels) # torch.tensor(len(self.label_list))
        else:
            loss = None
        # TODO: Note that here we elected to do NLL loss and is may NOT be what the
        # model is expecting; but since DkNN does not rely on back-prop (yet), 
        # this is not an issue YET
        # TODO: infinite loss
        return loss, logits
 
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     return_outputs: bool = False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
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
        neighbors = self.nearest_neighbors(hidden_states)
        loss, logits = self.compute_loss_and_logits_DkNN(neighbors, labels)
        outputs = {
            "loss": loss,
            "logits": logits
        }
        if labels is not None and self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # print(torch.cuda.memory_summary())
        # input()
        return (loss, outputs) if return_outputs else loss

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
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
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
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
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