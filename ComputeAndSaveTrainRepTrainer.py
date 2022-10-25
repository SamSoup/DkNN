from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments
)
from transformers.data.data_collator import default_data_collator, DataCollatorWithPadding
from datasets import Dataset
from typing import Optional, Union, List, Dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import get_layer_representations, find_signature_columns, prepare_inputs, remove_unused_columns, get_hidden_states, get_pooled_layer_representations
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import time

class ComputeAndSaveTrainRepTrainer:
    """
    Trainer Dedicated to Save Training Example Layer Representations.
    However, this does NOT inhert from Transformers.Trainer because it does not 
    need its entire functionality. Rather, this only takes in arguments that are
    sufficent to run a single iteration through the training dataset and save
    the specified layer representations for each example. Nevertheless, it bears
    quite a bit similarity w.r.t Transformers.Trainer.
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        layers_to_save: List[int] = [],
        read_from_database_path: bool = False,
        save_database_path: Optional[str] = None,
    ):
        torch.cuda.empty_cache() # save memory before iterating through dataset
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.layers_to_save = layers_to_save
        self.read_from_database_path = read_from_database_path
        self.save_database_path = save_database_path
        self._signature_columns = find_signature_columns(model, args)
        self._signature_columns += ["tag"] # also keep tag in training dataset only

    def compute_and_save_training_points_representations(self) -> Dict[int, np.ndarray]:
        """
        Following Antigoni Maria Founta et. al. - we make one more pass through the training set
        and save specified layers' representations in a database (for now simply a DataFrame, may
        migrate to Spark if necessary) stored in memory

        Args:
            train_dataset (Dataset): the dataset used to train the model wrapped by a dataloader
            layers_to_save (List[int]): the list of layers to save
            save_database_path (Optional[str]): directory to save the layer representations, if not None
        """
        if self.read_from_database_path:
            print("***** Running DkNN - Loading database of layer representation from specified path *****")
            start = time.time()
            database = { 
                layer: np.loadtxt(os.path.join(self.save_database_path, f"layer_{layer}.csv"), 
                                    delimiter=",") 
                for layer in self.layers_to_save 
            }
            end = time.time()
            print(f"Initializing tables took {end - start}")
            return database

        print("***** Running DkNN - Computing Layer Representations for Training Examples *****")
        train_dataloader = DataLoader(
            remove_unused_columns(self.train_dataset, self.model, True, self._signature_columns),
            shuffle=True, batch_size=self.args.train_batch_size, collate_fn=self.data_collator)
        progress_bar = tqdm(range(len(train_dataloader)))
        self.model.eval()
        # Column metadata: [0-self.layer_dim = representation, label of this example, tag (index in train data)]
        # total dim when finished = (training size) x (self.layer_dim + 3) per array, a total of len(self.layers_to_save) np.arrays
        database = { layer: None for layer in self.layers_to_save }
        for batch in train_dataloader:
            inputs = prepare_inputs(batch, self._signature_columns, self.args.device)
            tags = inputs.pop("tag").cpu().numpy()
            labels = inputs.pop("labels").cpu().numpy()
            attention_mask = inputs["attention_mask"]
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = get_hidden_states(self.model.config.is_encoder_decoder, outputs)
            # Hidden-states of the model = the initial embedding outputs + the output of each layer                            
            # filter representations to what we need: (num_layers+1, batch_size, max_seq_len, embedding_dim)
            for layer in self.layers_to_save:
                # layer_rep_np = get_layer_representations(hidden_states[layer])
                layer_rep_np = get_pooled_layer_representations(hidden_states[layer], attention_mask)
                layer_rep_np = np.concatenate(
                    (layer_rep_np, tags.reshape(-1, 1), labels.reshape(-1, 1)), axis=1) # (batch_size, embedding_dim + 2)
                database[layer] = (np.append(database[layer], layer_rep_np, axis=0) 
                                   if database[layer] is not None else layer_rep_np)
            progress_bar.update(1)
        if self.save_database_path is not None:
            print("***** Running DkNN - Saving Layer Representations for Training Examples *****")
            if os.path.exists(self.save_database_path) and os.path.isdir(self.save_database_path):
                shutil.rmtree(self.save_database_path)
            os.makedirs(self.save_database_path)
            progress_bar = tqdm(range(len(self.layers_to_save)))
            for layer in self.layers_to_save:
                save_file_name = os.path.join(self.save_database_path, f"layer_{layer}.csv")
                np.savetxt(save_file_name, database[layer], delimiter=",")
                progress_bar.update(1)
        return database

