from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from utils import (
    compute_layer_representations,
    find_signature_columns,
    prepare_inputs,
    remove_unused_columns,
    get_hidden_states,
    get_pooled_layer_representations,
    hidden_states_to_cpu,
    save_database_after_stacking,
)
from transformers.data.data_collator import (
    default_data_collator,
    DataCollatorWithPadding,
)
from datasets import Dataset
from typing import Optional, Union, List, Dict, Callable, Tuple
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import time


class ComputeEncodings:
    """
    Compute the training, validation, and test set examples, if specified
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        layers_to_save: List[int] = None,
        poolers: List[Callable[[torch.tensor], torch.tensor]] = None,
        save_train_encodings_path: Optional[str] = None,
        save_eval_encodings_path: Optional[str] = None,
        save_test_encodings_path: Optional[str] = None,
    ):
        # The init function is modeled like Transformers.Trainer
        torch.cuda.empty_cache()  # save memory before iterating through dataset
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        default_collator = (
            default_data_collator
            if tokenizer is None
            else DataCollatorWithPadding(tokenizer)
        )
        self.tokenizer = tokenizer
        self.data_collator = (
            data_collator if data_collator is not None else default_collator
        )
        self.layers_to_save = layers_to_save
        self.poolers = poolers
        self.save_train_encodings_path = save_train_encodings_path
        self.save_eval_encodings_path = save_eval_encodings_path
        self.save_test_encodings_path = save_test_encodings_path
        self._signature_columns = find_signature_columns(model, args)

    def compute_and_save_encodings(self):
        """
        Driver code
        """
        for dataset, path in zip(
            [self.train_dataset, self.eval_dataset, self.test_dataset],
            [
                self.save_train_encodings_path,
                self.save_eval_encodings_path,
                self.save_test_encodings_path,
            ],
        ):
            if path is not None:
                database = self.compute_encodings(dataset)
                self.save_encodings(database, path)

    def compute_encodings(self, dataset: Dataset) -> Dict[int, np.ndarray]:
        """
        Following Antigoni Maria Founta et. al. - we make one more pass through the dataset
        and save specified layers' representations (currently in np arrays)

        Args:
            dataset (Dataset): the input data to compute encodings for
        """
        dataloader = DataLoader(
            remove_unused_columns(
                dataset, self.model, True, self._signature_columns
            ),
            shuffle=False,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
        )
        progress_bar = tqdm(range(len(dataloader)))
        self.model.cuda().eval()
        # Column metadata: [0-self.layer_dim = representation, label of this example, tag (index in train data)]
        # total dim when finished = (training size) x (self.layer_dim + 3) per array, a total of len(self.layers_to_save) np.arrays
        database = {}
        print("***** Computing Layer Representations *****")
        for batch in dataloader:
            inputs = prepare_inputs(
                batch, self._signature_columns, self.args.device
            )
            # drop label to avoid risk of data leak
            attention_mask = inputs.get("attention_mask", None).detach().cpu()
            with torch.no_grad():
                if "t5" in self.args.output_dir:
                    del inputs["labels"]
                    outputs = self.model.generate(
                        **inputs,
                        # decoder_input_ids = torch.zeros()
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )
                    decoder_layers = len(outputs["decoder_hidden_states"][0])
                    tensors = []
                    for i in range(decoder_layers):
                        token_rep_stacked = torch.hstack(
                            [t[i] for t in outputs["decoder_hidden_states"]]
                        )
                        tensors.append(token_rep_stacked)
                    outputs["decoder_hidden_states"] = tuple(tensors)
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = get_hidden_states(
                self.model.config.is_encoder_decoder, outputs
            )
            # Hidden-states of the model = the initial embedding outputs + the output of each layer
            # filter representations to what we need: (num_layers+1, batch_size, max_seq_len, embedding_dim)
            # for layer, pooler in zip(self.layers_to_save, self.poolers):
            #     layer_rep_np = pooler(hidden_states[layer], attention_mask)
            #     database[layer] = (np.append(database[layer], layer_rep_np, axis=0)
            #                        if database[layer] is not None else layer_rep_np)
            database = compute_layer_representations(
                self.model.config.is_encoder_decoder,
                hidden_states,
                attention_mask,
                self.layers_to_save,
                self.poolers,
                database,
            )
            progress_bar.update(1)
        return database

    def save_encodings(self, database: Dict[int, np.ndarray], path: str):
        print("***** Saving Layer Representations *****")
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
        save_database_after_stacking(self.layers_to_save, path, database)
