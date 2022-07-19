from datasets import Dataset
from typing import Optional, List, Dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import numpy as np
import shutil
import os

def get_layer_representations(self, hidden_states: torch.tensor) -> torch.tensor:
    """
    Obtain the layer representations by averaging across all tokens to obtain the embedding
    at sentence level (for now). This method may be changed if necessary.
    
    Args:
        hidden_states (torch.tensor): the hidden states from the model for one layer, with shape
        (batch_size, max_seq_len, embedding_dim)

    Returns:
        torch.tensor: (batch_size, self.layer_dim) of layer representations in order
    """
    # average across all tokens to obtain embedding -> (batch_size, embedding_dim)
    return torch.mean(hidden_states, dim=1).squeeze().detach().cpu()

def save_training_points_representations(self, train_dataloader: DataLoader, layers_to_save: List[int], 
                                         save_database_path: Optional[str], model: nn.Module) -> Dict[int:np.array]:
    """
    Following Antigoni Maria Founta et. al. - we make one more pass through the training set
    and save specified layers' representations in a database (for now simply a DataFrame, may
    migrate to Spark if necessary) stored in memory

    Args:
        train_dataloader (DataLoader): the dataset used to train the model wrapped by a dataloader
        layers_to_save (List[int]): the list of layers to save
        save_database_path (Optional[str]): directory to save the layer representations, if not None
    """

    print("***** Running DkNN - Computing Layer Representations *****")
    progress_bar = tqdm(range(len(train_dataloader)))
    model.eval()
    # Column metadata: [0-self.layer_dim = representation, label of this example, tag (index in train data)]
    # total dim when finished = (training size) x (self.layer_dim + 3) per df, a total of len(self.layers_to_save) dataframes
    # dict { layer_num: pd.Dataframe of layer representations for all training examples}
    database = { layer: None for layer in layers_to_save }
    for batch in train_dataloader:
        print(batch)
        input()
        tags = batch.pop("tag").cpu().numpy()
        labels = batch.pop("label").cpu().numpy()
        print(labels)
        input()
        # inputs = self._prepare_inputs(batch)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)                   # dict: {'loss', 'logits', 'hidden_states'}
        hidden_states = outputs['hidden_states']
        # Hidden-states of the model = the initial embedding outputs + the output of each layer                            
        # filter representations to what we need: (num_layers+1, batch_size, max_seq_len, embedding_dim)
        for layer in self.layers_to_save:
            layer_rep_np = get_layer_representations(hidden_states[layer])
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
    return database
