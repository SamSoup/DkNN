from inspect import signature
from typing import List, Dict, Union, Any
from datasets import Dataset
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# TODO: figure out how to use LSh
# TODO: figure out how to finish DkNN 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DkNN(nn.Module):
    """
    The Deep K Nearest Neighbor Wrapper that's robust with respect to the underlying
    model architecture; All that this class expects is that the model must be a
    Pytorch model (for now), but really it expects a callable with the following
    interface:
    """

    def __init__(self, model: nn.Module, layers_to_save: List[int], train_data: Dataset, 
                 per_device_eval_batch_size: int):
        super().__init__()
        self.model = model.to(device)
        self.layer_dim = self.model.config.hidden_size
        # extract the input signature of this model to know which keyword arguments it expects
        self.args = signature(self.model.forward).parameters.keys()
        self.layers_to_save = layers_to_save
        # self.save_training_points_representations(train_data, per_device_eval_batch_size)
        # print(self.database)
        # self.database.to_csv("./data/layer_representation_database.csv", header=True, index=False)

    def save_training_points_representations(self, train_data: Dataset, batch_size: int):
        """
        Following Antigoni Maria Founta et. al. - we make one more pass through the training set
        and save specified layers' representations in a database (for now simply a DataFrame, may
        migrate to Spark if necessary) stored in memory

        Args:
            train_data (Dataset): the dataset used to train the model
            batch_size (int): how many samples to evaluate at once (from cmd line args)
        """

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        progress_bar = tqdm(range(len(train_dataloader)))
        self.model.eval()
        # 0-self.layer_dim = representation, layer #, label of this example, tag (index in train data)
        # total dim when finished = (training size * # of layers) x (self.layer_dim + 3)
        representations_df = pd.DataFrame(columns = [*range(self.layer_dim), "layer", "label", "tag"])
        for batch in train_dataloader:
            inputs = {k: torch.stack(v, dim=1) for k, v in batch.items() if k in self.args}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)               # dict: {'loss', 'logits', 'hidden_states'}
                # Hidden-states of the model = the initial embedding outputs + the output of each layer
                hidden_states = outputs['hidden_states']                                # (num_layers+1, batch_size, max_seq_len, embedding_dim)
                # filter representations to what we need
                for layer in self.layers_to_save:
                    # average across all tokens to obtain embedding
                    rep = torch.mean(hidden_states[layer], dim=1).squeeze().detach()    # (batch_size, embedding_dim)
                    df = pd.DataFrame(rep.numpy())
                    df['layer'] = layer
                    df['tag'] = batch['tag']
                    df['label'] = train_data.select(batch['tag'])['label']
                    representations_df = pd.concat([representations_df, df])
            progress_bar.update(1)
        self.database = representations_df
        self.tree = KDTree(representations_df.iloc[:, :self.layer_dim].to_numpy(), metric="l2")

    def nearest_neighbors(self, inputs: Dict[str, Union[torch.Tensor, Any]], k=10):
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
        print(inputs)
        input()
        # progress_bar = tqdm(range(len(inputs)))
        # self.model.eval()
        # # 0-self.layer_dim = representation, layer #, label of this example, tag (index in train data)
        # # total dim when finished = (training size * # of layers) x (self.layer_dim + 3)
        # representations_df = pd.DataFrame(columns = [*range(self.layer_dim), "layer", "label", "tag"])
        # for batch in train_dataloader:
        #     inputs = {k: torch.stack(v, dim=1) for k, v in batch.items() if k in self.args}
        #     with torch.no_grad():
        #         outputs = self.model(**inputs, output_hidden_states=True)               # dict: {'loss', 'logits', 'hidden_states'}
        #         # Hidden-states of the model = the initial embedding outputs + the output of each layer
        #         hidden_states = outputs['hidden_states']                                # (num_layers+1, batch_size, max_seq_len, embedding_dim)
        #         # filter representations to what we need
        #         for layer in self.layers_to_save:
        #             # average across all tokens to obtain embedding
        #             rep = torch.mean(hidden_states[layer], dim=1).squeeze().detach()    # (batch_size, embedding_dim)
        #             df = pd.DataFrame(rep.numpy())
        #             df['layer'] = layer
        #             df['tag'] = batch['tag']
        #             df['label'] = train_data.select(batch['tag'])['label']
        #             representations_df = pd.concat([representations_df, df])
        #     progress_bar.update(1)
        # self.database = representations_df
        # self.tree = KDTree(representations_df.iloc[:, :self.layer_dim].to_numpy(), metric="l2")

        # distances and indices of nearest neighbors
        dist, idx = self.tree.query(inputs, k=k)

    def forward(self, inputs, kwargs):
        print(inputs)
        print(kwargs)
        input()
    
        
    def __call__(self, inputs, kwargs):
        self.forward(inputs, kwargs)
