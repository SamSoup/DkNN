from typing import List, Tuple
from transformers.utils import ModelOutput
import random
import torch
import torch.nn.functional as F
import math

def get_actual_layers_to_save(layer_config: str, num_layers: int):
    switcher = {
        "All": list(range(num_layers)),
        "Embedding Only": [0],
        "Embedding + Last": [0, num_layers-1],
        "Last Only": [num_layers-1], 
        "Random 25%": random.sample(list(range(num_layers)), math.ceil(0.25*num_layers)), 
        "Random 50%": random.sample(list(range(num_layers)), math.ceil(0.5*num_layers)), 
        "Random 75%": random.sample(list(range(num_layers)), math.ceil(0.75*num_layers)),
    }
    return switcher[layer_config]

def get_actual_poolers_to_save(layer_config: str, pooler_config: str, layers_to_save: List[int]):
    switcher = {
        "mean_with_attention": ["mean_with_attention"] * len(layers_to_save), # for all layers regardless
        # # use mean with attention for the embedding layer (first), and the rest cls
        "mean_with_attention_and_cls": (
            ["cls"] if layer_config == "Last Only"
            else ["mean_with_attention"] + ["cls"] * (len(layers_to_save) - 1)
        )
    }
    return switcher[pooler_config]

def compute_layer_representations(is_encoder_decoder: bool, hidden_states,
                                  attention_mask, layers_to_save, poolers_to_use,
                                  database):
    if attention_mask is None:
        attention_mask = torch.ones(hidden_states[0].shape).detach().cpu()
    for layer, pooler in zip(layers_to_save, poolers_to_use):
        if is_encoder_decoder and layer >= len(hidden_states) / 2:
            layer_rep_np = pooler(hidden_states[layer], torch.ones(hidden_states[layer].shape[:2]))
        else:
            layer_rep_np = pooler(hidden_states[layer], attention_mask)
        if layer in database:
            database[layer].append(layer_rep_np)
        else:
            database[layer] = [layer_rep_np]
    return database

def hidden_states_to_cpu(hidden_states: Tuple[torch.tensor]) -> List[torch.tensor]:
    """
    Detach GPU tensors from hidden states to CPU

    Args:
        hidden_states (Tuple[torch.tensor]): Tuple of length equal to the number
        of layers, each tensor has shape (batch_size, seq_length, hidden_dim)

    Returns:
        List[torch.tensor]: Same shape and length as input, but inner tensors detached
    """
    ret = []
    for state in hidden_states:
        ret.append(state.detach().cpu())
    return ret

def get_hidden_states(is_encoder_decoder: bool, outputs: ModelOutput) -> Tuple[torch.tensor]:
    """
    Check if model is encoder-decoder, so that we get hidden states of both the encoder and decoder, 
    otherwise, return just the hidden states

    Args:
        is_encoder_decoder (bool): is this model an encoder-decoder model?
        outputs (ModelOutput): the outputs from the model

    Returns:
        Tuple[torch.tensor]: (batch_size, seq_length, hidden_dim) of len = embedding layer + model hidden layers
    """
    if is_encoder_decoder:
        return hidden_states_to_cpu(outputs["encoder_hidden_states"] + outputs["decoder_hidden_states"])
    return hidden_states_to_cpu(outputs["hidden_states"])

def get_layer_representations(hidden_states: torch.tensor) -> torch.tensor:
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

def get_pooled_layer_representations(hidden_states: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
    """
    Obtain the layer representations by averaging across all tokens with respect to the attention mask 
    to obtain the embedding at sentence level (for now).
    Args:
        hidden_states (torch.tensor): the hidden states from the model for one layer, with shape
        (batch_size, max_seq_len, embedding_dim)
        attention_mask (torch.tensor): the binary attention mask of shape
        (batch_size, num_heads, sequence_length, sequence_length)

    Returns:
        torch.tensor: (batch_size, self.layer_dim) of layer representations in order
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    return F.normalize(torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9), dim=0).squeeze().detach().cpu()
