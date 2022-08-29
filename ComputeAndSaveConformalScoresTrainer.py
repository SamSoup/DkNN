from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments
)
from transformers.data.data_collator import default_data_collator, DataCollatorWithPadding
from datasets import Dataset
from typing import Optional, Union, Callable
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import ( 
    compute_nonconformity_score, 
    find_signature_columns, 
    prepare_inputs, 
    remove_unused_columns, 
    get_hidden_states,
    hidden_states_to_cpu
)
from NearestNeighborFinders import AbstractNearestNeighbor
import torch
import torch.nn as nn
import numpy as np

class ComputeAndSaveConformalScoresTrainer:
    """
    Trainer Dedicated to Compute and Save Conform Scores from the caliberation set.
    However, this does NOT inhert from Transformers.Trainer because it does not 
    need its entire functionality. Rather, this only takes in arguments that are
    sufficent to run a single iteration through the caliberation dataset, compute, 
    and save the conform score per each example. Nevertheless, it bears quite a bit 
    similarity w.r.t Transformers.Trainer.
    """
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        caliberation_dataset: Optional[Dataset] = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        nearestNeighborFunction: AbstractNearestNeighbor = None,
        dist_to_weight_fct: Callable[[np.ndarray], np.ndarray] = None,
        read_from_scores_path: bool = False,
        save_nonconform_scores_path: Optional[str] = None
    ):
        torch.cuda.empty_cache() # save memory before iterating through dataset
        print(torch.cuda.memory_summary())
        self.model = model
        self.args = args
        self.caliberation_dataset = caliberation_dataset
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.nearestNeighborFunction = nearestNeighborFunction
        self.dist_to_weight_fct = dist_to_weight_fct
        self.read_from_scores_path = read_from_scores_path
        self.save_nonconform_scores_path = save_nonconform_scores_path
        self._signature_columns = find_signature_columns(model, args)

    def compute_and_save_nonconformity_scores(self) -> np.ndarray:
        """
        We compute and store all non-conformity scores for the caliberation set
        as an numpy array of dimensions (# of example in caliberation, # of possible labels)
        
        Args:
            eval_data (Dataset): caliberation dataset
            save_nonconform_scores_path (Optional[str], optional): path to save nonconformity scores for evaluation data. Defaults to None.
            label_to_id (Dict[Any, int]): dictionary mapping from original format of label to some integer id
        """
        if self.read_from_scores_path:
            print("***** Running DkNN - Loading scores from specified path *****")
            return np.loadtxt(self.save_nonconform_scores_path, delimiter=",")
    
        nonconformity_scores = np.zeros(len(self.caliberation_dataset))
        print("***** Running DkNN - Computing Nonconformity Scores for Caliberation Data *****")
        eval_dataloader = DataLoader(
            remove_unused_columns(self.caliberation_dataset, self.model, True, self._signature_columns),
            shuffle=True, batch_size=self.args.eval_batch_size, collate_fn=self.data_collator)
        progress_bar = tqdm(range(len(eval_dataloader)))
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                inputs = prepare_inputs(batch, self._signature_columns, self.args.device)
                labels = inputs.pop("labels").cpu().numpy()
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = get_hidden_states(self.model.config.is_encoder_decoder, outputs)
                hidden_states = hidden_states_to_cpu(hidden_states)
                distances, neighbors_labels, _ = self.nearestNeighborFunction.nearest_neighbors(hidden_states)
                weights = self.dist_to_weight_fct(distances)
                for j, label in enumerate(labels):
                    nonconform_score = compute_nonconformity_score(neighbors_labels[j, :].reshape(1, -1), label, weights[j, :])
                    nonconformity_scores[i*self.args.eval_batch_size + j] = nonconform_score
                progress_bar.update(1)
        if self.save_nonconform_scores_path is not None:
            print("***** Running DkNN - Saving Layer Representations *****")
            np.savetxt(self.save_nonconform_scores_path, nonconformity_scores, delimiter=",")
        return nonconformity_scores
