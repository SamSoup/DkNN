from typing import Union, Mapping, Optional, Dict, Any, List
import torch.nn as nn
import torch
import inspect
import datasets
from packaging import version
from transformers.utils import find_labels
from transformers import (
    PreTrainedModel,
    TrainingArguments
)

def find_signature_columns(model: Union[PreTrainedModel, nn.Module], args: TrainingArguments):
    default_label_names = find_labels(model.__class__)
    label_names = default_label_names if args.label_names is None else args.label_names
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += list(set(["label", "label_ids"] + label_names))
    return signature_columns

def prepare_input(data: Union[torch.Tensor, Any], device: torch.device) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    Taken directly from Transformers.Trainer, with slight modifications without deepseed support
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data

def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], 
                   signature_columns: List[str],
                   device: torch.device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state. Taken directly from Transformers.Trainer, with slight modifications without deepseed support
    """
    inputs = prepare_input(inputs, device)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
            f"training dataset contains keys expected by the model: {','.join(signature_columns)}."
        )
    return inputs

def remove_unused_columns(dataset: datasets.Dataset, 
                           model: Union[PreTrainedModel, nn.Module],
                           remove_unused_columns: bool,
                           signature_columns: List[str], 
                           description: Optional[str] = None) -> datasets.Dataset:
    """
    Trainer's internal function that drops extraenous keys from the dataset, taken directly from
    Transformers.Trainer with slight modifications

    Args:
        dataset (Dataset): the input dataset to strip
        remove_unused_columns (bool): should we remove unused columns or not
        signature_columns (List[str]): the list of columns to keep
        description (Optional[str], optional): description of the dataset. Defaults to None.

    Returns:
        Dataset: the cleaned dataset without unexpected arguments for the model to call forward on
    """

    if not remove_unused_columns:
        return dataset
    if signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]

    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    if len(ignored_columns) > 0:
        dset_description = "" if description is None else f"in the {description} set "
        print(
            f"The following columns {dset_description} don't have a corresponding argument in "
            f"`{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            f" If {', '.join(ignored_columns)} are not expected by `{model.__class__.__name__}.forward`, "
            f" you can safely ignore this message."
        )

    columns = [k for k in signature_columns if k in dataset.column_names]

    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        dataset.set_format(
            type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        )
        return dataset
    else:
        return dataset.remove_columns(ignored_columns)
