"""
Custom Trainer that wraps around a Transformer model from Huggingface, and
overrides test-time behavior as specifed in Papernot, Nicolas, and Patrick McDaniel
"""

import torch
import math
import inspect
import datasets
from packaging import version
from torch import nn
from torch.nn.functional import softmax, cosine_similarity
from transformers import Trainer
from GenericPolicy import PolicyInterface
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from math import floor
from trainer_utils import (
    has_length
)
from utils import (
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled
)

class DkNN_Trainer(Trainer):
    def __init__(
            self,
            model :  = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (None, None),
            preprocess_logits_for_metrics = None,
            budget = None,
            num_cosine_similarity_take = None,
            embedding_method = None,
            policy: PolicyInterface = None,
            policy_type = None,
            gamma: float = 0.99,                                                # discount factor
        ):
            Trainer.__init__(self, model, args, data_collator, train_dataset, 
                             eval_dataset, tokenizer, model_init, compute_metrics,
                             callbacks, optimizers, preprocess_logits_for_metrics)
            # add reward, already used training set here
            model.budget = budget if args.do_train else 0
            model.gamma = gamma # used for RAL only
            self.embedding_dim = model.config.hidden_size * 4 if embedding_method == "concat_last_four" else model.config.hidden_size
            self.embedding_method = embedding_method
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.num_cosine_similarity_take = num_cosine_similarity_take        # number of cosine similarity to take when compute w.r.t history
            self.history = None                                                 # (num_of_examples_selected, self.embedding_dim)
            model.policy = policy
            model.policy_type = policy_type
            model.trajs = []
            self.selected_samples = []
            if model.policy_type == "RAL":
                model.policy.model.eval()
                # only compute reward for RAL
                self.before_train_eval_acc = self.evaluate()['eval_accuracy']

    def get_sentence_pair_embeddings(self, hidden_states: Tuple[torch.tensor]) -> torch.tensor:
        """
        Obtain the batched sentence pair embeddings given the hidden states 
        of the model output
        Input:
            hidden states: a tuple of length == num of layers, where each element
                is the output of said layer in (batch_size, max_seq_len, embedding_dim)
        Several methods considered in the BERT paper:
        https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
            1. first layer only (worst)
            2. last hidden layer only (ok)
            3. sum all layers
            4. second-to-last layer only
            5. sum of last four layers
            6. concat last four layers
            7. average all layers
        It's shown that 2-6 have similar performance; 7 is not tested but is most intuitive
        Here we do 6), but any method is feasible and should be ablated ideally, if given time
        """
        if (self.embedding_method == "first_only"):
            sentence_embedding = torch.mean(hidden_states[0], dim=1).squeeze()  # (batch_size, embedding_dim)
        elif (self.embedding_method == "last_only"):
            sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()  # (batch_size, embedding_dim)
        elif (self.embedding_method == "sum_all"):
            last = [hidden_states[i] for i in (model.config.num_hidden_layers)]
            sum_hidden_states = torch.sum(tuple(last_four_layers), dim=-1)      # (batch_size, seq_len, embedding_dim)
            sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze() # (batch_size, embedding_dim)
        elif (self.embedding_method == "second_last_only"):
            sentence_embedding = torch.mean(hidden_states[-2], dim=1).squeeze()  # (batch_size, embedding_dim)
        elif (self.embedding_method == "sum_last_four"):
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            # attention mask already used during layer propagation
            sum_hidden_states = torch.sum(tuple(last_four_layers), dim=-1)      # (batch_size, seq_len, embedding_dim)
            sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze() # (batch_size, embedding_dim)
        elif (self.embedding_method == "concat_last_four"):
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            # attention mask already used during layer propagation
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)      # (batch_size, seq_len, embedding_dim * 4)
            sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze() # (batch_size, embedding_dim * 4)
        elif (self.embedding_method == "average_all"):
            h = torch.mean(torch.stack(hidden_states))                          # (batch_size, seq_len, embedding_dim )
            sentence_embedding = torch.mean(h, dim=1).squeeze()                 # (batch_size, embedding_dim)
        else:
            raise ValueError("Embedding method must one of first_only, last_only,\
                                sum_all, second_last_only, sum_last_four,\
                                concat_last_four, average_all")
        return sentence_embedding

    def encode_history(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Need to encode the history of examples trained, since the number of 
        selected examples grows w.r.t time. Possible solutions:
            1. Compute pairwise cosine similarity w.r.t each example, take top k
            2. Cluster original embedding space into k clusters, take the cosine
                similarity w.r.t cluster centroid as well as cluster variance
            3. Perform PCA on history; compute cosine similarity w.r.t the top
                k principal components vector
        Input:
            embeddings: (batch_size, self.embedding_dim)
        Output:
            a matrix of top 
        """
        ### we implement 1 for now
        # edge case: when no history: return 0
        if self.history is None:
            return torch.zeros((embeddings.shape[0], self.num_cosine_similarity_take)).to(self.device)
        # for each example, compute its cosine similarity w.r.t all history
        embeddings = embeddings.mT
        embeddings_norm = torch.linalg.norm(embeddings, dim=0, keepdim=True)    # (1, batch_size)
        history_norm = torch.linalg.norm(self.history, dim=1, keepdim=True)     # (n=num_examples_in_history, 1)

        # Distance matrix of size (batch_size, n).
        cos_similarity = ((self.history @ embeddings) / (history_norm @ embeddings_norm)).mT # (batch_size, num_examples_in_history)
        # assert torch.isclose(
        #     cosine_similarity(embeddings[:, 0].T, self.history[0], dim=0), 
        #     cos_similarity[0, 0]
        # )
        assert(cos_similarity.shape[0] == embeddings.shape[1])
        # take the top num_history_considered
        sorted_sim, indices = cos_similarity.sort(dim=1, descending=True)
        return sorted_sim[:, :self.num_cosine_similarity_take]

    def get_policy_inputs(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.tensor:
        """
        Obtain all components necessary for the states (policy input)
            1. Contextualized embeddings from the batch of inputs
            2. Model marginal probabilities per class label
            3. Each batch's top k cosine similarity to the sample
            4. budget (number of examples left that can acquire labels on)
            5. Potential diversity metric
            6. Potential difficulty metric
        Input:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                Keys: "labels", "input_ids", "attention_mask"
                input_ids and attention_mask have shape (batch_size, max_seq_len)
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        """
        # We can evoke self.tokenizer.decode or self.tokenizer.batch_decode
        # to convert ids back to string tokens
        # but what we really want are the embeddings of the each token

        if model.policy_type == "Random":
            return torch.zeros((inputs["labels"].shape[0], ))               # need nothing if policy is random
        out = model(**inputs, output_hidden_states=True)                    # dict: {'loss', 'logits', 'hidden_states'}
        logits = out['logits']                                              # (batch_size, num_of_labels_possible)
        # we can obtain the model confidence for each of the classes 
        # by softmax over the logits per example
        # we could use the logits directly, but RL functions better when normalizd
        probs = softmax(logits, dim=1)                                      # (batch_size, num_of_labels_possible)
        if model.policy_type == "AL_baseline":
            return probs                                                    # only need marginals for AL_baseline
        hidden_states = out['hidden_states']                                # (num_layers, batch_size, max_seq_len, embedding_dim=1024)
        embeddings = self.get_sentence_pair_embeddings(hidden_states)       # (batch_size, final_embedding_dim)
        similarities = self.encode_history(embeddings)                      # (batch_size, num_cosine_similarity_take)
        # incoporate difficulty?

        policy_input = torch.cat(
            (probs, embeddings, similarities, 
                torch.tensor([model.budget]).to(self.device).expand(similarities.shape[0], 1)),
            dim=1
        )  # (batch_size, num_of_labels_possible + final_embedding_dim + num_cosine_similarity_take + 1)
        return policy_input

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.
            Subclass and override to inject custom behavior.
            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    Keys: "labels", "input_ids", "attention_mask", "tag"
                    input_ids and attention_mask have shape (batch_size, max_seq_len)
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                    "tag" is something extra I have added for the sake of identification,
                        when calling the model, I temporary remove this field
            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            inputs = self._prepare_inputs(inputs)
            tags = inputs.pop('tag', None)
            model.eval()
            if model.policy_type == "RAL":
                model.policy.model.eval()
                # only compute reward for RAL
                before_train_eval_acc = self.evaluate()['eval_accuracy']
            # get input to RL policy training and train
            policy_input = self.get_policy_inputs(model, inputs)
    
            # call policy 
            actions = model.policy(policy_input)
            print(actions)
            # truncate actions if the number of samples to select in actions
            # is more than the current budget allows for
            if actions.sum() > model.budget:
                # keep the first - budget number of examples to label
                for i in range(actions.shape[0]-1, -1, -1):
                    actions[i] = False
                    if actions.sum() <= model.budget:
                        break
            # record data selected for labeling
            self.selected_samples += tags[actions].tolist()
            # "retrieve" labels for selected sample from oracle
            model.budget -= actions.sum().item()
            inputs = {
                "labels": inputs['labels'][actions],
                "input_ids": inputs['input_ids'][actions],
                "attention_mask": inputs['attention_mask'][actions]
            }
            assert inputs['attention_mask'].shape == inputs['input_ids'].shape
            # add obtained embeddings with embeddings_dim to history (only needed for RAL)
            if model.policy_type == "RAL":
                embeddings = policy_input[:, :self.embedding_dim][actions]
                if embeddings.shape[0] != 0:
                    if self.history is None:
                        self.history = embeddings.detach()
                    else:
                        self.history = torch.cat((self.history.detach(), embeddings.detach()), dim=0)

            model.train()
            with self.autocast_smart_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

            # compute and store reward
            if model.policy_type == "RAL":
                model.eval()
                model.policy.model.eval()
                after_train_eval_acc = self.evaluate()['eval_accuracy']
                reward = after_train_eval_acc - self.before_train_eval_acc
                print("Got reward", reward)
                self.before_train_eval_acc = after_train_eval_acc
                model.trajs.append([policy_input.detach(), actions.detach(), reward])
            return loss.detach()

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
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

    def get_selected_samples(self):
        return self.selected_samples
