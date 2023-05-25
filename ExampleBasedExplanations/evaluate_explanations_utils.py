"""
This script contains functions that compute metrics that we think are helpful
and say something about the quality of the retrieved neighbors (examples) w.r.t
some input example
"""
from transformers import (
    AutoConfig,
    set_seed
)
from sklearn.metrics import brier_score_loss
from datasets import load_metric
from typing import List
from ModelForSentenceLevelRepresentation import get_model_for_representation
from EmbeddingPooler import EmbeddingPooler
from utils import parse_json, flip_bits, get_train_representations_from_file, randargmax
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm
import pandas as pd
import os
import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt

def agg_distance_from_input_to_retrieved_neighbors(neighbor_distances: np.ndarray, 
                                                    agg_fct: Callable[np.ndarray]=np.mean):
    """
    Aggregate Distance from input example to retrieved neighbors,
    could either be mean or median
    (density: depends on data scale + embedding)

    Why: smaller distance indicates examples are similar to input and thus more likely 
    to be relevant / match (will they perceive it as perceptually similar or not) -- 
    should be intuitive to use why retrieved example is similar to the input and 
    thus relevant

    Assume neighbor_distances is already sorted ascendingly
    """
    return agg_fct(neighbor_distances, axis=1)

def neighbor_cohesion(neighbor_labels: np.ndarray):
    """
    Label cohesion/agreement in nearest K, or
    % of neighbors with the same label
    e.g. K=5, 3 NH -> cohesion=0.6 regardless if the 
    input is NH or H
    
    Why: if we show user K nearest neighbors, want fewer 
    incorrect ones to lead user astray. Most similar
    to confidence - highly cohesive neighbors for
    incorrectly predicted input sounds like unknowns unknowns
    """

    pct_positive = neighbor_labels.sum(axis=1) / neighbor_labels.shape[1]
    pct_negative = 1 - pct_positive
    return np.hstack((pct_positive.reshape(-1, 1), pct_negative.reshape(-1, 1))).max(axis=1)

def differentiate_neighbors(
    input_labels: np.ndarray,
    neighbor_labels: np.ndarray
):
    all_correct = np.all(neighbor_labels == input_labels[:, None], axis=1)
    no_correct = np.all(neighbor_labels != input_labels[:, None], axis=1)
    mixed = np.any(neighbor_labels != input_labels[:, None], axis=1) & np.any(neighbor_labels == input_labels[:, None], axis=1)
    assert(sum(all_correct) + sum(mixed) + sum(no_correct) == input_labels.size)

    return all_correct, no_correct, mixed

def find_closest_correct_neighbors(
    input_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_labels: np.ndarray
):
    """
    If we flip the label as input, then this function 
    finds the closest incorrect neighbor

    Returns idx, distance per example
    """
    # mask away incorrect examples! 
    mask = input_labels[:, None] != neighbor_labels
    distance_masked = np.ma.masked_array(neighbor_distances, mask)
    no_correct_neighbors = distance_masked.count(axis=1) == 0
    closest_correct_neighbors = distance_masked.argmin(axis=1, fill_value=MAX_DISTANCE)
    # in the rare case that no retrieved neighbor has correct label,
    # give a very large distance by default and no neighbor idx
    closest_correct_neighbors_indices = neighbor_indices[
        np.arange(neighbor_indices.shape[0]), closest_correct_neighbors
    ]
    closest_correct_neighbors_indices[no_correct_neighbors] = INVALID
    closest_correct_neighbors_distances = neighbor_distances[
        np.arange(neighbor_distances.shape[0]), closest_correct_neighbors
    ]
    closest_correct_neighbors_distances[no_correct_neighbors] = MAX_DISTANCE
    return closest_correct_neighbors_indices, closest_correct_neighbors_distances

def distance_from_input_to_nearest_incorrect_versus_distance_from_input_to_nearest_correct_neighbor(
    input_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_labels: np.ndarray
):
    """
    Relative (ratio of) distance from input example to nearest incorrect vs. 
    input to correct training example

    If no neighbor is correct -> Assign DISTANCE_MAX
    If no neighbor is incorrect -> Assign 0

    Only really makes sense when they are a mixture of both correct and incorrect
    examples present

    Why: if we show user K nearest neighbors, and incorrect examples are present, want them to be as 
    far (big) as possible so that user is more likely to select correct examples (closer, more similar) 
    as relevant and not incorrect examples (farther, less similar)
    """

    # find the closest correct (same label as input)
    _, closest_correct_neighbors_dists = \
    find_closest_correct_neighbors(
        input_labels, neighbor_indices, neighbor_distances, neighbor_labels
    )
    # find the closest incorrect (different label as input)
    inverse_labels = flip_bits(input_labels)
    _, closest_incorrect_neighbors_dists = \
    find_closest_correct_neighbors(
        inverse_labels, neighbor_indices, neighbor_distances, neighbor_labels
    )
    all_correct, no_correct, _ = differentiate_neighbors(input_labels, neighbor_labels)
    ratios = closest_incorrect_neighbors_dists / closest_correct_neighbors_dists
    ratios[all_correct] = 0
    ratios[no_correct] = MAX_DISTANCE

    return ratios

def agg_distance_from_input_to_incorrect_versus_agg_distance_from_input_to_nearest_correct_neighbor

def distance_from_nearest_incorrect_neighbor_to_nearest_correct_neighbor(
    input_labels: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    neighbor_labels: np.ndarray,
    train_representations: np.ndarray
):
    """
    Distance from nearest incorrect neighbor to nearest correct neighbor
    (want to be large)

    Why: if we show user K nearest neighbors, and incorrect examples are present, 
    want contrast between them to be distinguishable / large so that user 
    can more easily identify correct vs. incorrect ones -- 
    user mental model / understanding of decision space
    """
    # find the closest correct (same label as input)
    closest_correct_neighbors_indices, _ = find_closest_correct_neighbors(
        input_labels, neighbor_indices, neighbor_distances, neighbor_labels
    )
    # get the correct neighbor's representations
    correct_neighbors_reps = train_representations[
        closest_correct_neighbors_indices
    ]

    # find the closest incorrect (different label as input)
    inverse_labels = flip_bits(input_labels)
    closest_incorrect_neighbors_indices, _ = find_closest_correct_neighbors(
        inverse_labels, neighbor_indices, neighbor_distances, neighbor_labels
    )
    # get the incorrect neighbor's representations
    incorrect_neighbors_reps = train_representations[
        closest_incorrect_neighbors_indices
    ]

    # compute the L2 distance for given the example texts
    d = np.linalg.norm(incorrect_neighbors_reps-correct_neighbors_reps, axis=1)
    all_correct, no_correct, mixed = differentiate_neighbors(input_labels, neighbor_labels)
    d[all_correct] = MAX_DISTANCE
    d[no_correct] = 0
    return d


def visualize_metrics_boxplot(
    metrics: np.ndarray,
    is_pred_correct: np.ndarray,
    x_label: str, 
    y_label: str
):
    sns.boxplot(data=pd.DataFrame(data={
        "metrics": metrics,
        "correct": is_pred_correct,
    }), x="metrics", y="correct", orient='h')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()

def visualize_metrics_kdeplot(
    metrics: np.ndarray,
    is_pred_correct: np.ndarray,
    x_label: str, 
    y_label: str,
    hue_name: str,
    hue_labels: str,
):
    sns.kdeplot(data=pd.DataFrame(data={
        "metrics": metrics,
        "correct": is_pred_correct,
    }), x="metrics", hue="correct")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=hue_name, loc='upper left', labels=hue_labels)
    plt.tight_layout()
    plt.show()

def visualize_metrics_displot(
    metrics: np.ndarray,
    is_pred_correct: np.ndarray,
    x_label: str, 
    y_label: str,
    hue_name: str,
    hue_labels: str,
):
    metrics = np.round(metrics, 3)
    uniques, indices = np.unique(metrics, return_inverse=True)
    sns.displot(
        data=pd.DataFrame(data={
            "metrics": indices,
            "correct": is_pred_correct,
        }), x=indices, hue="correct", discrete=True,
        multiple="dodge", legend = False, shrink=0.8)
    plt.locator_params(axis='x', nbins=uniques.size+1)
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container)
    xlabels = [round(uniques[int(i)], 3) if i < uniques.size else i for i in ax.get_xticks() ]
    ax.set_xticklabels(xlabels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=hue_name, loc='upper left', labels=hue_labels)
    plt.tight_layout()

def sort_neighbors_according_to_distances(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
):
    sorted_indices = neighbor_distances.argsort(axis=1)
    neighbor_indices = np.take_along_axis(neighbor_indices, sorted_indices, axis=1)
    neighbor_distances = np.take_along_axis(neighbor_distances, sorted_indices, axis=1)
    return neighbor_indices, neighbor_distances

def get_neighbor_labels_given_indices(data: pd.DataFrame, neighbor_indices: np.ndarray):
    neighbor_labels = [
        data.iloc[indices, :]['label']
        for indices in neighbor_indices
    ]
    return np.stack(neighbor_labels, axis=0)

def compute_metrics_per_bin(df: pd.DataFrame, metric_descs: List[str]):
    """
    Return a dictionary of the form:
    {
        metric_name: { bin: metric_value }
    }
    """
    overall_res = {}
    for metric_desc in metric_descs:
        metric_func = load_metric(metric_desc)
        res = df.groupby("bin_idx").apply(
            lambda group: metric_func.compute(predictions=group["prediction"], references=group["label"])[metric_desc]
        ).to_dict()
        overall_res[metric_desc] = res
    return overall_res

def plot_caliberation_plot(ax, counts, bins, results, 
                           xlabel, ylabel, pad=20):
        ax.hist(bins[:-1], bins, weights=counts)
        ax.bar_label(ax.containers[0])
        # ax.stairs(counts, bins)
        ax.set_ylabel("# of Samples")
        ax.set_ylim(0, np.max(counts) * 1.1)
        ax.set_xlabel(xlabel)
        d = results[ylabel]
        lists = sorted(d.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists) # unpack a list of pairs into two tuples
        x_pos = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1) ] # center points of each bin
        x = [x_pos[i-1] for i in x ]
        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(x, y, color='red', linestyle="dashed", marker='o', fillstyle='full')
        # if we treat samples in each bin as a binomial distribution, then the 
        # mean is accuracy, while the std is defined as n*acc*(1-acc)
        y = np.array(y)
        counts = np.array([i for i in counts if i != 0])
        error = np.sqrt(y * (1-y) / counts)
        ax2.errorbar(x, y, yerr=error, fmt='ro--')
        # add point labels, if we want
#         for r,c in zip(x,y):
#             label = f"{round(c, 3)}"
#             plt.annotate(label, # this is the text
#                          (r,c), # these are the coordinates to position the label
#                          textcoords="offset points", # how to position the text
#                          xytext=(0,10), # distance from text to points (x,y)
#                          ha='center') # horizontal alignment can be left, right or center
        ax2.set_ylim(-0.1, 1.05)
        ax2.set_yticks([ i / 10 for i in range(0, 11, 2) ])
        ax2.tick_params(axis='y', colors='red')
        ax2.set_ylabel(ylabel.title(), color="red", fontsize=14, rotation=-90, labelpad=pad)

def visualize_metric_caliberation(metric, 
                                  input_labels, predicted_labels, xlabel, 
                                  evaluation_metrics = ['accuracy', 'f1'],
                                  bins_count=10):
    counts, bins = np.histogram(metric, bins=bins_count)
    bins[-1] += 0.01 # just to make the max a bit larger
    bin_indices = np.digitize(metric, bins=bins)
    # group predictions based on bins, then compute accuracy per bin for plotting purposes
    temp_df = pd.DataFrame({'label': input_labels, 'prediction': predicted_labels, 'bin_idx': bin_indices})
    results = compute_metrics_per_bin(temp_df, evaluation_metrics)
    fig, axes = plt.subplots(nrows=1, ncols = len(results), figsize=(9*len(results), 4))
    for ax, desc in zip(fig.axes, evaluation_metrics):
        plot_caliberation_plot(ax, counts, bins, results, 
                               xlabel=xlabel,
                               ylabel=desc)
