"""
Preprocess the data derived from Antigoni Maria Founta et al.

This script is ran only once and binarizes the dataset
"""
import pandas as pd
import sys

data_file = sys.argv[1]
output_file = sys.argv[2]

columns = ["Tweet", "Orig_Label", "Vote"]
data = pd.read_csv(data_file, sep=None, names=columns)
data["Label"] = data["Orig_Label"] == "hateful"
data.to_csv(output_file, header=True, index=False)

