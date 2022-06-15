"""
Utility functions for data processing
"""

from cgi import test
from typing import Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd

def read_data(file: str) -> pd.DataFrame:
    """
    Read the data specified by `file` as a pandas Dataframe

    Args:
        file (str): a path to the dataset, ending in ".csv" or ".json"

    Returns:
        pd.DataFrame: the data in tabular format
    """

    if file.endswith(".json"):
        data = pd.read_json(file)
    else:
        # let pandas auto detect the input separator
        data = pd.read_csv(file, sep=None, engine="python")
    
    return data

def train_val_test_split(data_file: str, train_pct: float, eval_pct: float, 
    test_pct : float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Here we perform train-validation-test split by calling the `train_test_split` 
    function from scikit-learn twice. Data is stratified to maintain class 
    distribution. 

    Args:
        data_file (str): a path to the entire dataset, note that the labels (y) column 
                         must be named `Label`
        train_pct (float): % of data to keep for the training set
        eval_pct (float): % of data to keep for the validation set
        test_pct (float): % of data to keep for the test set
        seed (int): the seed for random shuffling of the dataFrame, for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: the train, eval, and test split
    """

    data = read_data(data_file)
    eval_and_test_pct = eval_pct + test_pct
    data_train, data_eval_and_test = train_test_split(data, 
        train_size=train_pct, random_state=seed, shuffle=True, 
        stratify=data["Label"])
    data_eval, data_test = train_test_split(data_eval_and_test, 
        test_size=test_pct / eval_and_test_pct, random_state=seed, shuffle=True, 
        stratify=data_eval_and_test["Label"])
    
    return data_train, data_eval, data_test
