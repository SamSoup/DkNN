"""
Data related functions
"""

from sklearn.model_selection import train_test_split
import pandas as pd

def train_val_test_split(data_file: str, train_pct: float, eval_pct: float, 
    test_pct : float):
    """
    Here we perform train-validation-test split by calling the `train_test_split` 
    function from scikit-learn twice

    Input:
        data_file:
        train_pct:
        eval_pct:
        test_pct:
    
    Output:

    """
    if data_file.endswith(".json"):
        data = pd.read_json(data_file)
    else:
        data = pd.read_csv(data_file)
    
    