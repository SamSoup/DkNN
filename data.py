"""
Data related functions
"""

from sklearn.model_selection import train_test_split

def train_val_test_split(data_file: str, train_pct: float, eval_pct: float, 
    test_pct : float):
    """
    Here we perform a 70-20-10 train-validation-test split
    by calling the `train_test_split` function from scikit-learn twice
    """
    