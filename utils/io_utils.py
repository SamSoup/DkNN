from typing import Dict
import pandas as pd
import os, json

def mkdir_if_not_exists(dirpath: str):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def parse_json(path: str):
    with open(path, 'r') as fr:
        results = json.load(fr)
    return results

def add_to_dataframe(df, results: Dict[str, float], **kwargs):
    results.update(kwargs)
    # small helper to add to the current dataframe
    return pd.concat(
        [df, pd.DataFrame(results, index=[0])], ignore_index = True
    )