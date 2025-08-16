import json
from pathlib import Path
import pandas as pd


def load_json(file_path: Path) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_json(data: dict, file_path: Path) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_dataset(file_path: Path) -> dict:
    return load_json(file_path)

def parquet_to_df(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)

