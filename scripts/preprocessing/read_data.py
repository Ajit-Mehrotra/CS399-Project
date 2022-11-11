import os
import pandas as pd

def read_data(file_path: list[str]) -> pd.DataFrame:
    """Read the data from the named csv file."""

    print("Reading data from: ", os.path.join(*file_path))
    return pd.read_csv(os.path.join(*file_path))
