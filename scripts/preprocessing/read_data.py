import os
import pandas as pd

def read_data(file_path: list[str]) -> pd.DataFrame:
    """Read the data from the csv file."""

    print("Reading data from: ", os.path.join(*file_path))
    return pd.read_csv(os.path.join(*file_path))

def read_train(file_path: list[str]) -> pd.DataFrame:
    """Read the train data from the csv file."""

    print("Reading train data from: ", os.path.join(*file_path))
    return pd.read_csv(os.path.join(*file_path))

def read_test(file_path: list[str]) -> pd.DataFrame:
    """Read the test data from the csv file."""

    print("Reading test data from: ", os.path.join(*file_path))
    return pd.read_csv(os.path.join(*file_path))
