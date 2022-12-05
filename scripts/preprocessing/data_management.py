import os
import pandas as pd
import pickle


def read_data(file_path: list[str], delimiter=",") -> pd.DataFrame:
    """Read the data from the named csv file."""

    print("Reading data from: ", os.path.join(*file_path))
    return pd.read_csv(os.path.join(*file_path), delimiter=delimiter, index_col=0)


def write_model(model, file_path: list[str]) -> None:
    """Write the model to the named file."""

    print("Writing model to: ", os.path.join(*file_path))
    if not os.path.exists(os.path.join(*file_path[:-1])):
        os.makedirs(os.path.join(*file_path[:-1]))
    with open(os.path.join(*file_path) + ".pkl", 'wb') as f:
        pickle.dump(model, f)

# cant type hint the returned model because python doesnt support generic
# types :D


def read_model(file_path: list[str]):
    """Read the model from the named file."""

    print("Reading model from: ", os.path.join(*file_path))
    with open(os.path.join(*file_path) + ".pkl", 'rb') as f:
        return pickle.load(f)
