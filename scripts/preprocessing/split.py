import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(data: pd.DataFrame, ratio: float, folder_path: list[str], seed=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """split the dataset based on the split ratio. Used for initial setup."""

    rating = data['overall_rating']
    data.drop(['overall_rating'], axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(data, rating, test_size=ratio, random_state=seed)
    folder_path = os.path.join(*folder_path)

    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    train_set.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
    test_set.to_csv(os.path.join(folder_path, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test
    
def split_train_test(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-seperates the y variable from the train and test sets."""

    rating = train['overall_rating']
    train.drop(['overall_rating'], axis=1, inplace=True)
    x_train, y_train = train, rating

    rating = test['overall_rating']
    test.drop(['overall_rating'], axis=1, inplace=True)
    x_test, y_test = test, rating

    return x_train, x_test, y_train, y_test