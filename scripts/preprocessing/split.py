import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(data: pd.DataFrame, ratio: float,
               folder_path: list[str], seed=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """split the dataset based on the split ratio. Used for initial setup."""

    rating = data['work_life_balance']
    data.drop(['work_life_balance'], axis=1, inplace=True)

    # x_train, x_test, y_train, y_test = train_test_split(data, rating, test_size=ratio, random_state=seed)
    sss = StratifiedShuffleSplit(
        n_splits=20,
        test_size=ratio,
        random_state=seed)
    for train_index, test_index in sss.split(data, rating):
        x_train, x_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = rating.iloc[train_index], rating.iloc[test_index]

    folder_path = os.path.join(*folder_path)

    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    train_set.to_csv(os.path.join(folder_path, 'train.csv'), index=False)
    test_set.to_csv(os.path.join(folder_path, 'test.csv'), index=False)

    return x_train, x_test, y_train, y_test


def split_train_test(train: pd.DataFrame,
                     test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Re-seperates the y variable from the train and test sets."""

    rating = train['work_life_balance']
    train.drop(['work_life_balance'], axis=1, inplace=True)
    x_train, y_train = train, rating

    rating = test['work_life_balance']
    test.drop(['work_life_balance'], axis=1, inplace=True)
    x_test, y_test = test, rating

    return x_train, x_test, y_train, y_test
