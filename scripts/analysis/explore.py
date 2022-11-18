import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from util import printWithPadding

def explore(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    ''' Explores the data for phase 2 '''

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test]) 

    for col in X.columns:
        printWithPadding(col, qty=80)
        printWithPadding("Column info", qty=16)
        print("Type: ", X[col].dtype)
        print("Unique values: ", X[col].nunique())
        null = X[col].isnull().sum()
        print("Null values: ", null, "(", round(null / X.shape[0] * 100, 2), "%)")

        # Measures corr and covar for the numeric columns that we care about
        if is_numeric_dtype(X[col]) and col != "current":
            printWithPadding("Column data", qty=16)
            print("Correlation: ", round(X[col].corr(y), 2))
            print("Covariance: ", round(X[col].cov(y), 2))
        
    review_source(X, y)

def review_source(X: pd.DataFrame, y: pd.DataFrame) -> None:
    ''' See distribution of reviews by employment status at the time of review '''

    employment_status = {
        "Current Employee (Duration Unspecified)": 0,
        "Current Employee, less than 1 year": 1,
        "Current Employee, more than 1 year": 2,
        "Current Employee, more than 3 years": 3,
        "Current Employee, more than 5 years": 4,
        "Current Employee, more than 8 years": 5,
        "Current Employee, more than 10 years": 6,
        "Former Employee (Duration Unspecified)": 7,
        "Former Employee, less than 1 year": 8,
        "Former Employee, more than 1 year": 9,
        "Former Employee, more than 3 years": 10,
        "Former Employee, more than 5 years": 11,
        "Former Employee, more than 8 years": 12,
        "Former Employee, more than 10 years": 13,
    }

    # See distribution of data in current column
    fig = plt.figure(figsize=(10, 10))
    plt.title("Distribution of Reviews by Employment Status")
    plt.xlabel("Employment Status")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.bar(employment_status.keys(), X['current'].value_counts())
    plt.subplots_adjust(bottom=.25, left=.15)
    plt.show()
    plt.close(fig)