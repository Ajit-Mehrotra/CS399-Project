import pandas as pd
import numpy as np

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
        print("Type: ", X[col].dtype)
        print("Unique values: ", X[col].nunique())
        null = X[col].isnull().sum()
        print("Null values: ", null, "(", round(null / X.shape[0] * 100, 2), "%)")
    