from scripts.util import printWithPadding
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def explore(X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, y_test: pd.Series) -> None:
    ''' Explores the data for phase 2 '''
    print(y_test.describe())
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    for col in X.columns:
        printWithPadding(col, qty=80)
        printWithPadding("Column info", qty=16)
        print("Type: ", X[col].dtype)
        print("Unique values: ", X[col].nunique())
        null = X[col].isnull().sum()
        print(
            "Null values: ",
            null,
            "(",
            round(
                null /
                X.shape[0] *
                100,
                2),
            "%)")

        # Measures corr and covar for the numeric columns that we care about
        if is_numeric_dtype(X[col]) and col != "current":
            printWithPadding("Column data", qty=16)
            print("Correlation: ", round(X[col].corr(y), 2))
            print("Covariance: ", round(X[col].cov(y), 2))

    visualize_employmentstat(X, y)
    visualize_numeric(X, y)


def visualize_employmentstat(X: pd.DataFrame, y: pd.Series) -> None:
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
    sns.set_theme(style="whitegrid", palette="pastel")
    plt.title("Distribution of Reviews by Employment Status")
    plt.xlabel("Employment Status")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    sns.barplot(
        x=X['current'].value_counts().index,
        y=X['current'].value_counts())
    plt.subplots_adjust(bottom=.25, left=.15)
    plt.show()
    plt.close(fig)


def visualize_numeric(X: pd.DataFrame, y: pd.Series) -> None:

    for col in X.columns:
        if is_numeric_dtype(X[col]) and col != "current":
            print("Visualizing ", col)
            sns.set_theme(style="whitegrid", palette="pastel")
            fig = plt.figure(figsize=(10, 10))
            plt.title("Histogram distribution of " + col)
            plt.xlabel(col)
            plt.ylabel("Count")
            sns.histplot(X[col], bins=5, discrete=True, shrink=0.8, kde=True)
            sns.despine(left=True)
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            fig2 = plt.figure(figsize=(10, 10))
            plt.title("Boxplot distribution of " + col)
            plt.xlabel(col)
            plt.ylabel("Rating")
            sns.boxplot(x=X[col], y=y)
            plt.show()
            plt.close(fig2)

