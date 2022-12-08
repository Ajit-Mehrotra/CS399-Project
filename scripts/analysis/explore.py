from util import printWithPadding
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

    # visualize_employmentstat(X, y)
    visualize_numeric(X, y)

    visualize_work_life(y)


def visualize_employmentstat(X: pd.DataFrame, y: pd.Series) -> None:
    ''' See distribution of reviews by employment status at the time of review '''

    print("Visualizing employment status")
    sns.set_theme(style="whitegrid", palette="pastel")
    fig = plt.figure(figsize=(10, 10))
    plt.title("Histogram distribution of employment status")
    plt.xlabel("Employment status")
    plt.ylabel("Count")
    sns.histplot(X["current"], bins="5", discrete=True, shrink=0.8, kde=True)
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()
    plt.close(fig)



def visualize_numeric(X: pd.DataFrame, y: pd.Series) -> None:
    ''' See distribution of reviews by numeric columns '''
    for col in X.columns:
        if is_numeric_dtype(X[col]) and col != "current":
            print("Visualizing ", col)
            sns.set_theme(style="whitegrid", palette="pastel")
            fig = plt.figure(figsize=(10, 10))
            plt.title("Histogram distribution of " + col)
            plt.xlabel(col)
            plt.ylabel("Count")
            sns.histplot(X[col], bins="5", discrete=True, shrink=0.8)
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

def visualize_work_life(y: pd.Series) -> None:
    copy_y = y['work_life_balance'] = y.apply(lambda row: "Work life Balance >= 3" if row == 1 else "Work life Balance < 3")
    print("Visualizing work-life-balance")
    sns.set_theme(style="whitegrid", palette="pastel")
    fig3 = plt.figure(figsize=(10, 10))
    plt.title("Histogram distribution of work-life-balance")
    plt.xlabel("Work-life-balance")
    plt.ylabel("Count")
    sns.histplot(copy_y, bins="5", discrete=True, shrink=0.8)
    plt.tight_layout()
    plt.show()
    plt.close(fig3)