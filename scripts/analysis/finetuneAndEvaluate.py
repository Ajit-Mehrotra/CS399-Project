from util import printWithPadding
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from preprocessing.data_management import write_model, read_model

import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def run_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
              y_train: pd.Series, y_test: pd.Series) -> None:
    # rfc(X_train, X_test, y_train, y_test)
    # Gaussian(X_train, X_test, y_train, y_test)
    # logit_reg(X_train, X_test, y_train, y_test)
    # knn(X_train, X_test, y_train, y_test)


def graph_importances(feature_names: list[str], importances: list[float]) -> None:
    """Graph the feature importances."""
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)),
             importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def graph_conf_matrix(y_test: pd.Series, y_pred: pd.Series) -> None:
    """Graphs the confusion matrix."""

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
    # Adds text labels to corresponding cells
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.show()


def rfc(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series):
    ''' Random Forest Classifier '''
    model = RandomForestClassifier(oob_score=True, random_state=42)
    param_grid = {'n_estimators': [100, 200, 300],
                  'max_depth': [3, 5, 7, 9]}

    grid = GridSearchCV(model, param_grid,
                        scoring='precision', cv=5, verbose=2)
    grid.fit(X_train, y_train)

    # Gauges the model's performance
    printWithPadding("Random Forest")
    results = grid.predict(X_test)
    print(classification_report(y_test, results))
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
    print("Feature importances: ", grid.best_estimator_.feature_importances_)
    graph_importances(
        X_train.columns, grid.best_estimator_.feature_importances_)
    graph_conf_matrix(y_test, results)

    # Saves the model
    write_model(model, ["models", "rfc"])


def Gaussian(X_train: pd.DataFrame, X_test: pd.DataFrame,
             y_train: pd.Series, y_test: pd.Series) -> None:
    model = GaussianNB()
    model.fit(X_train, y_train)

    # grid search
    param_grid = {'var_smoothing': np.logspace(0, -9, num=10)}
    grid = GridSearchCV(model, param_grid,
                        scoring='precision', cv=5, verbose=2)
    grid.fit(X_train, y_train)

    # Gauges the model's performance
    printWithPadding("Gaussian Naive Bayes")
    results = grid.predict(X_test)
    print(classification_report(y_test, results))
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
    graph_conf_matrix(y_test, results)

    # Saves the model
    write_model(model, ["models", "gaussian"])


def logit_reg(X_train: pd.DataFrame, X_test: pd.DataFrame,
              y_train: pd.Series, y_test: pd.Series) -> None:
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # grid search
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty': ['l1', 'l2']}
    grid = GridSearchCV(model, param_grid,
                        scoring='precision', cv=5, verbose=2)
    grid.fit(X_train, y_train)

    # Gauges the model's performance
    printWithPadding("Logistic Regression")
    results = grid.predict(X_test)
    print(classification_report(y_test, results))
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
    print("Feature importances: ", grid.best_estimator_.coef_)
    graph_importances(X_train.columns, grid.best_estimator_.coef_[0])
    graph_conf_matrix(y_test, results)

    # Saves the model
    write_model(model, ["models", "logit_reg"])


def knn(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # grid search
    param_grid = {'n_neighbors': [
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]}
    grid = GridSearchCV(model, param_grid,
                        scoring='precision', cv=5, verbose=2)
    grid.fit(X_train, y_train)

    # Gauges the model's performance
    printWithPadding("K Nearest Neighbors")
    results = grid.predict(X_test)
    print(classification_report(y_test, results))
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)
    graph_conf_matrix(y_test, results)

    # Saves the model
    write_model(model, ["models", "knn"])
