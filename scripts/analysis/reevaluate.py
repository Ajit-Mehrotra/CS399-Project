from sklearn.metrics import classification_report
from preprocessing.data_management import read_model
from util import printWithPadding
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def reevaluate(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    printWithPadding("Re-evaluating models", qty=50)

    # re_rfc(X_train, X_test, y_train, y_test)
    re_gnb(X_train, X_test, y_train, y_test)
    # re_logit_reg(X_train, X_test, y_train, y_test)
    # re_knn(X_train, X_test, y_train, y_test)
    # re_dt(X_train, X_test, y_train, y_test)

def re_gnb(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    printWithPadding("Gaussian Naive Bayes")

    model = read_model(["models", "gaussian"])
    print(classification_report(y_test, model.predict(X_test)))

def re_knn(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    printWithPadding("KNN")

    model = read_model(["models", "knn"])
    print(classification_report(y_test, model.predict(X_test)))

def re_logit_reg(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    printWithPadding("Logistic Regression")

    model = read_model(["models", "logit_reg"])
    print(classification_report(y_test, model.predict(X_test)))

def re_rfc(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    printWithPadding("Random Forest Classifier")

    model = read_model(["models", "rfc"])
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

def re_dt(X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series) -> None:
    printWithPadding("Decision Tree Classifier")

    model = read_model(["models", "dt"])
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))