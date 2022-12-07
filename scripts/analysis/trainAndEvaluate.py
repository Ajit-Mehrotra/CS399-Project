from util import printWithPadding
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_squared_error, classification_report

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def trainAndEvaluate(X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_train: pd.Series, y_test: pd.Series) -> None:
    ''' Train and evaluate models '''
    # test_lr(X_train, X_test, y_train, y_test)
    # test_knn(X_train, X_test, y_train, y_test)
    # test_dt(X_train, X_test, y_train, y_test)
    test_rf(X_train, X_test, y_train, y_test)
    # test_nb(X_train, X_test, y_train, y_test)


def test_lr(X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, y_test: pd.Series) -> None:
    ''' Testing Linear Regression '''
    model = LinearRegression()
    model.fit(X_train, y_train)

    printWithPadding("Linear Regression")
    results = model.predict(X_test)
    print("R2: ", r2_score(y_test, results))
    print("MSqE: ", mean_squared_error(y_test, results))


def test_knn(X_train: pd.DataFrame, X_test: pd.DataFrame,
             y_train: pd.Series, y_test: pd.Series) -> None:
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    printWithPadding("KNN")
    results = model.predict(X_test)
    print(classification_report(y_test, results))


def test_dt(X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, y_test: pd.Series) -> None:
    ''' Testing Decision Tree Classifier '''
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    printWithPadding("Decision Tree")
    results = model.predict(X_test)
    print(classification_report(y_test, results))


def test_rf(X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, y_test: pd.Series) -> None:
    ''' Testing Random Forest Classifier '''
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    printWithPadding("Random Forest")
    results = model.predict(X_test)
    print(classification_report(y_test, results))


def test_nb(X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, y_test: pd.Series) -> None:
    ''' Testing Naive Bayes Classifier '''
    model = GaussianNB()
    model.fit(X_train, y_train)

    printWithPadding("Guassian Naive Bayes")
    results = model.predict(X_test)
    print(classification_report(y_test, results))
