from util import printWithPadding
import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import SGDClassifier # later
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def run_model(X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.Series, y_test: pd.Series) -> None:
    rfc(X_train, X_test, y_train, y_test)
    # ensemble_models(X_train, X_test, y_train, y_test)
    # Gaussian(X_train, X_test, y_train, y_test)

def rfc(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series):
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [100, 200, 300],
                    'max_depth': [5, 8, 10, 15],
                    'min_samples_split': [2, 5, 10]}
    
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, verbose=2)
    grid.fit(X_train, y_train)

    printWithPadding("Random Forest")
    results = grid.predict(X_test)
    print(classification_report(y_test, results))
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)

    return grid.best_estimator_

def Gaussian(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series) -> None:
    model = GaussianNB()
    model.fit(X_train, y_train)
    

    printWithPadding("Gaussian Naive Bayes")
    results = model.predict(X_test)
    print(classification_report(y_test, results))


def ensemble_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series):
    # clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier()

    eclf = VotingClassifier(estimators=[('rf', clf2), ('gnb', clf3), ('knn', clf4)], voting='soft')

    params = {'rf__n_estimators': [20, 200], 'rf__max_depth': [2, 6], 'knn__n_neighbors': [1, 3, 5]}
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5, verbose=2)
    grid.fit(X_train, y_train)

    printWithPadding("Ensemble")
    results = grid.predict(X_test)
    print(classification_report(y_test, results))
    print("Best parameters: ", grid.best_params_)
    print("Best cross-validation score: ", grid.best_score_)
    print("Best estimator: ", grid.best_estimator_)


