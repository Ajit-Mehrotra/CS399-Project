import os 
import numpy as np


from preprocessing.data_management import read_data
from preprocessing.split import split_data
from preprocessing.split import split_train_test
from preprocessing.refactor import remove_non_us
from preprocessing.refactor import codeify

from analysis.explore import explore

def process() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' Initialization for data processing, only run when needed '''

    data = read_data(['data', 'glassdoor_reviews.csv'])
    data = remove_non_us(data)
    data = codeify(data)
    
    X_train, X_test, y_train, y_test = split_data(data, 0.2, ['data'])
    return X_train, X_test, y_train, y_test

def establish_database() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' Verify if the database exists, if not, create it '''
    
    if not os.path.exists(os.path.join('data', 'glassdoor_reviews.csv')):
        from fetch import fetch
        fetch(['data', 'glassdoor_reviews.csv'])
        X_train, X_test, y_train, y_test = process()

    elif not os.path.exists(os.path.join('data', 'train.csv')) or not os.path.exists(os.path.join('data', 'test.csv')):
        X_train, X_test, y_train, y_test = process()

    else:
        print("Database already exists")
        X_train, X_test, y_train, y_test = split_train_test(read_data(['data', 'train.csv']), read_data(['data', 'test.csv']))
        return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = establish_database()
    
    explore(X_train, X_test, y_train, y_test)

    


if __name__ == '__main__':
    main()