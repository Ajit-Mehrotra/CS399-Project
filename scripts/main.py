import os 
import numpy as np
import pandas as pd
# Ignores copy warnings on working code, python is just being annoying
pd.options.mode.chained_assignment = None # default='warn'

from preprocessing.data_management import read_data
from preprocessing.split import split_data, split_train_test
from preprocessing.clean import remove_non_us, fill_na, column_droppage, remove_na
from preprocessing.transformAndScale import codeify

from analysis.explore import explore
from analysis.trainAndEvaluate import trainAndEvaluate

def process() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' Initialization for data processing, only run when needed '''

    data = read_data(['data', 'glassdoor_reviews.csv'])
    data = remove_non_us(data)
    data = remove_na(data, ["headline"])
    data = codeify(data)
    data = fill_na(data, ["work_life_balance", "culture_values", "career_opp", "comp_benefits", "senior_mgmt"])
    data = column_droppage(data, ["firm", "date_review", "job_title","diversity_inclusion", "location", "pros", "cons", "headline"])
    
    X_train, X_test, y_train, y_test = split_data(data, 0.2, ['data'])
    
    return X_train, X_test, y_train, y_test

def establish_database() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' Verify if the database exists, if not, create it '''
    
    if not os.path.exists(os.path.join('data', 'glassdoor_reviews.csv')):
        from fetch import fetch
        fetch(['data', 'glassdoor_reviews.csv'])
        X_train, X_test, y_train, y_test = process()
        return X_train, X_test, y_train, y_test

    elif not os.path.exists(os.path.join('data', 'train.csv')) or not os.path.exists(os.path.join('data', 'test.csv')):
        X_train, X_test, y_train, y_test = process()
        return X_train, X_test, y_train, y_test

    else:
        print("Database already exists")
        X_train, X_test, y_train, y_test = split_train_test(read_data(['data', 'train.csv']), read_data(['data', 'test.csv']))
        return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = establish_database()
    
    explore(X_train, X_test, y_train, y_test)
    trainAndEvaluate(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()