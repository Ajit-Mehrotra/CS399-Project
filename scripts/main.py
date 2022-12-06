from analysis.trainAndEvaluate import trainAndEvaluate
from analysis.explore import explore
from analysis.finetuneAndEvaluate import run_model
from preprocessing.transformAndScale import codeify
from preprocessing.clean import remove_non_us, fill_na, column_droppage, remove_na, drop_dupes
from preprocessing.split import split_data, split_train_test
from preprocessing.data_management import read_data
from preprocessing.NLP import get_tokenized_data
from analysis.reevaluate import reevaluate
import os
import pandas as pd
import numpy as np
# Ignores copy warnings on working code, python is just being annoying
pd.options.mode.chained_assignment = None   


def process() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ''' Initialization for data processing, only run when needed '''

    np.random.seed(42)
    
    data = read_data(['data', 'glassdoor_reviews.csv'], delimiter='\t')
    # Drops all Non-US reviews
    data = remove_non_us(data)
    # Drops all duplicate reviews
    data = drop_dupes(data)
    # Drops all missing values from our response variable
    data = remove_na(data, ["work_life_balance"])

    # Don't run, takes hours!
    # The dataset provided already has it ran
    # data = get_tokenized_data(data)

    # Processes data into stuff the models can use
    data = codeify(data)
    # Fills missing values with the median of the column
    data = fill_na(data, ["culture_values",
                   "career_opp", "comp_benefits", "senior_mgmt"])
    # Drops unnecessary columns
    data = column_droppage(data, ["firm", "date_review", "job_title",
                           "diversity_inclusion", "location", "overall_rating", "tokenized_pros", "tokenized_cons", "tokenized_headline"])

    X_train, X_test, y_train, y_test = split_data(data, 0.2, ['data'])

    return X_train, X_test, y_train, y_test


def establish_database() -> tuple[pd.DataFrame,
                                  pd.DataFrame, pd.Series, pd.Series]:
    ''' Verify if the database exists, if not, create it '''

    # Downloads and processes the database if it doesn't exist
    if not os.path.exists(os.path.join('data', 'glassdoor_reviews.csv')):
        from fetch import fetch
        fetch(['data', 'glassdoor_reviews.csv'])
        X_train, X_test, y_train, y_test = process()
        return X_train, X_test, y_train, y_test

    # if it hasn't been processed, process it
    elif not os.path.exists(os.path.join('data', 'train.csv')) or not os.path.exists(os.path.join('data', 'test.csv')):
        X_train, X_test, y_train, y_test = process()
        return X_train, X_test, y_train, y_test

    # if it has been processed, read it
    else:
        print("Database already exists")
        X_train, X_test, y_train, y_test = split_train_test(
            read_data(['data', 'train.csv']), read_data(['data', 'test.csv']))
        return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = establish_database()

    # After elaborate feature testing, models performed better without this for some odd reason
    X_train.drop(columns=['current'], inplace=True)
    X_test.drop(columns=['current'], inplace=True)

    # explore(X_train, X_test, y_train, y_test)
    # trainAndEvaluate(X_train, X_test, y_train, y_test)

    run_model(X_train, X_test, y_train, y_test)

    reevaluate(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
