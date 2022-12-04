from analysis.trainAndEvaluate import trainAndEvaluate
from analysis.explore import explore
from preprocessing.transformAndScale import codeify
from preprocessing.clean import remove_non_us, fill_na, column_droppage, remove_na, drop_dupes
from preprocessing.split import split_data, split_train_test
from preprocessing.data_management import read_data
from analysis.finetuneAndEvaluate import run_model
from preprocessing.NLP import get_tokenized_data
import os
import pandas as pd
# Ignores copy warnings on working code, python is just being annoying
pd.options.mode.chained_assignment = None   


def process() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ''' Initialization for data processing, only run when needed '''

    data = read_data(['data', 'glassdoor_reviews.csv'])
    data = remove_non_us(data)
    data = drop_dupes(data)
    data = remove_na(data, ["work_life_balance"])
    data = codeify(data)
    data = fill_na(data, ["culture_values",
                   "career_opp", "comp_benefits", "senior_mgmt"])
    data = column_droppage(data, ["firm", "date_review", "job_title",
                           "diversity_inclusion", "location", "pros", "cons", "headline", "overall_rating"])

    # only use once
    # data = get_tokenized_data(data)

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

    # explore(X_train, X_test, y_train, y_test)
    # trainAndEvaluate(X_train, X_test, y_train, y_test)

    run_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
