import numpy as np
import pandas as py
import os 

from preprocessing.read_data import read_data, read_train, read_test
from preprocessing.split import split_data
from preprocessing.refactor import remove_non_us

def process() -> None:
    ''' Initialization for data processing, only run when needed '''
    data = read_data(['data', 'glassdoor_reviews.csv'])
    data = remove_non_us(data)
    split_data(data, 0.2, ['data'])

def verify_database() -> None:
    ''' Verify if the database exists, if not, create it '''
    if not os.path.exists(os.path.join('data', 'glassdoor_reviews.csv')):
        from fetch import fetch
        fetch(['data', 'glassdoor_reviews.csv'])
        process()
    else:
        if not os.path.exists(os.path.join('data', 'train.csv')) or not os.path.exists(os.path.join('data', 'test.csv')):
            process()

def main():
    verify_database()

    train = read_train(['data', 'train.csv'])
    test = read_test(['data', 'test.csv'])
    print(train.shape, test.shape)



if __name__ == '__main__':
    main()