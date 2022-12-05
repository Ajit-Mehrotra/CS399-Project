import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize

def codeify(data: pd.DataFrame) -> pd.DataFrame:
    """Convert specific columns in data to numeric."""

    data = codeify_current(data)
    data = codeify_numerics(data)
    data = codeify_ovrs(data)
    # data = encode_firms(data)
    data = codeify_get_bing_scores(data)
    data = binary_balanced_work(data)

    return data


def codeify_current(data: pd.DataFrame) -> pd.DataFrame:
    """Convert employment status to numeric."""

    print("Codeifying Employment Status")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    year_map = {
        0: 0,
        "Former Employee": 0,
        "Former Employee, less than 1 year": 0,
        "Former Employee, more than 1 year": 1,
        "Former Employee, more than 3 years": 3,
        "Former Employee, more than 5 years": 5,
        "Former Employee, more than 8 years": 8,
        "Former Employee, more than 10 years": 10,
        "Current Employee": 0,
        "Current Employee, less than 1 year": 0,
        "Current Employee, more than 1 year": 1,
        "Current Employee, more than 3 years": 3,
        "Current Employee, more than 5 years": 5,
        "Current Employee, more than 8 years": 8,
        "Current Employee, more than 10 years": 10,
    }

    # Creats 2 columns, one for current, one for former
    data["former"] = data.apply(lambda row: row['current'] if "Former" in row['current'] else 0, axis=1)
    data["current"] = data.apply(lambda row: row['current'] if "Current" in row['current'] else 0, axis=1)

    # Maps the columns to numeric years
    data["current"] = data["current"].map(year_map)
    data["former"] = data["former"].map(year_map)

    return data

def codeify_numerics(data: pd.DataFrame) -> pd.DataFrame:
    ''' Scales numeric columns. '''

    columns = ["work_life_balance", "career_opp", "comp_benefits", "senior_mgmt", "culture_values"]
    data[columns] = data[columns].apply(LabelEncoder().fit_transform)
    return data

def codeify_ovrs(data: pd.DataFrame) -> pd.DataFrame:
    ''' Convert OVR columns to numeric. '''

    print("Codeifying OVRs")

    rec_categories = {
        'x': 0,
        'o': 1,
        'r': 2,
        'v': 3
    }

    for cols in ["recommend", "ceo_approv", "outlook"]:
        data[cols] = data[cols].map(rec_categories)
    return data

def encode_firms(data: pd.DataFrame) -> pd.DataFrame:
    ''' Turn firms to ints '''

    enc = OrdinalEncoder()
    data[['firm']] = enc.fit_transform(data[['firm']])

    return data

def create_lexicon_dict():
    ''' Create a dictionary of words and their sentiment scores. '''

    pos_score = 1
    neg_score = -1
    word_dict = {}

    # Adding the positive words to the dictionary
    for word in opinion_lexicon.positive():
            word_dict[word] = pos_score

    # Adding the negative words to the dictionary
    for word in opinion_lexicon.negative():
            word_dict[word] = neg_score
    
    return word_dict

def bing_liu_score(text, word_dict):
    ''' Calculate the Bing Liu score for a given text. '''

    sentiment_score = 0
    score = 0
    bag_of_words = word_tokenize(str(text).lower())
    for word in bag_of_words:
        if word in word_dict:
            sentiment_score += word_dict[word]
        if len(bag_of_words) == 0:
            score = sentiment_score / 1
        else:
            score = sentiment_score / len(bag_of_words)
    return score

def codeify_get_bing_scores(data: pd.DataFrame) -> pd.DataFrame: 
    ''' Gets the scores of headline, pros, and cons according to the Bing Liu lexicon '''

    nltk.download('opinion_lexicon')
    nltk.download('punkt')
    columns = ['tokenized_headline', 'tokenized_pros', 'tokenized_cons']
    data.dropna(subset=columns, inplace = True)
    word_dict = create_lexicon_dict()
    data['bing_headline'] = data['tokenized_headline'].apply(lambda row: bing_liu_score(row, word_dict))
    data['bing_pros'] = data['tokenized_pros'].apply(lambda row: bing_liu_score(row, word_dict))
    data['bing_cons'] = data['tokenized_cons'].apply(lambda row: bing_liu_score(row, word_dict))
    return data

def binary_balanced_work(data: pd.DataFrame) -> pd.DataFrame:
    ''' Make work_life_balance binary. '''

    data['work_life_balance'] = data.apply(lambda row: 1 if row['work_life_balance'] >= 3 else 0, axis=1)
    return data
