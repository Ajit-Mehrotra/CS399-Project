import pandas as pd
from sklearn.impute import SimpleImputer

def remove_non_us(data: pd.DataFrame) -> pd.DataFrame:
    """Remove non US reviews."""

    print("Removing non US reviews")

    States = [', AK', ', AL', ', AR', ', AZ', ', CA', ', CO', ', CT', ', DC', ', DE', ', FL', ', GA',
              ', HI', ', IA', ', ID', ', IL', ', IN', ', KS', ', KY', ', LA', ', MA', ', MD', ', ME',
              ', MI', ', MN', ', MO', ', MS', ', MT', ', NC', ', ND', ', NE', ', NH', ', NJ', ', NM',
              ', NV', ', NY', ', OH', ', OK', ', OR', ', PA', ', RI', ', SC', ', SD', ', TN', ', TX',
              ', UT', ', VA', ', VT', ', WA', ', WI', ', WV', ', WY']

    data.dropna(subset=['location'], inplace=True)
    return data[data['location'].str.contains('|'.join(States))]


def codeify(data: pd.DataFrame) -> pd.DataFrame:
    """Convert specific columns in data to numeric."""

    print("Codeifying Employment Status")

    employment_status = {
        "Current Employee": 0,
        "Current Employee, less than 1 year": 1,
        "Current Employee, more than 1 year": 2,
        "Current Employee, more than 3 years": 3,
        "Current Employee, more than 5 years": 4,
        "Current Employee, more than 8 years": 5,
        "Current Employee, more than 10 years": 6,
        "Former Employee": 7,
        "Former Employee, less than 1 year": 8,
        "Former Employee, more than 1 year": 9,
        "Former Employee, more than 3 years": 10,
        "Former Employee, more than 5 years": 11,
        "Former Employee, more than 8 years": 12,
        "Former Employee, more than 10 years": 13,
    }

    data['current'] = data['current'].map(employment_status)

    return data

def fill_na(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in data."""

    print("Filling missing values")

    for col in ["work_life_balance", "culture_values", "diversity_inclusion", "career_opp", "comp_benefits", "senior_mgmt"]:
        mode_imputer = SimpleImputer(strategy='most_frequent')
        data[col] = mode_imputer.fit_transform(data[[col]])
    return data