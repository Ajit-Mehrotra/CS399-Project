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
    data = data[data['location'].str.contains('|'.join(States))]
    return data


def remove_na(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove rows with missing values in data."""

    print("Removing rows with missing values:", *cols)
    data.dropna(subset=cols, inplace=True)
    return data


def column_droppage(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Drop columns from data."""

    print("Dropping columns:", *cols)
    data.drop(cols, axis=1, inplace=True)
    return data


def fill_na(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values in data."""

    print("Filling missing values")

    for col in cols:
        mode_imputer = SimpleImputer(strategy='median')
        data[col] = mode_imputer.fit_transform(data[[col]])
    return data

def drop_dupes(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""

    print("Dropping duplicate rows")
    data.drop_duplicates(inplace=True)
    return data