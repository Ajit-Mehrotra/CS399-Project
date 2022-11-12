import pandas as pd

def remove_non_us(data: pd.DataFrame) -> pd.DataFrame:
    """Remove non US reviews."""

    States = [ ', AK', ', AL', ', AR', ', AZ', ', CA', ', CO', ', CT', ', DC', ', DE', ', FL', ', GA',
           ', HI', ', IA', ', ID', ', IL', ', IN', ', KS', ', KY', ', LA', ', MA', ', MD', ', ME',
           ', MI', ', MN', ', MO', ', MS', ', MT', ', NC', ', ND', ', NE', ', NH', ', NJ', ', NM',
           ', NV', ', NY', ', OH', ', OK', ', OR', ', PA', ', RI', ', SC', ', SD', ', TN', ', TX',
           ', UT', ', VA', ', VT', ', WA', ', WI', ', WV', ', WY']

    print("Removing non US reviews")
    data.dropna(subset=['location'], inplace=True)
    return data[data['location'].str.contains('|'.join(States))]