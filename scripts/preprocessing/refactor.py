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

def codeify(data: pd.DataFrame) -> pd.DataFrame:
    """Convert specific columns in data to numeric."""
    
    data = codeify_current(data)
    data = codeify_ovrs(data)
    data = tokenize(data)

    return data


def codeify_current(data: pd.DataFrame) -> pd.DataFrame:
    """Convert employment status to numeric."""

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

def tokenize(data: pd.DataFrame) -> pd.DataFrame:
    """Tokenize text in data."""

    print("Tokenizing text")
    data['headline'] = data['headline'].apply(lambda x: str(x).lower().split()) 
    
    ghead_list = ["Adaptable", "Courageous", "Giving", "Neat", "Self-confident", "Adventurous", "Creative",
               "Good", "Nice", "Self-disciplined", "Affable", "Decisive", "Gregarious", "Non-judgemental",
               "Sensible", "Affectionate", "Dependable", "Hardworking", "Observant", "Sensitive", "Agreeable",
               "Determined", "Helpful", "Optimistic", "Ambitious", "Diligent", "Organized", "Amiable", "Diplomatic",
               "Honest", "Passionate", "Sincere", "Amicable", "Smart", "Socialable", "Easy-going", "Impartial",
               "Pioneering", "Straight-Forward", "Sympathetic", "Bright", "Efficient", "Talkative", "Energetic",
               "Intelligent", "Thoughtful", "Calm", "Enthusiastic", "Intellectual", "Polite", "Tidy", "Extroverted",
               "Intuitive", "Charismatic", "Exuberant", "Inventive", "Powerful", "Trustworthy", "Charming",
               "Fair-minded", "Joyful", "Practical", "Chatty", "Faithful", "Kind", "Pro-active", "Understanding",
               "Cheerful", "Upbeat", "Clever", "Laid-back", "Quiet", "Versatile", "Communicative", "Likable",
               "Rational", "Warmhearted", "Compassionate", "Friendly", "Loving", "Reliable", "Conscientious",
               "Funny", "Loyal", "Reserved", "Wise", "Considerate", "Generous", "Lucky", "Resourceful", "amazing", "good"
               "great", "awesome", "dope", "greatest", "transparent", "Great", "Fun"]

    def list_of_strings(x: list[str]) -> int: 
        for word in x:  
            if word in ghead_list:  
                return 1  
            else: 
                return 0 
    data['positive_headline'] = data.apply(lambda row: list_of_strings(row['headline']), axis=1) 

    return data

def fill_na(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill missing values in data."""

    print("Filling missing values")

    for col in cols:
        mode_imputer = SimpleImputer(strategy='median')
        data[col] = mode_imputer.fit_transform(data[[col]])
    return data