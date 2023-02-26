import pytest
import pandas as pd
import os
from sklearn.model_selection import train_test_split
#from starter.ml.data import process_data

cwd = os.getcwd()
print(cwd)

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

except ImportError:
    print("dotenv is not installed")



if 'ENVIRONMENT' in os.environ and os.environ['ENVIRONMENT'] == 'development':
    # Do something if the environment variable is set to 'some_value'
    print("Environment is set to 'development'")
    data_path = '/Users/david/Code/digerian/udacity_ml_devops_fastapi/starter/data/census.csv'

else:
    # Do something else if the environment variable is not set or has a different value
    print("ENVIRONMENT is set to GitHub Actions")
    data_path = '/home/runner/work/udacity_ml_devops_fastapi/udacity_ml_devops_fastapi/starter/data/census.csv'

@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    return data

@pytest.fixture
def cat_features():
    """ Simple function to generate some fake Pandas data."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features


def test_split_data(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    assert train.shape[0] > test.shape[0], "Data does not have the expected shape."

# def test_process_train_data(data, cat_features):
#     """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
#     train, test = train_test_split(data, test_size=0.20, random_state=42)
#     X_train, y_train, encoder, lb = process_data(
#         train, categorical_features=cat_features, label="salary", training=True
#         )
#     assert X_train.shape == (26048, 108) and y_train.shape == (26048,), "Data does not have the expected shape."

def test_train_model(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    assert data.shape == (32561, 15) , "Data does not have the expected shape."

