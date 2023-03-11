import pytest
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data
import pdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import warnings

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


def test_data_shape(data):
    """ Tests if our data has all 32561 rows containing 15 features each"""
    assert data.shape == (32561, 15) , "Data does not have the expected shape."


def test_process_data(data, cat_features):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    assert X_train.shape[0] == y_train.shape[0], "train data has not been processed correctly."
    assert isinstance(encoder, OneHotEncoder), "encoder is not a OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb is not a LabelBinarizer"

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, lb=lb, encoder=encoder
    )

    assert X_test.shape[0] == y_test.shape[0], "test data has not been processed correctly."
    assert X_train.shape[1] == X_test.shape[1], "train and test data dont have the same amount of features"
    assert X_train.shape[1] == 108, "the data has not been encoded correctly"


def test_model_inference(data, cat_features):
    """ Tests model inference"""
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, lb=lb, encoder=encoder
    )

    rfc = RandomForestClassifier(random_state=42)
    rfc = train_model(rfc, X_train, y_train)
    preds = inference(rfc, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert precision > 0, "Model was not able to learn and predict"
    assert recall > 0, "Model was not able to learn and predict"
    assert fbeta > 0, "Model was not able to learn and predict"

warnings.filterwarnings("ignore")
