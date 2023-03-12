import requests
import os
import pytest
from starter.ml.model import load_model
import numpy as np
import pdb


try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

except ImportError:
    print("dotenv is not installed")


if 'ENVIRONMENT' in os.environ and os.environ['ENVIRONMENT'] == 'development':
    # Do something if the environment variable is set to 'some_value'
    print("Environment is set to 'development'")
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    # Get the parent directory of the current file
    parent_directory_path = os.path.dirname(current_file_path)
    # Get the parent directory of the current file
    project_directory_path = os.path.dirname(parent_directory_path)
    label_encoder_path = os.path.join(project_directory_path,'model/lb.joblib')

else:
    # Do something else if the environment variable is not set or has a different value
    print("Running on GitHub Actions")
    label_encoder_path = '/home/runner/work/udacity_ml_devops_fastapi/udacity_ml_devops_fastapi/starter/model/lb.joblib'


lb = load_model(label_encoder_path)

@pytest.fixture
def post_data():
    """ Simple function to generate some fake Pandas data."""
    post_data = {
        'age': 25,
        'workclass': 'Private',
        'fnlwgt': 226802,
        'education': '11th',
        'education_num': 7,
        'marital_status': 'Never-married',
        'occupation': 'Machine-op-inspct',
        'relationship': 'Own-child',
        'race': 'Black',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }

    return post_data

@pytest.fixture
def high_income_data():
    """ Simple function to generate some fake Pandas data."""
    post_data = {
        'age': 35,
        'workclass': 'Self-emp-not-inc',
        'fnlwgt': 150000,
        'education': 'Masters',
        'education_num': 14,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Husband',
        'race': 'Asian-Pac-Islander',
        'sex': 'Male',
        'capital_gain': 100000,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'India'
    }

    return post_data

def test_get_method():
    """ Tests if our data has all 32561 rows containing 15 features each"""
    response = requests.get('http://localhost:8000/')
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to my FastAPI"}

def test_post_method(post_data):
    """ Tests if our data has all 32561 rows containing 15 features each"""
    response = requests.post('http://localhost:8000/predict', json=post_data)
    # Get the prediction result from the response
    prediction = response.json()
    print(response.text)

    assert response.status_code == 200, "Response was not 200"
    assert prediction["predictions"] == '[" <=50K"]', "Prediction was not like expected"

def test_income_prediction_high(high_income_data):
    response = requests.post('http://localhost:8000/predict', json=high_income_data)
    prediction = response.json()

    assert response.status_code == 200, "Response was not 200"
    assert prediction["predictions"] == '[" >50K"]', "Prediction was not like expected"

