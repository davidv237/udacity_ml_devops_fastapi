import pytest
import pandas as pd
import os


try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    print("dotenv is not installed")


if 'ENVIRONMENT' in os.environ and os.environ['ENVIRONMENT'] == 'development':
    # Do something if the environment variable is set to 'some_value'
    print("Environment is set to 'development'")
    @pytest.fixture
    def data():
        data_path = '/Users/david/Code/digerian/udacity_ml_devops_fastapi/starter/data/census.csv'

        """ Simple function to generate some fake Pandas data."""
        data = pd.read_csv(data_path)
        return data
else:
    # Do something else if the environment variable is not set or has a different value
    print("ENVIRONMENT is set to GitHub Actions")
    @pytest.fixture
    def data():
        """ Simple function to generate some fake Pandas data."""
        data = pd.read_csv('/home/runner/work/udacity_ml_devops_fastapi/udacity_ml_devops_fastapi/starter/data/census.csv')
        return data


def test_data_shape(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    assert data.shape == (32561, 15) , "Data does not have the expected shape."
