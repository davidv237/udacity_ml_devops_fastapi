import pytest
import pandas as pd
import os



@pytest.fixture
def data():
    cwd = os.getcwd()
    print("cwd")
    print(cwd)
    files = os.listdir(cwd)

    print(files)
    data_path = '/starter/data/census.csv'
    data_path = os.path.join(cwd,data_path)
    print(data_path)

    """ Simple function to generate some fake Pandas data."""
    data = pd.read_csv(data_path)
    return data

def test_data_shape(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    assert data.shape == (32561, 15) , "Data does not have the expected shape."
