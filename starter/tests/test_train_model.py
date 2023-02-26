import pytest
import pandas as pd
import os

cwd = os.getcwd()
print("cwd")
print(cwd)

parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
print("parent_dir")
print(parent_dir)

data_path = './data/census.csv'

data_path = os.path.join(parent_dir,data_path)


@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    data = pd.read_csv(data_path)
    return data

def test_data_shape(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    assert data.shape == (32561, 15) , "Data does not have the expected shape."