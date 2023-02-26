# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.model import train_model
from ml.data import process_data


# Add the necessary imports for the starter code.
import joblib
import os

import pandas as pd


cwd = os.getcwd()
print("cwd")
print(cwd)

parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
print("parent_dir")
print(parent_dir)

data_path = './data/census.csv'
model_path = './model'

data_path = os.path.join(parent_dir,data_path)
model_path = os.path.join(parent_dir,model_path)


# Add code to load in the data.
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

print("data.shape")
print(data.shape)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
print('Training model ...')
model = train_model(X_train,y_train)

#Save model
print('Saving model ...')
filename = os.path.join(model_path,'logistic_regression.joblib')
joblib.dump(model, filename)
