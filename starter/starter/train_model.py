# Script to train machine learning model.


from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference, load_model, save_model
from ml.data import process_data, prepare_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Add the necessary imports for the starter code.
import joblib
import os

import pandas as pd


cwd = os.getcwd()
data_path = os.path.join(cwd,'data/census.csv')
print(data_path)
model_path = os.path.join(cwd,'./model')


# Add code to load in the data.
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()


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


# Splitting and preparing data
X_train, y_train, X_test, y_test = prepare_data(data, cat_features)

# Train model
print('Training model ...')
model = train_model(X_train,y_train)

#Save model
print('Saving model ...')
path_to_model = save_model(model, cwd)

#Load model
print('Loading model ...')
loaded_model = load_model(path_to_model)


print('Making predictions ...')
preds = inference(loaded_model, X_test)
print(preds)


print('Printing model scores..')
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Fbeta: {fbeta}")


def evaluate_slices(model, data, feature):

    train, test = train_test_split(data, test_size=0.20, random_state=42)
    _, _, encoder, lb = process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )
    unique_values = data[feature].unique()
    metrics = {}

    for value in unique_values:
        subset = test[test[feature] == value]
        X_test, y_test, encoder, lb = process_data(
                subset, categorical_features=cat_features, label="salary", training=False, lb=lb, encoder=encoder
            )
        y_pred = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
        metrics[value] = {
            "Precision": precision,
            "Recall": recall,
            "Fbeta": fbeta
        }

    return metrics


metrics = evaluate_slices(loaded_model, data, 'education')
print(metrics)
