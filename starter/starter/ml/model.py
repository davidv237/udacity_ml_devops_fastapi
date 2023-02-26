from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from starter.ml.data import process_data

import os
import joblib

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

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Define the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)

    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(model, pth):
    """ Save model to specific location.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    pth: str
        String with path

    """
    path_to_model = os.path.join(pth,'logistic_regression.joblib')
    joblib.dump(model, path_to_model)
    return path_to_model

def load_model(path_to_model):
    """ Load model from specific location.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    pth: str
        String with path

    """
    model = joblib.load(path_to_model)
    return model

