from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

import os
import joblib

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

