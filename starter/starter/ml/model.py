from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,  RandomizedSearchCV
from ml.data import process_data
from scipy.stats import randint

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
def hyperparameter_tuning(X_train, y_train):
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
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt']
    }
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': [10, 20, 30, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False]
    }
    # Define the Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)


    # Perform grid search to find the best hyperparameters
    grid_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist, n_iter=10, cv=3,verbose=2)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a new classifier with the best hyperparameters
    best_rfc = RandomForestClassifier(random_state=42, **best_params)

    return best_rfc


def train_model(model, X_train, y_train):
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
    #model = LogisticRegression(max_iter=1000)
    #model.fit(X_train,y_train)

    #rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

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

def save_model(model, encoder, pth):
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

    path_to_encoder = os.path.join(pth,'encoder.joblib')
    joblib.dump(encoder, path_to_encoder)

    return path_to_model, path_to_encoder

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




