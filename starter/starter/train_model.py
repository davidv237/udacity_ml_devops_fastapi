# Script to train machine learning model.


from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference, load_model, save_model, evaluate_slices, cat_features
from ml.data import process_data, prepare_data



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

results = []

for feature in cat_features:
    try:
        results.append(evaluate_slices(loaded_model, data, feature))
    except ValueError:
        pass

print(results)
