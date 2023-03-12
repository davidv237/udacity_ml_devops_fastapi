# Script to train machine learning model.

from ml.model import hyperparameter_tuning, train_model, compute_model_metrics, evaluate_slice_performance,inference, save_model,load_model, cat_features
from ml.data import process_data
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import joblib
import os

import pandas as pd

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
parent_directory_path = os.path.dirname(current_file_path)
# Get the parent directory of the current file
project_directory_path = os.path.dirname(parent_directory_path)

path_model_folder = os.path.join(project_directory_path,'model')
data_path = os.path.join(project_directory_path,'data/census.csv')


# Add code to load in the data.
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip()

# splitting data
train, test = train_test_split(data, test_size=0.20, random_state=42)

# processing train data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, lb=lb, encoder=encoder
)

# Optimize & Train model
print('Optimizing model ...')
optimized_model = hyperparameter_tuning(X_train, y_train)

print('Training model ...')
model = train_model(optimized_model, X_train,y_train)

#Save model
print('Saving model and encoder ...')
path_to_model, path_to_encoder = save_model(model, encoder, path_model_folder)
path_to_lb = os.path.join(path_model_folder,'lb.joblib')
joblib.dump(lb, path_to_lb)

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


slice_features = ["sex", "education"]

try:
    metrics = evaluate_slice_performance(loaded_model, test, encoder, lb, slice_features)
except ValueError:
    pass

print("Creating txt file")
# Open a file in write mode
with open('model/slice_output.txt', 'a') as file:
    # Iterate over the list and write each element to the file
    for feature, metric in metrics:
        file.write(feature + '\n')
        for key, value in metric.items():
            file.write(str(key) + ': ' + str(value) + '\n')
        file.write('\n')

print("Success ..")
