import requests
import os
from ml.model import load_model

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the parent directory of the current file
parent_directory_path = os.path.dirname(current_file_path)
# Get the parent directory of the current file
project_directory_path = os.path.dirname(parent_directory_path)
label_encoder_path = os.path.join(project_directory_path,'model/lb.joblib')
lb = load_model(label_encoder_path)


data = {
        'age': 35,
        'workclass': 'Self-emp-not-inc',
        'fnlwgt': 150000,
        'education': 'Masters',
        'education_num': 14,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Husband',
        'race': 'Asian-Pac-Islander',
        'sex': 'Male',
        'capital_gain': 100000,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'India'
    }

response = requests.post('https://udacity-ml-devops-fastapi.herokuapp.com/predict', json=data)

# Reverse the transformation

# Get the prediction result from the response
prediction = response.json()
prediction = prediction["predictions"]
status_code = response.status_code

print(f"Status: {status_code}")
print(f"Prediction: {prediction}")
