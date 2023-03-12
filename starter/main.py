# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from starter.ml.model import load_model, cat_features, inference
from starter.ml.data import process_data
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os

# Define the FastAPI app
app = FastAPI()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

except ImportError:
    print("dotenv is not installed")


if 'ENVIRONMENT' in os.environ and os.environ['ENVIRONMENT'] == 'development':
    # Do something if the environment variable is set to 'some_value'
    print("Environment is set to 'development'")


    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    print(current_file_path)

    # Get the parent directory of the current file
    parent_directory_path = os.path.dirname(current_file_path)

    # Get the parent directory of the current file
    project_directory_path = os.path.dirname(parent_directory_path)

    model_path = 'model/randomforest.joblib'
    encoder_path = 'model/encoder.joblib'
    label_encoder_path = 'model/lb.joblib'

elif 'ENVIRONMENT' in os.environ and os.environ['ENVIRONMENT'] == 'production':
    # Do something if the environment variable is set to 'some_value'
    print("Environment is set to 'production'")
    model_path = '/app/starter/model/randomforest.joblib'
    encoder_path = '/app/starter/model/encoder.joblib'
    label_encoder_path = '/app/starter/model/lb.joblib'

else:
    # Do something else if the environment variable is not set or has a different value
    print("ENVIRONMENT is set to GitHub Actions")
    model_path = "/home/runner/work/udacity_ml_devops_fastapi/udacity_ml_devops_fastapi/starter/model/randomforest.joblib"
    encoder_path = "/home/runner/work/udacity_ml_devops_fastapi/udacity_ml_devops_fastapi/starter/model/encoder.joblib"
    label_encoder_path = "/home/runner/work/udacity_ml_devops_fastapi/udacity_ml_devops_fastapi/starter/model/lb.joblib"



# Define the input schema using Pydantic
class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


print("loading model, encoder, lb")
model = load_model(model_path)
encoder = load_model(encoder_path)
lb = load_model(label_encoder_path)


# # Load the trained model and encoder
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('encoder.pkl', 'rb') as f:
#     encoder = pickle.load(f)

# Define the root domain (GET request)
@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI"}


# Define the /predict endpoint to make predictions on new data
@app.post("/predict")
async def predict(data: InputData):
    # Convert the input data to a dictionary
    print("data_dict")
    data_dict = data.dict()
    print(data_dict)

    # Convert the dictionary to a pandas DataFrame
    X_test = pd.DataFrame.from_dict([data_dict])

    # Preprocess the test data
    X_test_processed, _, _, _ = process_data(X_test, categorical_features=["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"], training=False, encoder=encoder, lb=lb)

    # Make predictions using the loaded model
    print("predictions")
    predictions = inference(model, X_test_processed)


    original_labels = lb.inverse_transform(predictions)
    # print(original_labels)
    # print(response.json())
    print(original_labels)


    # Return the predictions as a JSON response
    json_data = json.dumps(original_labels.tolist())


    return {"predictions": json_data}
