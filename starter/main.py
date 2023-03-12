# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from starter.ml.model import load_model, inference
from starter.ml.data import process_data
import json

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
    print("Running GitHub Actions ..")
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

    # define examples for the input data
    class Config:
        schema_extra = {
            "example": {
                "age": 38,
                "workclass": "Private",
                'fnlwgt': 150000,
                "education": "HS-grad",
                "marital_status": "Married-civ-spouse",
                "occupation": "Handlers-cleaners",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 100000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

print("loading model, encoder, lb")
model = load_model(model_path)
encoder = load_model(encoder_path)
lb = load_model(label_encoder_path)

# Define the root domain (GET request)
@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI"}


# Define the /predict endpoint to make predictions on new data
@app.post("/predict")
async def predict(data: InputData):
    # Convert the input data to a dictionary
    data_dict = data.dict()
    # Convert the dictionary to a pandas DataFrame
    X_test = pd.DataFrame.from_dict([data_dict])

    # Preprocess the test data
    X_test_processed, _, _, _ = process_data(X_test, categorical_features=["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"], training=False, encoder=encoder, lb=lb)

    # Make predictions using the loaded model
    predictions = inference(model, X_test_processed)

    # Get original labels
    original_labels = lb.inverse_transform(predictions)

    # Return the predictions as a JSON response
    json_data = json.dumps(original_labels.tolist())

    return {"predictions": json_data}
