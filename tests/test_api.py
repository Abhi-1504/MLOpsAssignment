import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
import os
from unittest.mock import MagicMock

# Add the root directory of the project to the Python path
# This allows us to import modules from the 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from our source files
# Note: This assumes you have an __init__.py file in your 'src' folder
from src.app import app 
from src.datamodels import HouseFeatures

# --- 1. API Integration Test ---

# Create a TestClient instance for our FastAPI app
client = TestClient(app)

def test_predict_endpoint_success(mocker):
    """
    Tests if the /predict endpoint returns a successful response (status code 200)
    with valid input data. This test mocks the MLflow model loading.
    """
    # --- MOCKING THE MODEL ---
    # Create a fake model object with a predict method
    mock_model = MagicMock()
    # Configure the predict method to return a predictable value
    mock_model.predict.return_value = [250000.0] 
    
    # Replace the real `mlflow.pyfunc.load_model` with our fake model
    # The path 'src.app.mlflow.pyfunc.load_model' points to where the function is used.
    mocker.patch('src.app.mlflow.pyfunc.load_model', return_value=mock_model)
    
    # We also need to mock the model object within the app itself for the test client
    app.model = mock_model
    # --- END MOCKING ---

    # Define a valid sample payload
    sample_payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }
    
    # Send a POST request to the /predict endpoint
    response = client.post("/predict", json=sample_payload)
    
    # Assert that the status code is 200 (OK)
    assert response.status_code == 200
    
    # Assert that the response is a JSON object
    response_json = response.json()
    assert isinstance(response_json, dict)
    
    # Assert that the prediction key is in the response
    assert "predicted_median_house_value" in response_json
    
    # Assert that the prediction value is the one we set in our mock
    assert response_json["predicted_median_house_value"] == 250000.0

def test_predict_endpoint_invalid_input():
    """
    Tests if the /predict endpoint correctly returns a 422 Unprocessable Entity
    error when a required field is missing.
    """
    # Define a payload with a missing 'ocean_proximity' field
    invalid_payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
        # Missing 'ocean_proximity'
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    # Assert that the status code is 422, which is what FastAPI returns for validation errors
    assert response.status_code == 422

# --- 2. Unit Test for Feature Engineering ---

# A standalone function to test our feature engineering logic
def create_ratio_features(df):
    """Creates new ratio-based features."""
    df_copy = df.copy()
    df_copy["rooms_per_household"] = df_copy["total_rooms"] / df_copy["households"]
    df_copy["bedrooms_per_room"] = df_copy["total_bedrooms"] / df_copy["total_rooms"]
    return df_copy

def test_feature_engineering():
    """
    Tests if the feature engineering function correctly calculates new columns.
    """
    # Create a sample DataFrame
    data = {
        'total_rooms': [1000.0, 2000.0],
        'households': [200.0, 400.0],
        'total_bedrooms': [250.0, 500.0]
    }
    sample_df = pd.DataFrame(data)
    
    # Apply the feature engineering function
    engineered_df = create_ratio_features(sample_df)
    
    # Assert that the new columns were created
    assert 'rooms_per_household' in engineered_df.columns
    assert 'bedrooms_per_room' in engineered_df.columns
    
    # Assert that the calculations are correct
    assert engineered_df['rooms_per_household'].iloc[0] == 5.0  # 1000 / 200
    assert engineered_df['bedrooms_per_room'].iloc[1] == 0.25 # 500 / 2000
