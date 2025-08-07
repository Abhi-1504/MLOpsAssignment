import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
import os
from unittest.mock import MagicMock

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FastAPI app and datamodel
from src.app import app 
from src.datamodels import HouseFeatures

# --- 1. API Integration Test ---

def test_predict_endpoint_success(mocker):
    """
    Tests if the /predict endpoint returns a successful response (status code 200)
    with valid input data. This test mocks the MLflow model loading.
    """
    # --- MOCKING THE MODEL ---
    mock_model = MagicMock()
    mock_model.predict.return_value = [250000.0]

    # Patch before TestClient triggers startup event
    mocker.patch('src.app.mlflow.pyfunc.load_model', return_value=mock_model)

    # Initialize client after mocking
    client = TestClient(app)

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

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    response_json = response.json()
    assert "predicted_median_house_value" in response_json
    assert response_json["predicted_median_house_value"] == 250000.0

def test_predict_endpoint_invalid_input():
    """
    Tests if the /predict endpoint correctly returns a 422 Unprocessable Entity
    error when a required field is missing.
    """
    client = TestClient(app)

    # Missing 'ocean_proximity'
    invalid_payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


# --- 2. Unit Test for Feature Engineering ---

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
    data = {
        'total_rooms': [1000.0, 2000.0],
        'households': [200.0, 400.0],
        'total_bedrooms': [250.0, 500.0]
    }
    sample_df = pd.DataFrame(data)

    engineered_df = create_ratio_features(sample_df)

    assert 'rooms_per_household' in engineered_df.columns
    assert 'bedrooms_per_room' in engineered_df.columns
    assert engineered_df['rooms_per_household'].iloc[0] == 5.0     # 1000 / 200
    assert engineered_df['bedrooms_per_room'].iloc[1] == 0.25      # 500 / 2000
