import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FastAPI app and datamodel (after mocking)
from src.datamodels import HouseFeatures

# --- 1. API Integration Tests ---
@patch('mlflow.get_tracking_uri')
@patch('mlflow.set_tracking_uri') 
@patch('mlflow.pyfunc.load_model')
def test_predict_endpoint_success(mock_load_model, mock_set_uri, mock_get_uri):
    """Test successful prediction endpoint with mocked MLflow model"""
    # Setup mocks
    mock_model = MagicMock()
    mock_model.predict.return_value = [250000.0]
    mock_load_model.return_value = mock_model
    mock_get_uri.return_value = "http://localhost:5000"
    
    # Import app after mocking to avoid MLflow connection issues
    from src.app import app
    
    # Use TestClient in context to trigger startup AFTER patching
    with TestClient(app) as client:
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

@patch('mlflow.get_tracking_uri')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.pyfunc.load_model')
def test_predict_endpoint_invalid_input(mock_load_model, mock_set_uri, mock_get_uri):
    """
    Tests if the /predict endpoint correctly returns a 422 Unprocessable Entity
    error when a required field is missing.
    """
    # Setup mocks
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_get_uri.return_value = "http://localhost:5000"
    
    from src.app import app
    
    with TestClient(app) as client:
        # Missing 'ocean_proximity' - required field
        invalid_payload = {
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252
            # Missing "ocean_proximity"
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

# --- 3. Pydantic Model Validation Tests ---
def test_house_features_validation():
    """Test HouseFeatures pydantic model validation"""
    # Valid data
    valid_data = {
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
    
    features = HouseFeatures(**valid_data)
    assert features.longitude == -122.23
    assert features.ocean_proximity == "NEAR BAY"

def test_house_features_invalid_ocean_proximity():
    """Test validation with invalid ocean_proximity value"""
    from pydantic import ValidationError
    
    invalid_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "INVALID_VALUE"  # This should be invalid
    }
    
    with pytest.raises(ValidationError):
        HouseFeatures(**invalid_data)

# --- 4. Additional Tests (improvements over original) ---
@patch('mlflow.get_tracking_uri')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.pyfunc.load_model')
def test_predict_endpoint_model_not_loaded(mock_load_model, mock_set_uri, mock_get_uri):
    """Test behavior when model fails to load"""
    # Simulate model loading failure
    mock_load_model.side_effect = Exception("Model 'best_model_auto' not found")
    mock_get_uri.return_value = "http://localhost:5000"
    
    from src.app import app
    
    with TestClient(app) as client:
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
        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

@patch('mlflow.get_tracking_uri')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.pyfunc.load_model')
def test_root_endpoint(mock_load_model, mock_set_uri, mock_get_uri):
    """Test the root endpoint"""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_get_uri.return_value = "http://localhost:5000"
    
    from src.app import app
    
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "API is running"}

def test_feature_engineering_edge_cases():
    """Test feature engineering with edge cases (division by zero)"""
    data = {
        'total_rooms': [0.0, 2000.0],
        'households': [200.0, 0.0],
        'total_bedrooms': [250.0, 500.0]
    }
    sample_df = pd.DataFrame(data)
    
    # Your original function doesn't handle division by zero
    # This test documents the current behavior
    engineered_df = create_ratio_features(sample_df)
    
    assert 'rooms_per_household' in engineered_df.columns
    assert 'bedrooms_per_room' in engineered_df.columns
    assert engineered_df['rooms_per_household'].iloc[0] == 0.0     # 0 / 200
    # The second row will be inf due to division by zero (2000 / 0)
    assert engineered_df['rooms_per_household'].iloc[1] == float('inf')

def test_house_features_missing_required_field():
    """Test validation with missing required field"""
    from pydantic import ValidationError
    
    incomplete_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        # Missing other required fields
    }
    
    with pytest.raises(ValidationError):
        HouseFeatures(**incomplete_data)