import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import FastAPI app and datamodel
from src.datamodels import HouseFeatures

# API Integration Tests

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
    
    # Import app after mocking to avoid MLflow connection issues during import
    from src.app import app
    
    # Use TestClient as a context manager to handle the lifespan events
    with TestClient(app) as client:
        sample_payload = {
            "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
            "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
            "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"
        }
        
        response = client.post("/predict", json=sample_payload)
        
        assert response.status_code == 200
        response_json = response.json()
        assert "predicted_median_house_value" in response_json
        assert response_json["predicted_median_house_value"] == 250000.0

@patch('mlflow.get_tracking_uri')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.pyfunc.load_model')
def test_health_endpoint(mock_load_model, mock_set_uri, mock_get_uri):
    """Test the /health endpoint to ensure it returns the correct status."""
    # Setup mocks
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_get_uri.return_value = "http://localhost:5000"
    
    from src.app import app
    
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "healthy"
        assert response_json["model_status"] == "loaded"

@patch('mlflow.get_tracking_uri')
@patch('mlflow.set_tracking_uri')
@patch('mlflow.pyfunc.load_model')
def test_metrics_endpoint(mock_load_model, mock_set_uri, mock_get_uri):
    """Test that the /metrics endpoint is exposed and returns Prometheus metrics."""
    # Setup mocks
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_get_uri.return_value = "http://localhost:5000"
    
    from src.app import app
    
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        # Check for a known Prometheus metric in the response text
        assert 'http_requests_total' in response.text

# Unit Test for Feature Engineering
def test_feature_engineering():
    """
    Tests if the feature engineering function correctly calculates new columns.
    """
    from src.app import create_ratio_features
    data = {
        'total_rooms': [1000.0, 2000.0],
        'households': [200.0, 400.0],
        'total_bedrooms': [250.0, 500.0]
    }
    sample_df = pd.DataFrame(data)
    engineered_df = create_ratio_features(sample_df)
    
    assert 'rooms_per_household' in engineered_df.columns
    assert 'bedrooms_per_room' in engineered_df.columns
    assert engineered_df['rooms_per_household'].iloc[0] == 5.0
    assert engineered_df['bedrooms_per_room'].iloc[1] == 0.25
