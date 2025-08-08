import mlflow
import uvicorn
import pandas as pd
from src.datamodels import HouseFeatures
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None  # Declare model globally

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global model
    
    # Set MLflow tracking URI to your server
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://13.51.85.124:1234")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    model_name = "best_model_auto"
    model_stage = "Staging"
    model_uri = f"models:/{model_name}/{model_stage}"
    
    try:
        logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
        logger.info(f"Loading model from MLflow Registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Successfully loaded model '{model_name}' stage '{model_stage}'")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
    
    yield  # App is running
    
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(
    title="Housing Price Prediction API",
    description="An API to predict housing prices using the best registered ML model.",
    version="1.0.0",
    lifespan=lifespan
)

def create_ratio_features(df):
    """Creates new ratio-based features for prediction."""
    df_copy = df.copy()
    
    # Avoid division by zero
    df_copy["rooms_per_household"] = df_copy["total_rooms"] / df_copy["households"].replace(0, 1)
    df_copy["bedrooms_per_room"] = df_copy["total_bedrooms"] / df_copy["total_rooms"].replace(0, 1)
    
    return df_copy

@app.post("/predict")
async def predict(features: HouseFeatures):
    """Predict housing price based on input features"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded or available. Check server logs."
        )
    
    try:
        # Convert pydantic model to dataframe
        input_df = pd.DataFrame([features.model_dump()])
        
        # Apply feature engineering
        input_df = create_ratio_features(input_df)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {"predicted_median_house_value": float(prediction[0])}
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during prediction: {str(e)}"
        )

@app.get("/")
async def read_root():
    """Root endpoint to check API status"""
    return {"status": "API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    mlflow_uri = mlflow.get_tracking_uri()
    return {
        "status": "healthy",
        "model_status": model_status,
        "mlflow_tracking_uri": mlflow_uri
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)