import uvicorn
import mlflow
import requests
import pandas as pd
from src.datamodels import HouseFeatures
from fastapi import FastAPI, HTTPException
import os

app = FastAPI(
    title="Housing Price Prediction API",
    description="An API to predict housing prices using the best registered ML model.",
    version="1.0.0"
)
# Get MLflow model URL from environment variable
MLFLOW_MODEL_URL = os.environ.get("MLFLOW_MODEL_URL", "http://localhost:1234/invocations")

@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Accepts housing features via JSON and returns a prediction.
    """
    try:
        # Convert the Pydantic model to a pandas DataFrame, as the model expects it.
        input_df = pd.DataFrame([features.model_dump()])
        input_df["rooms_per_household"] = input_df["total_rooms"] / input_df["households"]
        input_df["bedrooms_per_room"] = input_df["total_bedrooms"] / input_df["total_rooms"]

         # Format as MLflow expects: dataframe_split orientation
        payload = {"dataframe_split": input_df.to_dict(orient="split")}

        # Send request to MLflow model server
        response = requests.post(MLFLOW_MODEL_URL, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"MLflow prediction error: {response.text}")

        prediction = response.json()
        return {"predicted_median_house_value": prediction}

    except Exception as e:
        raise e
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running", "model_loaded": model is not None}

# This block allows you to run the API directly for testing.
# Command: python app.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
