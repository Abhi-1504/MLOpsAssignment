import mlflow
import uvicorn
import pandas as pd
from src.datamodels import HouseFeatures
from fastapi import FastAPI, HTTPException
import os

# Create the FastAPI application 
app = FastAPI(
    title="Housing Price Prediction API",
    description="An API to predict housing prices using the best registered ML model.",
    version="1.0.0"
)

# Load the registered MLflow model
model_name = "best_model_auto"
model_stage = "Staging"
model_uri = f"models:/{model_name}/{model_stage}"

model = None
try:
    print(f"Loading model from MLflow Registry: {model_uri}")
    # Downloading the promoted model
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Successfully loaded model '{model_name}' stage '{model_stage}'")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the prediction endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Accepts housing features via JSON and returns a prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or available. Check server logs.")

    try:
        # Convert the Pydantic model to a pandas DataFrame, as the model's
        # signature expects this format.
        input_df = pd.DataFrame([features.model_dump()])

        # Adding feature engineered features
        input_df['rooms_per_household'] = input_df['total_rooms'] / input_df['households']
        input_df['bedrooms_per_room'] = input_df['total_bedrooms'] / input_df['total_rooms']

        # Make a prediction using the loaded model.
        prediction = model.predict(input_df)

        # Return the prediction in a JSON response.
        return {"predicted_median_house_value": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


@app.get("/")
def read_root():
    """A simple health check endpoint to verify the API is running."""
    return {"status": "API is running"}

# This block allows you to run the API directly for testing.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
