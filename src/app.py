import mlflow
import uvicorn
import pandas as pd
from src.datamodels import HouseFeatures
from fastapi import FastAPI, HTTPException
import os

app = FastAPI(
    title="Housing Price Prediction API",
    description="An API to predict housing prices using the best registered ML model.",
    version="1.0.0"
)

model = None  # Declare model globally

@app.on_event("startup")
def load_model():
    global model
    model_name = "best_model_auto"
    model_stage = "Staging"
    model_uri = f"models:/{model_name}/{model_stage}"

    try:
        print(f"Loading model from MLflow Registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{model_name}' stage '{model_stage}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or available. Check server logs.")

    try:
        input_df = pd.DataFrame([features.model_dump()])
        input_df['rooms_per_household'] = input_df['total_rooms'] / input_df['households']
        input_df['bedrooms_per_room'] = input_df['total_bedrooms'] / input_df['total_rooms']
        prediction = model.predict(input_df)
        return {"predicted_median_house_value": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
