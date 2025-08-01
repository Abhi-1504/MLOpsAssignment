import mlflow
import uvicorn
import pandas as pd
from datamodels import HouseFeatures
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Housing Price Prediction API",
    description="An API to predict housing prices using the best registered ML model.",
    version="1.0.0"
)

model_name = "best_model_auto"
model_stage = "Production"
model_uri = f"models:/{model_name}/{model_stage}"

try:
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Successfully loaded model '{model_name}' stage '{model_stage}'")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Accepts housing features via JSON and returns a prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or available.")
    try:
        # Convert the Pydantic model to a pandas DataFrame, as the model expects it.
        input_df = pd.DataFrame([features.model_dump()])

        # Make a prediction using the loaded model.
        prediction = model.predict(input_df)

        # Return the prediction in a JSON response.
        return {"predicted_median_house_value": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "API is running", "model_loaded": model is not None}

# This block allows you to run the API directly for testing.
# Command: python app.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
