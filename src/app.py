from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# ✅ Set the correct MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Name of the model registered
MODEL_NAME = "best_model_auto"

# ✅ Load model from 'Production' stage
print(f"📦 Loading model from: models:/{MODEL_NAME}/Production")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
