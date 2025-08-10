import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import psutil
import json
import joblib
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

def train_model(processed_data_path, model_output_path):
    """
    Loads processed data, defines a full preprocessing and model pipeline,
    trains it, evaluates it, and logs everything to MLflow.
    """
    print("--- Starting model training stage ---")

    # Enable automatic logging of system resource usage
    mlflow.system_metrics.enable_system_metrics_logging()
    
    # Load processed data (which includes engineered features)
    X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()

    # Define the preprocessing steps for the pipeline
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist() + ['rooms_per_household', 'bedrooms_per_room']
    categorical_features = ['ocean_proximity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # Use the best hyperparameters found during experimentation
    best_params = {'max_depth': 10, 'min_samples_leaf': 10}

    with mlflow.start_run(run_name="Automated_Retraining_Pipeline"):
        mlflow.set_experiment("MLOPs Assignment - Housing Price Prediction")
        
        # Log parameters for this run
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_params(best_params)
        mlflow.log_param("training_data_shape", X_train.shape)
        mlflow.log_param("cpu_count", psutil.cpu_count())
        
        # Add descriptive tags
        mlflow.set_tag("model_family", "Tree")
        mlflow.set_tag("pipeline_type", "Automated Retraining")

        # Define the full model pipeline, bundling preprocessing and the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42, **best_params))
        ])

        # Train the model and track duration
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_duration = time.time() - start_time
        
        # Evaluate the model on the test set
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("training_duration_sec", training_duration)

        print(f"Training complete. RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

        # Define Model Signature based on the raw input format
        input_schema = Schema([
            ColSpec("double", "longitude"), ColSpec("double", "latitude"),
            ColSpec("double", "housing_median_age"), ColSpec("double", "total_rooms"),
            ColSpec("double", "total_bedrooms"), ColSpec("double", "population"),
            ColSpec("double", "households"), ColSpec("double", "median_income"),
            ColSpec("string", "ocean_proximity"),
            ColSpec("double", "rooms_per_household"), ColSpec("double", "bedrooms_per_room"),
        ])
        output_schema = Schema([ColSpec("double")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Log the entire pipeline to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name="best_model_auto"
        )
        print("Full model pipeline saved and registered to MLflow as 'best_model_auto'")

        # Save local outputs for DVC to track
        os.makedirs(model_output_path, exist_ok=True)
        joblib.dump(pipeline, os.path.join(model_output_path, "model.pkl"))
        
        metrics = {"rmse": rmse, "r2_score": r2}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        
        print("Local model and metrics files saved for DVC tracking.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()
    
    train_model(args.processed_data, args.model_output)
