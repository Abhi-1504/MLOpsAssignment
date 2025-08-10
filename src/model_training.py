import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import argparse

def train_model(processed_data_path, model_output_path):
    """
    Train the housing price prediction model using preprocessed data
    """
    # Enable system metrics tracking (matching experimentation notebook)
    mlflow.system_metrics.enable_system_metrics_logging()
    
    # Set MLflow experiment (EXACT SAME NAME as experimentation)
    mlflow.set_experiment("MLOPs Assignment - Housing Price Prediction")
    
    # Load processed data
    X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()
    
    # Load the fitted preprocessor
    preprocessor = joblib.load(os.path.join(processed_data_path, 'preprocessor.pkl'))
    
    # Transform the data using the fitted preprocessor
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Start MLflow run with descriptive name
    with mlflow.start_run(run_name="automated_retraining_decision_tree"):
        # Log dataset info
        mlflow.log_param("dataset_size", len(X_train))
        mlflow.log_param("n_features", X_train_processed.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("training_data_shape", X_train.shape)
        
        # Train the model (matching experimentation best params)
        model = DecisionTreeRegressor(random_state=42)
        
        # Track training time (matching experimentation format)
        import time
        start_time = time.time()
        model.fit(X_train_processed, y_train)
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log model parameters (matching experimentation format)
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_depth", None)  # Default value
        mlflow.log_param("min_samples_leaf", 1)  # Default value
        mlflow.log_param("preprocessing", "ColumnTransformer_with_feature_engineering")
        
        # Log metrics (matching experimentation format)
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "training_duration_sec": training_duration
        })
        
        # Log additional tags for pipeline identification (matching experimentation)
        mlflow.set_tag("model_family", "Tree")
        mlflow.set_tag("experiment_type", "Automated_Pipeline")
        mlflow.set_tag("pipeline_stage", "automated_retraining")
        mlflow.set_tag("model_purpose", "production_candidate")
        mlflow.set_tag("data_version", "latest")
        
        # Create output directory
        os.makedirs(model_output_path, exist_ok=True)
        
        # Save model locally
        model_path = os.path.join(model_output_path, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Log model to MLflow with the same name as experimentation
        model_info = mlflow.sklearn.log_model(
            model, 
            "model",  # Match artifact name from experimentation
            registered_model_name="best_model_auto"
        )
        
        # Log model URI for reference
        mlflow.log_param("model_uri", model_info.model_uri)
        
        # Save metrics to file for DVC tracking
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "training_duration_sec": float(training_duration),
            "model_uri": model_info.model_uri,
            "run_id": mlflow.active_run().info.run_id
        }
        
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("✅ Model training completed successfully!")
        print(f"🧪 Experiment: MLOPs Assignment - Housing Price Prediction")
        print(f"🏃 Run ID: {mlflow.active_run().info.run_id}")
        print(f"📊 RMSE: {rmse:.2f}")
        print(f"📊 MAE: {mae:.2f}")
        print(f"📊 R² Score: {r2:.4f}")
        print(f"⏱️ Training Duration: {training_duration:.2f}s")
        print(f"💾 Model saved to: {model_path}")
        print(f"🔗 Model URI: {model_info.model_uri}")
        print("📈 System metrics logging enabled")
        
        return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train housing price prediction model')
    parser.add_argument('--processed_data', required=True, help='Path to processed data directory')
    parser.add_argument('--model_output', required=True, help='Path to save trained model')
    
    args = parser.parse_args()
    
    train_model(args.processed_data, args.model_output)
