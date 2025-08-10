import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import psutil
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(processed_data_path, model_output_path):
    """
    Loads processed data, trains the best model configuration,
    evaluates it, and logs everything to MLflow.
    """
    print("Starting model training...")

    # Enable system metrics logging
    mlflow.system_metrics.enable_system_metrics_logging()
    
    # Load processed data
    X_train = pd.read_csv(os.path.join(processed_data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_data_path, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_path, 'y_test.csv')).values.ravel()

    # Define the preprocessing pipeline (as in the notebook)
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['ocean_proximity']

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Define the best model configuration based on experimentation
    best_params = {
        'max_depth': 10,
        'min_samples_leaf': 10
    }

    with mlflow.start_run(run_name="Automated Retraining"):
        mlflow.set_experiment("MLOPs Assignment - Housing Price Prediction")
        
        # Log parameters
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_params(best_params)
        mlflow.log_param("training_data_shape", X_train.shape)
        mlflow.log_param("cpu_count", psutil.cpu_count())
        
        # Add Tags
        mlflow.set_tag("model_family", "Tree")
        mlflow.set_tag("pipeline_type", "Automated Retraining")

        # Create the full model pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42, **best_params))
        ])

        # Track training time
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Evaluate the model on the test set
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("training_duration_sec", training_duration)

        print(f"Training complete. RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

        # Log the model pipeline
        os.makedirs(model_output_path, exist_ok=True)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="best_model_auto" # Register the model directly
        )
        print(f"Model saved and registered as 'best_model_auto'")

if __name__ == "__main__":
    train_model('data/processed', 'models/')
