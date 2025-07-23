import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import joblib
import os
from data_preprocessing import load_and_preprocess_data

# Set MLflow experiment
mlflow.set_experiment("iris_classification")

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model."""
    
    with mlflow.start_run(run_name="logistic_regression") as run:
        # Parameters
        params = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Create input example for model signature
        input_example = X_train[:5]  # First 5 samples as example
        
        # Log model with signature
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=input_example
        )
        
        print(f"Logistic Regression Accuracy: {accuracy:.4f}")
        
        # Return model, accuracy, and run_id
        return model, accuracy, run.info.run_id

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    
    with mlflow.start_run(run_name="random_forest") as run:
        # Parameters
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'random_state': 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("test_samples", len(y_test))
        
        # Create input example for model signature
        input_example = X_train[:5]  # First 5 samples as example
        
        # Log model with signature
        mlflow.sklearn.log_model(
            model, 
            "model",
            input_example=input_example
        )
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        
        # Return model, accuracy, and run_id
        return model, accuracy, run.info.run_id

def save_best_model(models_results, scaler):
    """Save the best performing model."""
    
    # Find best model
    best_model_name = max(models_results.keys(), 
                         key=lambda k: models_results[k][1])  # Compare accuracy
    best_model = models_results[best_model_name][0]
    best_accuracy = models_results[best_model_name][1]
    best_run_id = models_results[best_model_name][2]  # Get run_id
    
    print(f"Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Save model and scaler locally
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f" Model saved to models/best_model.pkl")
    print(f" Scaler saved to models/scaler.pkl")
    
    # Register model in MLflow (using the run_id)
    try:
        model_name = "iris_classifier_production"
        model_uri = f"runs:/{best_run_id}/model"
        
        # Register the model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f" Model registered in MLflow as '{model_name}'")
        print(f"   Model version: {registered_model.version}")
        
    except Exception as e:
        print(f" Warning: Could not register model in MLflow: {e}")
        print("   Model is still saved locally and can be used.")
    
    return best_model

if __name__ == "__main__":
    # Load and preprocess data
    print(" Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Train models
    print("\n Training models...")
    models_results = {}
    
    print("\n Training Logistic Regression...")
    lr_model, lr_accuracy, lr_run_id = train_logistic_regression(X_train, y_train, X_test, y_test)
    models_results['logistic_regression'] = (lr_model, lr_accuracy, lr_run_id)
    
    print("\n Training Random Forest...")
    rf_model, rf_accuracy, rf_run_id = train_random_forest(X_train, y_train, X_test, y_test)
    models_results['random_forest'] = (rf_model, rf_accuracy, rf_run_id)
    
    # Save best model
    print("\n Selecting and saving best model...")
    best_model = save_best_model(models_results, scaler)
    
    print("\n Model training completed!")
    print(f" Check MLflow UI at: http://localhost:5000")
    print(f"Models saved in: models/ directory")