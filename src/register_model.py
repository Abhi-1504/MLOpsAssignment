import mlflow
from mlflow.tracking import MlflowClient
import argparse
import os

def register_and_promote_model(model_name="housing_price_predictor"):
    """
    Register and promote the best model to staging
    """
    # Initialize MLflow client
    client = MlflowClient()
    
    try:
        # Get all versions of the model, regardless of current stage
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            print(f"No versions found for model '{model_name}'")
            return False
        
        # Get the latest version by version number
        latest_version = max(model_versions, key=lambda x: int(x.version))
        model_version = latest_version.version
        
        print(f"Found latest model version: {model_version}")
        print(f"Current stage: {latest_version.current_stage}")
        
        # Promote to Staging
        print(f"Promoting model version {model_version} to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )
        
        print(f"Model version {model_version} promoted to 'Staging' successfully!")
        
        # Log model details
        model_version_details = client.get_model_version(model_name, model_version)
        print(f"Model URI: {model_version_details.source}")
        print(f"Run ID: {model_version_details.run_id}")
        
        return True
        
    except Exception as e:
        print(f"Error during model registration and promotion: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register and promote model')
    parser.add_argument('--model_name', default='housing_price_predictor', 
                       help='Name of the model to register')
    
    args = parser.parse_args()
    
    success = register_and_promote_model(args.model_name)
    if not success:
        exit(1)
