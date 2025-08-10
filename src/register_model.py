import mlflow
from mlflow.tracking import MlflowClient

def register_and_promote_model():
    """
    Finds the latest version of the registered model and promotes
    it to the 'Staging' environment.
    """
    print("Starting model registration and promotion...")
    
    client = MlflowClient()
    model_name = "best_model_auto"
    
    try:
        # Get the latest version of the model
        latest_versions = client.get_latest_versions(name=model_name, stages=["None"])
        if not latest_versions:
            print(f"No versions in 'None' stage found for model '{model_name}'.")
            return
            
        model_version = latest_versions[0].version
        print(f"Found latest version: {model_version} for model '{model_name}'.")
        
        # Promote the model to the "Staging" stage
        print(f"Promoting model version {model_version} to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"Success! Model version {model_version} is now in the 'Staging' stage.")

    except Exception as e:
        print(f"Error during model promotion: {e}")

if __name__ == "__main__":
    register_and_promote_model()
