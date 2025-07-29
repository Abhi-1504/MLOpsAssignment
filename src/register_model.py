import mlflow
from mlflow.tracking import MlflowClient

# Constants
EXPERIMENT_NAME = "Housing Price Prediction-2"
REGISTERED_MODEL_NAME = "best_model_auto"

# Set MLflow URI (adjust this if using a remote server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or leave empty for local runs

client = MlflowClient()

# Get experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"❌ No experiment found with name: '{EXPERIMENT_NAME}'")

experiment_id = experiment.experiment_id
print(f"📘 Using experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")

# Get best run by highest r2_score
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="attributes.status = 'FINISHED' and metrics.r2_score > 0",
    order_by=["metrics.r2_score DESC"],
    max_results=1
)

if not runs:
    raise ValueError("❌ No successful runs found with r2_score in this experiment.")

best_run = runs[0]
run_id = best_run.info.run_id
r2_score = best_run.data.metrics["r2_score"]
print(f"🏆 Best run ID: {run_id}, R² Score: {r2_score:.4f}")

# Path to the model (you used artifact_path="model" while logging)
model_uri = f"runs:/{run_id}/model"
print(f"📦 Registering model from: {model_uri}")

# Register the model
try:
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"✅ Model registered: {result.name}, Version: {result.version}")
except Exception as e:
    print(f"❌ Failed to register model: {e}")
