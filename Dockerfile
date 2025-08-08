# Base Image
FROM python:3.12-slim

# Set Working Directory
WORKDIR /app

# Copy and Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

# This allows Python to find modules inside /app, like 'src'
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose Port
EXPOSE 8000

# Set MLflow Tracking URI
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# --- FIX: Update the run command to point to the module ---
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]