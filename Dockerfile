# Base Python Image
FROM python:3.12-slim

# Set Working Directory 
WORKDIR /app

# Copy and Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  Copy Application Code 
COPY app.py .
COPY datamodels.py .

# Expose Port
EXPOSE 8000

# Set MLflow Tracking URI
# Set an environment variable to tell the app where to find the MLflow server.
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Run Command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]