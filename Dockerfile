# Base Python Image
FROM python:3.12-slim

# Set Working Directory 
WORKDIR /src

# Copy and Install Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create src library
RUN mkdir src

#  Copy Application Code 
COPY src/app.py src/
COPY src/datamodels.py src

# Expose Port
EXPOSE 8000

# Set MLflow Tracking URI
# Set an environment variable to tell the app where to find the MLflow server.
# ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 #for MAC and windows
ENV MLFLOW_MODEL_URL=http://127.0.0.1:1234/invocations

# Run Command
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]