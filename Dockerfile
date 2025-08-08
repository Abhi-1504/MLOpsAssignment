# Base Image
FROM python:3.12-slim

# Set Working Directory 
WORKDIR /app

# Copy and Install Dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY src/app.py .
COPY src/datamodels.py .

# Expose Port
EXPOSE 8000

# Set MLflow Tracking URI
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Run Command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]