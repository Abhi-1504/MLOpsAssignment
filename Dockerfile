# Use a base image with Python and MLflow
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY app.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5001

# Run the API
CMD ["python", "app.py"]
