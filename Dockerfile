# Dockerfile
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app

# Run the Python script
CMD ["python", "parse_documents.py"]
