# Use official Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies required for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (FastAPI default)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/docs', timeout=5)" || exit 1

# Run the application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
