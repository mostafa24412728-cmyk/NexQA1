# Use Python 3.10 slim base
FROM python:3.10-slim

# Install system dependencies for OpenCV and YOLOv8
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create uploads directory
RUN mkdir -p static/uploads

# Railway injects $PORT automatically — expose it
EXPOSE $PORT

# Use gunicorn with Railway's dynamic $PORT
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
