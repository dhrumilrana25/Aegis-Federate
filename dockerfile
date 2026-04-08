# Use a lightweight Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building C++ extensions (needed for some ML libs)
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose ports for the Aggregator (8080) and Dashboard (8501)
EXPOSE 8080
EXPOSE 8501