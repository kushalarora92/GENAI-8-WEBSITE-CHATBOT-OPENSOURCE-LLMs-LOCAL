# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire application first
COPY . .

# Create required directories with proper permissions
RUN mkdir -p /app/static/docs /app/templates && \
    chown -R 1000:1000 /app/static /app/templates

# Install dependencies including the local package
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user with UID 1000
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8008

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]
