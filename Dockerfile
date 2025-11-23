# Use an official Python base image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt /app/requirements.txt

# If you use heavy torch wheel separately, you can keep the two-step install you used earlier
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and startup script
COPY . /app

# Make start script executable
RUN chmod +x /app/start.sh

EXPOSE $PORT

# Start script will create /root/.postgresql/root.crt from env var and start uvicorn
CMD ["/app/start.sh"]
