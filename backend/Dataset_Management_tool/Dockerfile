FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    MPLCONFIGDIR=/app/.matplotlib \
    HOME=/app

# Create an unprivileged user with a home directory
RUN groupadd -r appuser && useradd -m -g appuser appuser

# Install Python 3.12, venv, git, and required system packages for YOLO/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set proper permissions so YOLO can write to required directories
RUN mkdir -p /app/.ultralytics /app/.matplotlib /app/.config /app/app/storage && \
    chown -R appuser:appuser /app /opt/venv

# Switch to the non-root user
USER appuser

EXPOSE 8000

# Run uvicorn in production mode (workers=1 to prevent GPU OOM during training)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
