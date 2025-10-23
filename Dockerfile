# Multi-stage Dockerfile optimized for AWS ECS deployment
# Supports both CPU and GPU inference

ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG PYTHON_VERSION=3.11

# ================================
# Stage 1: Builder
# ================================
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ================================
# Stage 2: Runtime
# ================================
FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

# Metadata
LABEL maintainer="EzSpeech Team"
LABEL description="ASR WebSocket Server for AWS deployment"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY ezspeech/ /app/ezspeech/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY setup.py /app/

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models
ENV MODEL_PATH=/app/models/model.ckpt
ENV CONFIG_PATH=/app/config
ENV CONFIG_NAME=ctc_sc
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8765
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8765 8080

# Create a non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command - run WebSocket server with health check endpoint
CMD ["python", "scripts/serve_websocket.py", \
     "--model-path", "${MODEL_PATH}", \
     "--config-path", "${CONFIG_PATH}", \
     "--config-name", "${CONFIG_NAME}", \
     "--host", "${SERVER_HOST}", \
     "--port", "${SERVER_PORT}"]
