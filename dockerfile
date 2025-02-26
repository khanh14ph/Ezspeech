# Use PyTorch's official CUDA image as base
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libopenblas-dev \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for ASR
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CUDA-specific environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Copy the ASR application
COPY . .

# Default command to run the ASR model
CMD ["python", "main.py"]