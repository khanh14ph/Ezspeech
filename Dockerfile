# Use PyTorch's official CUDA image as base
FROM python:3.12-slim

# Set the working directory
WORKDIR /deployment/

# Install system dependencies required for audio processing

# Install Python dependencies for ASR
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the ASR application
COPY ezspeech .

# Default command to run the ASR model
CMD ["python", "main.py"]