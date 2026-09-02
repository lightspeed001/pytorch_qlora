# Use NVIDIA CUDA base image with PyTorch pre-installed
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*
        
# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.36.2 \
    datasets==2.14.5 \
    accelerate==0.24.1 \
    peft==0.6.2 \
    bitsandbytes==0.41.3 \
    trl==0.7.11 \
    sentencepiece==0.1.99 \
    scikit-learn==1.3.2

# Create and set working directory
WORKDIR /workspace

# Copy the training script
COPY finetune_mistral_qlora.py .

# Set entrypoint to run the training script
ENTRYPOINT ["python", "finetune_mistral_qlora.py"]
