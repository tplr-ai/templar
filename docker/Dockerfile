# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

ENV PATH="/root/.cargo/bin:${PATH}"

# Copy pyproject.toml first for better caching
COPY pyproject.toml .
COPY . .

# Install dependencies using uv
RUN uv sync 

# Environment variables that can be passed
ENV WALLET_NAME=""
ENV WALLET_HOTKEY=""
ENV CUDA_DEVICE="cuda:0"
ENV NETWORK="test"
ENV DEBUG="false"
ENV NODE_TYPE="" 

# Make entrypoint script executable
RUN chmod +x /app/scripts/entrypoint.sh

# Create volumes for persistence
VOLUME ["/root/.bittensor/wallets", "/app/logs"]

# Set entrypoint
ENTRYPOINT ["/app/scripts/entrypoint.sh"] 