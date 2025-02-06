#!/bin/bash

# Create a specific directory for profiles
PROFILE_DIR="$HOME/nsys_profiles"
mkdir -p "$PROFILE_DIR"

# Print current user and permissions for debugging
echo "Running as user: $(whoami)"
echo "Output directory: $PROFILE_DIR"

# Get the path to the virtual environment's Python
VENV_PYTHON="$(which python3)"
echo "Using Python from: $VENV_PYTHON"

# Set required environment variables
export NSYS_PROFILING_SESSION_ID="$$"

# Run nsys with sudo but preserve the virtual environment
if command -v sudo >/dev/null 2>&1; then
    sudo -E nsys profile \
        --trace=cuda,nvtx,osrt \
        --output="$PROFILE_DIR/validator_profile_$(date +%Y%m%d_%H%M%S)" \
        --session-new=true \
        --force-overwrite=true \
        --trace-fork-before-exec=true \
        "$VENV_PYTHON" neurons/validator.py "$@"
else
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --output="$PROFILE_DIR/validator_profile_$(date +%Y%m%d_%H%M%S)" \
        --session-new=true \
        --force-overwrite=true \
        "$VENV_PYTHON" neurons/validator.py "$@"
fi