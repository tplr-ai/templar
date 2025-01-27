source .venv/bin/activate

 # Install PyTorch with CUDA support
 uv pip install torch --index-url https://download.pytorch.org/whl/cu118\

 # uv sync to install required packages
 uv sync --extra all
