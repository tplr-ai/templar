## Installation

Add these packages to pyproject.toml file

    "Flask==2.2.2",
    "Flask-SQLAlchemy==2.5.1",
    "Flask-Migrate==3.1.0",
    "asyncio==3.4.3",
    "botocore==1.27.9",
    "werkzeug==2.2.3"
    "psycopg2"

```bash
# Prerequisite
snap install astral-uv --classic

sudo apt-get update
sudo apt-get install libpq-dev

# Create a virtual environment
python3 -m venv env
source env/bin/activate

# Install uv (pipx is also possible to install uv) and configure venv
pip install uv && uv python install 3.12 && uv python pin 3.12 && uv venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install torch --index-url https://download.pytorch.org/whl/cu118\

# uv sync to install required packages
uv sync --extra all

# After installation
sudo apt install nvidia-cuda-toolkit

pip uninstall torch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(torch.cuda.is_available())"

# Migration commands
1. Initialize migrations: flask db init
2. Create migration file: flask db migrate -m "Initial migration"
3. Apply migrations: flask db upgrade
```