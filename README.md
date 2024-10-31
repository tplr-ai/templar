# τemplar: Incentivized Wide-Internet Training

## Getting Started

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

2. Set up the environment:
```bash
uv venv
source .venv/bin/activate
uv sync --extra all
uv pip install flash-attn --no-build-isolation
```
