# Default recipe to run when just is called without arguments
default:
    @just --list

# Run ruff check with auto-fix and format
lint:
    ruff check --fix .
    ruff format .

# Run both check and format in a single command
fix: lint

test-run:
    ./scripts/start.sh

dev:
    uv pip install -e ".[dev]"

test:
    uv run pytest -sv

bistro:
    ps aux | grep Bistro