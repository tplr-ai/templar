# Default recipe to run when just is called without arguments
default:
    @just --list

# Run ruff check with auto-fix and format
lint:
    ruff check --fix .
    ruff format .

# Run both check and format in a single command
fix: lint

# Run the application and monitor with PM2
test-run:
    ./scripts/start.sh

# Install package in development mode using uv package manager
dev:
    uv pip install -e ".[dev]"

# Run all tests with verbose output
test:
    uv run pytest -sv

# Run specific test file
test-file file:
    uv run pytest -sv {{file}}

# Run tests with coverage
test-cov:
    uv run pytest -sv --cov=src --cov-report=term-missing

# Run tests in parallel
# Do not use this . Current test have weird async behaviour 
# and it breaks when you try to run this
test-parallel:
    uv run pytest -sv -n auto

# Run tests matching a specific pattern
test-k pattern:
    uv run pytest -sv -k "{{pattern}}"

bistro:
    ps aux | grep Bistro