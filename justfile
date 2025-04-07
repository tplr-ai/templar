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
    sed -i "s/__version__ = .*/__version__ = \"dev-$(cat /dev/urandom \
        | tr -dc 'a-z0-9' \
        | fold -w 8 \
        | head -n 1)\"/" \
        src/tplr/__init__.py
    ./scripts/start.sh
    git restore src/tplr/__init__.py

dev:
    uv pip install --pre -e ".[dev]"

test: dev
    uv run --prerelease=allow  pytest -sv

bistro:
    ps aux | grep Bistro
