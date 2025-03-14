# Grafana Indexer Installation Guide

## Prerequisites
Before proceeding with the installation, ensure the following dependencies are installed:

```bash
snap install astral-uv --classic
sudo apt-get update
sudo apt-get install libpq-dev
```

## Setting Up a Virtual Environment
Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

## Installing uv and Configuring Virtual Environment
Install `uv` and set up the Python version:

```bash
pip install uv && uv python install 3.12 && uv python pin 3.12 && uv venv .venv
source .venv/bin/activate
```

## Installing PyTorch with CUDA Support
Install PyTorch with CUDA 11.8 support:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Installing Required Packages
Use `uv sync` to install all required dependencies:

```bash
uv sync --extra all
```

## Configuring Environment Variables
Copy and update the required environment configuration files:

```bash
cp grafana/.env.sample grafana/.env
vi grafana/.env

cp src/tplr/.env.sample src/tplr/.env
vi src/tplr/.env
```

## Applying Database Migrations
Navigate to the Grafana directory and apply database migrations:

```bash
cd grafana
flask db upgrade
```

## Running the Grafana App
Start the Grafana application using `pm2`:

```bash
cd ..
pm2 start grafana/app.py --interpreter python3 --name grafana-app
```

---

# Grafana Dashboard Setup

## Importing a Dashboard
To import a predefined dashboard, use the JSON file provided in `grafana/dashboard.json`. You can import this file directly through the Grafana web interface.

---

# Grafana Database Setup

- Database migrations are applied during the Grafana indexer installation.
- To create necessary database views, execute the SQL script located in `view.sql`.


This completes the installation and setup process for the Grafana application.
