#!/bin/bash
# Stop any existing processes
pm2 delete all

./scripts/clean_versions.py 

# Start all applications
pm2 start ecosystem.config.js


