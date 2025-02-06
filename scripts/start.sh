#!/bin/bash
# Stop any existing processes
pm2 delete all

# hack : check if there are any zombie processes
ps aux | grep Bistro

./scripts/clean_versions.py 

# Start all applications
pm2 start ecosystem.config.js


pm2 log TV1


