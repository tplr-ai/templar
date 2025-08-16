#!/bin/bash
# Stop any existing processes
pm2 delete all

# hack : check if there are any zombie processes
ps aux | grep Bistro

# ./scripts/clean_versions.py 

runfile=${1:-"ecosystem.config.js"}
# Start all applications
pm2 start $runfile


pm2 log TM1


