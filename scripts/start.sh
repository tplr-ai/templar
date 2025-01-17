#!/bin/bash
# Stop any existing processes
pm2 delete all

# Start all applications
pm2 start ecosystem.config.js


