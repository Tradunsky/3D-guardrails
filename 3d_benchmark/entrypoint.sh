#!/bin/bash
set -e

# Start Xvfb in background
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
export DISPLAY=:99

# Wait for Xvfb to start
sleep 2

# Run benchmark
exec python3 -u benchmark.py "$@"
