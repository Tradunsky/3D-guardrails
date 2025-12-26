#!/bin/bash
set -e

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3


# Default to Gradio if RUN_MODE is not set to 'api'
if [ "$RUN_MODE" = "api" ]; then
    echo "Starting FastAPI backend..."
    exec uvicorn dddguardrails.api:app --host 0.0.0.0 --port 8000
else
    echo "Starting Gradio demo..."
    # Ensure Gradio listens on all interfaces (Dockerfile already handles port exposition)
    exec python demo.py
fi
