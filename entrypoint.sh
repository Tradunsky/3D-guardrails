#!/bin/bash
set -e

# Default to Gradio if RUN_MODE is not set to 'api'
if [ "$RUN_MODE" = "api" ]; then
    echo "Starting FastAPI backend..."
    exec uvicorn dddguardrails.api:app --host 0.0.0.0 --port 8000
else
    echo "Starting Gradio demo..."
    # Ensure Gradio listens on all interfaces (Dockerfile already handles port exposition)
    exec python demo.py
fi
