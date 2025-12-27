# ============================================
# Single Stage: Build and Runtime combined
# ============================================
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies required for Headless rendering (EGL/OSMesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never


# Copy project files needed for dependency installation
COPY pyproject.toml uv.lock README.md ./

# Sync dependencies to system Python
RUN --mount=type=cache,target=/root/.cache \
    uv export --frozen --no-hashes --no-dev --no-emit-project | \
    uv pip install --system -r -

# Copy the source code
COPY src ./src
COPY tests/data ./tests/data
COPY demo.py .
COPY entrypoint.sh .

# Expose ports
# 8000 for API, 7860 for Gradio
EXPOSE 8000 7860

RUN chmod +x entrypoint.sh
ENV PYTHONPATH=/app/src:$PYTHONPATH


# Default command uses entrypoint logic
CMD ["./entrypoint.sh"]
