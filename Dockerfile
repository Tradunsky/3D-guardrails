# ============================================
# Stage 1: Build - Install dependencies with uv
# ============================================
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Configure uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy project files needed for dependency installation
COPY pyproject.toml uv.lock README.md ./

# Export dependencies (excluding project itself) and install to system Python
# Reinstall vtk-egl last to overwrite conflicting vtk shared libraries
RUN --mount=type=cache,target=/root/.cache \
    uv export --frozen --no-hashes --no-dev --no-emit-project | \
    uv pip install --system -r - && \
    uv pip install --system --reinstall vtk-egl

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for Headless rendering (EGL/OSMesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the source code
COPY src ./src
COPY tests/data ./tests/data
COPY demo.py .
COPY entrypoint.sh .

# Expose ports
# 8000 for API, 7860 for Gradio
EXPOSE 8000 7860

RUN chmod +x entrypoint.sh

# Default command uses entrypoint logic
CMD ["./entrypoint.sh"]
