# Multi-stage build for optimized image size
# FROM python:3.14-slim as builder
FROM python:3.14-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system Mesa libraries with EGL and OSMesa support
RUN apt-get update && apt-get install -y \
    libegl-mesa0 \
    libegl1-mesa-dev \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libglapi-mesa \
    libgbm1 \
    libosmesa6 \
    libosmesa6-dev \
    libglut-dev \
    mesa-common-dev \
    mesa-utils \
    wget \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install standard PyOpenGL (newer versions support OSMesa)
RUN pip install --upgrade PyOpenGL

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# # Runtime stage
# FROM python:3.14-slim as runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libegl-mesa0 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libglapi-mesa \
    libgbm1 \
    libosmesa6 \
    libglut3.12 \
    libglu1-mesa \
    libglu1-mesa-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/* && \
    # Create symlink for GLU library that pyglet expects
    ln -sf /usr/lib/x86_64-linux-gnu/libGLU.so.1 /usr/lib/x86_64-linux-gnu/libGLU.so && \
    # Ensure EGL libraries are properly linked
    ldconfig

# # Copy virtual environment from builder stage
# COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables for headless EGL rendering
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=surfaceless
ENV PYTHONPATH=/app/src

# Set up the application directory
WORKDIR /app

# Copy the application code
COPY src/ ./src/
COPY demo.py .
COPY README.md .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose ports (Gradio uses 7860, FastAPI uses 8000)
EXPOSE 7860 8000

# Default to Gradio demo, can be overridden with environment variable
ENV APP_MODE=gradio
CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"fastapi\" ]; then python -m uvicorn src.dddguardrails.api:app --host 0.0.0.0 --port 8000; else python demo.py; fi"]