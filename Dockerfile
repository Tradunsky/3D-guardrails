FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Headless Polyscope (EGL/OSMesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libosmesa6 \
    libglu1-mesa \
    xvfb \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files needed for dependency installation
COPY pyproject.toml uv.lock README.md ./

# Install dependencies into the system environment
# We export from the lockfile to ensure reproducible builds
RUN user_requirements=$(uv export --frozen --no-emit-project --format=requirements-txt) && \
    echo "$user_requirements" > requirements.txt && \
    uv pip install --system -r requirements.txt

# Copy the source code
COPY . .

# Install the project itself into the system environment
RUN uv pip install --system --no-deps -e .

# Expose ports
# 8000 for API, 7860 for Gradio
EXPOSE 8000 7860

RUN chmod +x entrypoint.sh

# Default command uses entrypoint logic
CMD ["./entrypoint.sh"]
