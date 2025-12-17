FROM python:3.12-slim

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Headless Polyscope (EGL/OSMesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libosmesa6 \
    xvfb \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install Python dependencies
# Standard install as wheels should work with system libs installed
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ .
COPY . .

# Expose ports
# 8000 for API, 7860 for Gradio
EXPOSE 8000 7860

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Default command uses entrypoint logic
CMD ["./entrypoint.sh"]
