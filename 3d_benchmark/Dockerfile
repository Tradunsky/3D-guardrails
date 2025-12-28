FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-dev \
    libegl1-mesa-dev \
    libgl1-mesa-dri \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgl1 \
    build-essential \
    xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/runtime-root && chmod 0700 /tmp/runtime-root

WORKDIR /bench
COPY 3d_benchmark/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 3d_benchmark/benchmark.py .
COPY tests/data /tests/data

# Force EGL/Vulkan headless mode
ENV PYOPENGL_PLATFORM=egl
ENV WGPU_BACKEND_TYPE=Vulkan
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV LIBGL_ALWAYS_SOFTWARE=1

COPY 3d_benchmark/entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["/bin/bash", "./entrypoint.sh"]