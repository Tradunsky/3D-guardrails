# 3D Rendering Benchmark

This directory contains a comprehensive benchmark suite for evaluating the performance of various 3D rendering libraries in Python. It measures the latency of loading and rendering 3D assets (GLB format) across multiple frameworks.

## Supported Frameworks

The benchmark currently evaluates the following libraries:
- **Pyrender** (OpenGL/EGL)
- **PyGfx** (WGPU)
- **PyVista** (VTK)
- **Open3D**
- **MeshLib** (Loading & Stats only*)

## Prerequisites

The benchmark is designed to run in an isolated environment to avoid dependency conflicts. 

### 1. Setup Virtual Environment
It is recommended to use the provided `requirements.txt` to set up a clean environment.

```bash
cd 3d_benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** Headless rendering requires EGL support. Ensure your system has the necessary drivers installed (e.g., `libegl1-mesa-dev`).

## Running the Benchmark

Run the benchmark script directly using Python. You can specify the number of iterations per mesh to get more statistically significant results.

```bash
# Run with default settings (1 iteration)
python benchmark.py

# Run with multiple iterations for better accuracy
python benchmark.py --iterations 5
```

### Debug Mode
To inspect the rendered output, you can run with `DEBUG=1`. This will save the rendered frames as PNG files in the current directory.

```bash
DEBUG=1 python benchmark.py
```
### Running with Docker

You can run the benchmark in a completely isolated environment using Docker. This ensures all dependencies (including system libraries) are correctly installed.

```bash
# Build and run with results saved to ./3d_benchmark
docker build -f 3d_benchmark/Dockerfile -t benchmark . && docker run --rm -v $(pwd)/3d_benchmark:/bench benchmark
```

This command will:
1. Build the Docker image.
2. Run the benchmark inside the container.
3. Save the results (`benchmark_results.csv`, `benchmark_results.png`) back to your local `3d_benchmark` directory.

## Reading the Results

The benchmark generates three types of output:

### 1. Console Output
Real-time progress and summary statistics (median latency) are printed to the console.

### 2. CSV Report (`benchmark_results.csv`)
A raw data file containing detailed timing for every run.
- **Mesh**: The name of the 3D asset.
- **Framework**: The library used.
- **Iteration**: The iteration number.
- **Latency_ms**: Time taken in milliseconds.

### 3. Visualizations
A chart is automatically generated after a successful run:
- **`benchmark_results.png`**: A grouped bar chart showing the median latency for each mesh across all frameworks.

![Benchmark Results](benchmark_results.png)

## Directory Structure

- `benchmark.py`: Main entry point script.
- `requirements.txt`: Python dependencies specific to this benchmark.
- `benchmark_results.csv`: Generated results file.
- `*.png`: Generated charts and debug screenshots.
