---
title: 3D guardrails
short_description: 3D content you can trust
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
sdk_version: 	6.1.0
pinned: true
license: apache-2.0
app_file: demo.py
tags:
  - 3D guardrails
  - 3D 
---

[![Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg.svg)](https://huggingface.co/spaces/tradunsky/3D-guardrails)

## 3D Guardrails API

Run the FastAPI server to scan 3D assets for trust-and-safety risks.

![Example](./.docs/images/weapons.png)

### 3D Rendering Benchmark

The most expensive part is not 3D rendering, but the LLM analysis. Here is a benchmark of the 3D rendering part running on a CPU:

![3D rendering benchmark](./3d_benchmark/benchmark_results.png)


### Prerequisites

- Python 3.11
- For OpenAI: `OPENAI_API_KEY` exported in your environment
- For Gemini: `GEMINI_API_KEY` exported in your environment
- For Ollama: Local Ollama server running (default: http://localhost:11434)

### Install

```bash
uv sync
```

### Start the server

```bash
uv run fastapi dev -e dddguardrails.api:app
```

### Gradio Demo

Launch an interactive web interface for testing the 3D guardrails:

```bash
uv run python -m dddguardrails.demo
```

The demo will be available at http://localhost:7860 and provides:
- File upload interface for 3D models
- LLM provider and model selection
- Real-time safety analysis results
- Clear display of risk findings and severity levels

### Docker

Run the application using Docker for easy deployment with off-screen rendering support.

#### Build the Image

```bash
docker build -t dddguardrails .
```

#### Run Gradio Demo (Default)

```bash
# Using docker run
docker run -p 7860:7860 -e APP_MODE=gradio dddguardrails

# Using docker-compose (recommended)
docker-compose -p local up -d --no-deps --build app
```

Access the Gradio interface at http://localhost:7860

#### Run FastAPI Service

```bash
# Using docker run
docker run -p 8000:8000 -e APP_MODE=fastapi dddguardrails

# Using docker-compose (recommended)
docker-compose up fastapi
```

Access the FastAPI docs at http://localhost:8000/v1/guardrails/docs

#### Environment Variables

- `APP_MODE`: Set to `gradio` (default) or `fastapi`
- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `GEMINI_API_KEY`: Your Google Gemini API key (optional)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

### Testing Guidelines

**Avoid Change-Detector Tests**: This codebase follows best practices for unit testing. Mock tests that verify implementation details rather than behavior are harmful because they break during refactoring and don't provide real confidence in functionality.

For guidance on writing effective tests, see: [Testing on the Toilet: Change-Detector Tests Considered Harmful](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html)

### Endpoint

- `POST /v1/guardrails/scan`
  - Request format: `multipart/form-data`
  - Form Fields:
    - `file`: The 3D asset file. Accepts `.glb`, `.gltf`, `.fbx`, `.obj`, `.stl`, `.ply`.
    - `llm_provider`: (optional, default: "ollama") LLM provider to use ("openai", "gemini", "ollama").
    - `model`: (optional) Specific model to use (e.g., "gpt-4o", "gemini-3-flash-preview", "qwen3-vl:235b-cloud").
    - `resolution_width`: (optional) Width of screenshots for rendering.
    - `resolution_height`: (optional) Height of screenshots for rendering.
    - `risk_categories`: (optional) JSON array of custom `RiskCategory` objects (name and description).
  - Returns: `ScanResponse` containing detected categories with severity and rationale.

### Roadmap

- [x] OpenAI support
- [x] Gemini support
- [x] Ollama support
- [x] User specified provider and models support
- [x] Early exit on the first violation (save tokens)
- [x] Docker/docker-compose support with headless rendering
- [x] Improve rendering performance
- [x] Benchmark
- [ ] Evals
- [x] Gradio demo
- [ ] AWS Bedrock support
- [ ] External AI gateways with OpenAI API compatibility
- [x] Configurable Categories: Allow users to define custom risk categories in their scan requests instead of being limited to the hardcoded ones (weapons, nudity, self-harm, etc.)
- [ ] Add multi-modal (image, text, sound, video?)
- [ ] Batch API
- [ ] Async API
- [ ] Streaming API
- [ ] Reading content from URLs (presigned URL, internet content, ect)
- [ ] Backwards compatible API with OpenAI and AWS Bedrock Guardrails
- [ ] Configurable early exit (based on severity of findings?)

