## 3D Guardrails API

Run the FastAPI server to scan 3D assets for trust-and-safety risks.

### Prerequisites

- Python 3.14+
- For OpenAI: `OPENAI_API_KEY` exported in your environment
- For Gemini: `GEMINI_API_KEY` exported in your environment
- For Ollama: Local Ollama server running (default: http://localhost:11434)

### Install

```bash
uv sync
```

### Start the server

```bash
uv run fastapi dev 3dguardrails/api.py
```

### Testing Guidelines

**Avoid Change-Detector Tests**: This codebase follows best practices for unit testing. Mock tests that verify implementation details rather than behavior are harmful because they break during refactoring and don't provide real confidence in functionality.

For guidance on writing effective tests, see: [Testing on the Toilet: Change-Detector Tests Considered Harmful](https://testing.googleblog.com/2015/01/testing-on-toilet-change-detector-tests.html)

### Endpoint

- `POST /v1/guardrails/scan`
  - Body: multipart form with file field named `file`
  - Query parameter: `llm_provider` (optional, default: "gemini")
    - Supported values: "openai", "gemini", "ollama"
  - Query parameter: `model` (optional, uses provider default if not specified)
    - Examples: "gpt-4o", "gemini-pro", "llama3.2-vision"
  - Accepts `.glb`, `.gltf`, `.fbx`, `.obj`, `.stl`, `.ply`
  - Returns detected categories with severity and rationale after multi-view screenshotting plus multimodal LLM analysis.

