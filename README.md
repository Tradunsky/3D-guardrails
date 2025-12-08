## 3D Guardrails API

Run the FastAPI server to scan 3D assets for trust-and-safety risks.

### Prerequisites

- Python 3.14+
- `OPENAI_API_KEY` exported in your environment
- `GEMINI_API_KEY` exported in your environment

### Install

```bash
uv sync
```

### Start the server

```bash
uv run fastapi dev 3dguardrails/api.py
```

### Endpoint

- `POST /v1/guardrails/scan`
  - Body: multipart form with file field named `file`
  - Accepts `.glb`, `.gltf`, `.fbx`, `.obj`, `.stl`, `.ply`
  - Returns detected categories with severity and rationale after multi-view screenshotting plus multimodal LLM analysis.

