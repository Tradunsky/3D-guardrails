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

### Roadmap

- [x] OpenAI support
- [x] Gemini support
- [x] Ollama support
- [x] User specified provider and models support
- [x] Early exit on first violation (safe tokens)
- [ ] Benchmark
- [x] Gradio demo
- [ ] AWS Bedrock support
- [ ] External AI gateways with OpenAI API compatibility
- [ ] Configurable Categories: Allow users to define custom risk categories in their scan requests instead of being limited to the hardcoded ones (weapons, nudity, self-harm, etc.)
- [ ] Add multi-modal (image, text, sound, video?)
- [ ] Batch API
- [ ] Async API
- [ ] Streaming API
- [ ] Reading content from URLs (presigned URL, internet content, ect)
- [ ] Backwards compatible API with OpenAI and AWS Bedrock Guardrails
- [ ] Configurable early exit (based on severity of findings?)

