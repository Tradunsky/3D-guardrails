"""FastAPI app exposing the 3D guardrail endpoint."""

from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, File, HTTPException, UploadFile, status, Query
from fastapi.responses import JSONResponse
import uvicorn
from google.genai.errors import ClientError
import ollama


from dddguardrails.config import settings
from dddguardrails.guardrails.gemini_llm import GeminiGuardrail
from dddguardrails.guardrails.ollama_llm import OllamaGuardrail
from dddguardrails.logger_config import configure_logging
from dddguardrails.guardrail import Guardrail
from dddguardrails.guardrails.openai_llm import OpenAIGuardrail
from dddguardrails.rendering import AssetProcessingError, RenderConfig, generate_multiview_images, load_mesh
from dddguardrails.schemas import ScanResponse


configure_logging()
log = logging.getLogger("dddguardrails.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "API ready | formats=%s res=%s dist=%.1f views=%d",
        ",".join(settings.supported_formats),
        f"{settings.screenshot_resolution[0]}x{settings.screenshot_resolution[1]}",
        settings.camera_distance,
        len(settings.multi_view_angles),        
    )
    yield    


app = FastAPI(
    title="3D Guardrails API", version="0.1.0", root_path="/v1/guardrails", 
    lifespan=lifespan,
    tags=["3d-guardrails"]
)

_guardrails: dict[str, Guardrail] = {
    "openai": OpenAIGuardrail(),
    "gemini": GeminiGuardrail(),
    "ollama": OllamaGuardrail()
}


def _get_guardrail(provider: str = "gemini") -> Guardrail:
    guardrail = _guardrails.get(provider)
    if not guardrail:        
        supported_guardrails = ", ".join(_guardrails.keys())
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported: {supported_guardrails}")
    return guardrail


def _normalize_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")    

@app.exception_handler(AssetProcessingError)
def asset_processing_error(request: Request, exc: AssetProcessingError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=str(exc))

@app.exception_handler(ClientError)
def gemini_client_error(request: Request, exc: ClientError):
    return JSONResponse(status_code=exc.code, content=exc.response.json())

@app.exception_handler(ollama.ResponseError)
def ollama_response_error(request: Request, exc: ollama.ResponseError):
    return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"error": f"Ollama API error: {str(exc)}"})


@app.post(
    "/scan",
    summary="Scan 3D asset for trust and safety risks",
    response_model=ScanResponse,
    status_code=status.HTTP_200_OK,
)
async def scan_asset(
    file: UploadFile = File(..., description="3D asset in GLB/FBX/OBJ/STL/PLY format"),
    llm_provider: str = Query("ollama", description="LLM provider to use for analysis (openai, gemini, ollama)"),
    model: str | None = Query("qwen3-vl:235b-cloud", description="Specific model to use (optional, uses default if not specified)")
) -> ScanResponse:
    extension = _normalize_extension(file.filename)
    if extension not in settings.supported_formats:
        supported_formats_csv = ', '.join(settings.supported_formats)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Supported: {supported_formats_csv}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    log.info(
        "scan start | name=%s ext=%s bytes=%d",
        file.filename or "asset",
        extension,
        len(contents),
    )
    
    fd, temp_path = tempfile.mkstemp(suffix=f".{extension}")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(contents)
        tmp_path = Path(temp_path)

        mesh = load_mesh(str(tmp_path))
        render_config = RenderConfig(
            resolution=settings.screenshot_resolution,
            distance=settings.camera_distance,
            view_angles=settings.multi_view_angles,
        )
        screenshots = generate_multiview_images(mesh, render_config)
        log.debug("rendered screenshots | count=%d res=%s", len(screenshots), render_config.resolution)
    finally:
        os.unlink(temp_path)

    guard = _get_guardrail(llm_provider)
    findings = guard.classify(
        screenshots=screenshots,
        file_name=file.filename or "asset",
        file_format=extension,
        model=model
    )
    log.info("scan done | findings=%d", len(findings))

    # Calculate views_evaluated as the maximum view_number from findings,
    # or len(screenshots) if no findings (meaning all were evaluated)
    views_evaluated = max((finding.view_number for finding in findings), default=len(screenshots))

    return ScanResponse(
        file_name=file.filename or "asset",
        file_format=extension,
        findings=findings,
        metadata={"views_evaluated": views_evaluated},
    )



if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)