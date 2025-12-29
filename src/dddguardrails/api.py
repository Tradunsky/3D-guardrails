"""FastAPI app exposing the 3D guardrail endpoint."""

from __future__ import annotations

import logging
import os
import time
import tempfile
import json
from contextlib import asynccontextmanager
from pathlib import Path

import ollama
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
from google.genai.errors import ClientError
from pydantic import Json

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail
from dddguardrails.guardrails.gemini_llm import GeminiGuardrail
from dddguardrails.guardrails.ollama_llm import OllamaGuardrail
from dddguardrails.guardrails.openai_llm import OpenAIGuardrail
from dddguardrails.logger_config import configure_logging
from dddguardrails.rendering import (
    AssetProcessingError,
    render_views_generator,    
)

from dddguardrails.schemas import RiskCategory, CATEGORIES, ScanResponse

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
    title="3D Guardrails API",
    version="0.1.0",
    root_path="/v1/guardrails",
    lifespan=lifespan,
    tags=["3d-guardrails"],
)

_guardrail_classes: dict[str, type[Guardrail]] = {
    "openai": OpenAIGuardrail,
    "gemini": GeminiGuardrail,
    "ollama": OllamaGuardrail,
}

_guardrail_cache: dict[str, Guardrail] = {}


def _get_guardrail(provider: str = "gemini", base_url: str | None = None) -> Guardrail:    
    cache_key = f"{provider}_{base_url}"
    if cache_key in _guardrail_cache:
        return _guardrail_cache[cache_key]
    
    cls = _guardrail_classes.get(provider)
    if not cls:
        supported = ", ".join(_guardrail_classes.keys())
        raise ValueError(
            f"Unsupported LLM provider: {provider}. Supported: {supported}"
        )
    guardrail = cls(base_url=base_url)
    _guardrail_cache[cache_key] = guardrail
    return guardrail


def _normalize_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower().lstrip(".")


@app.exception_handler(AssetProcessingError)
def asset_processing_error(request: Request, exc: AssetProcessingError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=str(exc)
    )


@app.exception_handler(ClientError)
def gemini_client_error(request: Request, exc: ClientError):
    return JSONResponse(status_code=exc.code, content=exc.response.json())


@app.exception_handler(ollama.ResponseError)
def ollama_response_error(request: Request, exc: ollama.ResponseError):
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": f"Ollama API error: {str(exc)}"},
    )


@app.post(
    "/scan",
    summary="Scan 3D asset for trust and safety risks",
    response_model=ScanResponse,
    status_code=status.HTTP_200_OK,
)
async def scan_asset(
    file: UploadFile = File(..., description="3D asset in GLB/FBX/OBJ/STL/PLY format"),
    llm_provider: str = Form(
        "ollama",
        description="LLM provider to use for analysis (openai, gemini, ollama)",
    ),
    model: str | None = Form(
        "qwen3-vl:235b-cloud",
        description="Specific model to use (optional, uses default if not specified)",
    ),
    resolution_width: int = Form(
        settings.screenshot_resolution[0], description="Resolution width for rendering"
    ),
    resolution_height: int = Form(
        settings.screenshot_resolution[1], description="Resolution height for rendering"
    ),
    risk_categories: list[RiskCategory] | None = Form(
        None, description="JSON string of RiskCategory objects"
    ),
) -> ScanResponse:
    start_time = time.perf_counter()
    extension = _normalize_extension(file.filename)
    file_name = file.filename or "asset"
    if extension not in settings.supported_formats:
        supported_formats_csv = ", ".join(settings.supported_formats)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Supported: {supported_formats_csv}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty."
        )
    log.info("scan start | name=%s ext=%s bytes=%d", file_name, extension, len(contents))
    
    if risk_categories is None:
        risk_categories = CATEGORIES    

    resolution = (resolution_width, resolution_height) or settings.screenshot_resolution

    guard = _get_guardrail(llm_provider)
    findings = []
    views_evaluated = 0
    
    llm_total_ms = 0.0
    rendering_total_ms = 0.0
    rendering_start_timer = time.perf_counter()    

    for idx, screenshot in enumerate(render_views_generator(contents, extension, resolution), start=1):
        views_evaluated = idx
        rendering_total_ms += (time.perf_counter() - rendering_start_timer) * 1000
            
        llm_step_start = time.perf_counter()
        view_findings = guard.classify(
            screenshot=screenshot,
            view_number=idx,
            file_name=file_name,
            file_format=extension,
            risk_categories=risk_categories,
            model=model,            
        )
        llm_total_ms += (time.perf_counter() - llm_step_start) * 1000
        rendering_start_timer = time.perf_counter()

        # Dump screenshot if requested (for debugging)
        if os.getenv("DDDG_DUMP_SCREENSHOTS") == "True":
            with open(f"screenshot-{idx}.png", "wb") as f:
                f.write(screenshot)

        if view_findings:
            findings = view_findings
            break

    end_time = time.perf_counter()
    total_ms = (end_time - start_time) * 1000
    
    log.info("scan done | findings=%d views=%d", len(findings), views_evaluated)

    return ScanResponse(
        file_name=file_name,
        file_format=extension,
        findings=findings,
        metadata={
            "views_evaluated": views_evaluated, 
            "model": model, 
            "llm_provider": llm_provider,
            "latency": {
                "total_ms": round(total_ms, 2),
                "llm_ms": round(llm_total_ms, 2),
                "rendering_ms": round(rendering_total_ms, 2),
            }
        },
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
