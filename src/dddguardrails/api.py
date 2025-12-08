"""FastAPI app exposing the 3D guardrail endpoint."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import uvicorn
from google.genai.errors import ClientError

from dddguardrails import guardrail
from dddguardrails.config import settings
from dddguardrails.guardrails.gemini_llm import GeminiGuardrail
from dddguardrails.logger_config import configure_logging
from dddguardrails.guardrail import Guardrail
from dddguardrails.guardrails.openai_llm import OpenAIGuardrail
from dddguardrails.rendering import AssetProcessingError, RenderConfig, generate_multiview_images, load_mesh
from dddguardrails.schemas import RiskFinding, ScanResponse


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

_guardrail: Guardrail | None = None


def _get_guardrail() -> Guardrail:
    global _guardrail
    if _guardrail is None:
        # _guardrail = OpenAIGuardrail()
        _guardrail = GeminiGuardrail()
    return _guardrail


def _normalize_extension(filename: str | None) -> str:
    if not filename:
        return ""
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext

@app.exception_handler(AssetProcessingError)
def asset_processing_error(request: Request, exc: AssetProcessingError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=str(exc))

@app.exception_handler(ClientError)
def gemini_client_error(request: Request, exc: ClientError):    
    return JSONResponse(status_code=exc.code, content=exc.response.json())


@app.post(
    "/scan",
    summary="Scan 3D asset for trust and safety risks",
    response_model=ScanResponse,
    status_code=status.HTTP_200_OK,
)
async def scan_asset(file: UploadFile = File(..., description="3D asset in GLB/FBX/OBJ/STL/PLY format")) -> ScanResponse:
    extension = _normalize_extension(file.filename)
    if extension not in settings.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Supported: {', '.join(settings.supported_formats)}",
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

        mesh = load_mesh(str(tmp_path))
        render_config = RenderConfig(
            resolution=settings.screenshot_resolution,
            distance=settings.camera_distance,
            view_angles=settings.multi_view_angles,
        )
        screenshots = generate_multiview_images(mesh, render_config)
        log.debug("rendered screenshots | count=%d res=%s", len(screenshots), render_config.resolution)            

    guard = _get_guardrail()
    findings_raw = guard.classify(screenshots=screenshots, file_name=file.filename or "asset", file_format=extension)
    findings: List[RiskFinding] = [RiskFinding(**finding) for finding in findings_raw]
    log.info("scan done | findings=%d", len(findings))
    return ScanResponse(
        file_name=file.filename or "asset",
        file_format=extension,
        findings=findings,
        metadata={"views_evaluated": len(screenshots)},
    )



if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

