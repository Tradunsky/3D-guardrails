"""Pydantic schemas for the FastAPI surface."""

from __future__ import annotations

from dddguardrails.guardrail import RiskFinding

from pydantic import BaseModel


class ScanResponse(BaseModel):
    file_name: str
    file_format: str
    findings: list[RiskFinding]
    metadata: dict[str, str | int | float | dict[str, str | int | float]]
    
