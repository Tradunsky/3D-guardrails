"""Pydantic schemas for the FastAPI surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RiskFinding(BaseModel):
    category: str = Field(..., description="Trust and safety category name.")
    severity: Literal["none", "low", "medium", "high"]
    rationale: str = Field(..., description="Reason for the classification.")


class ScanResponse(BaseModel):
    file_name: str
    file_format: str
    findings: list[RiskFinding]
    metadata: dict[str, str | int | float]

