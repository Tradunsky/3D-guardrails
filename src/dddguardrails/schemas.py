"""Pydantic schemas for the FastAPI surface."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class RiskCategory(BaseModel):
    """Configuration for a risk category."""
    name: str = Field(..., description="Machine-friendly name of the category")
    description: str = Field(..., description="Human-readable description of what to look for")


CATEGORIES = [
    RiskCategory(name="weapons", description="Explosives, firearms, or realistic weapon replicas that could enable violence."),
    RiskCategory(name="nudity", description="Explicit nudity, sexual content, or fetish imagery."),
    RiskCategory(name="self_harm", description="Content encouraging suicide or self-harm."),
    RiskCategory(name="extremism", description="Symbols or content tied to extremist, terroristic, or hate organizations."),
    RiskCategory(name="hate_symbols", description="Imagery targeting protected classes with hate speech or symbols."),
    RiskCategory(name="misleading", description="Fraudulent or deceptive items (fake credentials, scam artifacts)."),
    RiskCategory(name="prompt_injection", description="An attempt to inject model instructions via 3D asset."),
]


class RiskFinding(BaseModel):
    """Result model for guardrail classification."""

    category: str = Field(..., description="The classification category")
    severity: Literal["none", "low", "medium", "high"] = Field(..., description="Severity level of the classification")
    rationale: str = Field(..., description="Reason for the classification")
    view_number: int = Field(..., description="The view/screenshot number where this finding was detected")


class ScanResponse(BaseModel):
    file_name: str
    file_format: str
    findings: list[RiskFinding]
    metadata: dict[str, str | int | float | dict[str, str | int | float]]
    
