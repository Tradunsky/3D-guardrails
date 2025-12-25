"""LLM integration for asset trust analysis."""

from __future__ import annotations

import logging
from typing import List

from pydantic import BaseModel, Field
from typing import Literal


log = logging.getLogger("dddguardrails.llm")

CATEGORIES = {
    "weapons": "Explosives, firearms, or realistic weapon replicas that could enable violence.",
    "nudity": "Explicit nudity, sexual content, or fetish imagery.",
    "self_harm": "Content encouraging suicide or self-harm.",
    "extremism": "Symbols or content tied to extremist, terroristic, or hate organizations.",
    "hate_symbols": "Imagery targeting protected classes with hate speech or symbols.",
    "misleading": "Fraudulent or deceptive items (fake credentials, scam artifacts).",
}


class RiskFinding(BaseModel):
    """Result model for guardrail classification."""

    category: str = Field(..., description="The classification category")
    severity: Literal["none", "low", "medium", "high"] = Field(..., description="Severity level of the classification")
    rationale: str = Field(..., description="Reason for the classification")
    view_number: int = Field(..., description="The view/screenshot number where this finding was detected")


class Guardrail:
    def classify(
        self,
        *,
        screenshot: bytes,
        view_number: int,
        file_name: str,
        file_format: str,
        model: str | None = None,
    ) -> List[RiskFinding]:
        pass
