"""LLM integration for asset trust analysis."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field
from typing import Literal

from dddguardrails.schemas import RiskCategory, RiskFinding

class Guardrail:
    def classify(
        self,
        *,
        screenshot: bytes,
        view_number: int,
        file_name: str,
        file_format: str,
        risk_categories: List[RiskCategory],
        model: str | None = None,
    ) -> List[RiskFinding]:
        pass
