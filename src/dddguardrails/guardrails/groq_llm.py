"""LLM integration for Groq AI provider."""

from __future__ import annotations

from typing import List

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail
from dddguardrails.guardrails.openai_llm import OpenAIGuardrail
from dddguardrails.schemas import RiskFinding, RiskCategory


class GroqGuardrail(Guardrail):
    """Client wrapper for the Groq API (OpenAI-compatible)."""

    def __init__(self, api_key: str = settings.groq_api_key, base_url: str | None = None):
        # Groq's default base URL if not provided
        self._guardrail = OpenAIGuardrail(api_key, base_url or settings.groq_base_url)
        self._default_model = settings.groq_model

    async def classify(
        self,
        *,
        screenshot: bytes,
        view_number: int,
        file_name: str,
        file_format: str,
        risk_categories: List[RiskCategory],
        model: str | None = None,
    ) -> List[RiskFinding]:
        return await self._guardrail.classify(
            screenshot=screenshot,
            view_number=view_number,
            file_name=file_name,
            file_format=file_format,
            risk_categories=risk_categories,
            model=model,
        )
