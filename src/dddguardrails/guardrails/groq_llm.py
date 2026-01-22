"""LLM integration for Groq AI provider."""

from __future__ import annotations

import base64
import json
import logging
from typing import List

from openai import AsyncOpenAI

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail
from dddguardrails.schemas import RiskFinding, RiskCategory


log = logging.getLogger("dddguardrails.llm")


class GroqGuardrail(Guardrail):
    """Client wrapper for the Groq API (OpenAI-compatible)."""

    def __init__(self, api_key: str = settings.groq_api_key, base_url: str | None = None):
        # Groq's default base URL if not provided
        base_url = base_url or "https://api.groq.com/openai/v1"
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
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
        """Classify a single screenshot."""
        cat_names = {c.name.lower() for c in risk_categories}
        
        categories_text = "\n".join(f"- {c.name.lower()}: {c.description}" for c in risk_categories)
        view_text = f"view #{view_number}" if view_number > 0 else "multiple views (tiled)"
        instructions = (
            f"You are a trust and safety analyst. Review these {view_text} of the 3D asset "
            f"named '{file_name}' ({file_format}) and flag any category that could undermine "
            "user trust. Only use the allowed categories and respond with a JSON object containing a 'findings' array."
            f"Allowed categories:\n"
            f"{categories_text}"
            "\n\nSeverity must be one of: none, low, medium, high."
            "\n If a category is not present, omit it from the list."
            "\nYour response must be a valid JSON object."
        )

        model_to_use = model or self._default_model
        log.info(
            "classifying view %d | model=%s file=%s via Groq",
            view_number,
            model_to_use,
            file_name,
        )

        b64 = base64.b64encode(screenshot).decode("ascii")
        
        response = await self._client.chat.completions.create(
            model=model_to_use,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
        )

        output_text = response.choices[0].message.content or ""
        parsed = json.loads(output_text)
        
        findings_list = (
            parsed.get("findings", []) if isinstance(parsed, dict) else []
        )

        if findings_list:
            log.info(
                "found %d violations in screenshot %d for file=%s via Groq",
                len(findings_list),
                view_number,
                file_name,
            )
            normalized: List[RiskFinding] = []
            for finding in findings_list:
                category = finding.get("category", "").strip().lower()
                if category not in cat_names:
                    continue
                normalized.append(
                    RiskFinding(
                        category=category,
                        severity=finding.get("severity", "none").lower(),
                        rationale=finding.get("rationale", ""),
                        view_number=view_number,
                    )
                )
            return normalized

        return []
