"""LLM integration for asset trust analysis."""

from __future__ import annotations

import base64
import json
import logging
from typing import List

from openai import AsyncOpenAI

from dddguardrails.guardrail import Guardrail
from dddguardrails.schemas import RiskFinding, RiskCategory


log = logging.getLogger("dddguardrails.llm")


class OpenAIGuardrail(Guardrail):
    """Minimal client wrapper for the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def classify(
        self,
        *,
        screenshot: bytes,
        view_number: int,
        file_name: str,
        file_format: str,
        risk_categories: List[RiskCategory],
        model: str,
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
            "\n Do not wrap output into markdown like ```json```."
        )

        log.info(
            "classifying view %d | model=%s file=%s",
            view_number,
            model,
            file_name,
        )

        b64 = base64.b64encode(screenshot).decode("ascii")
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            },
        ]

        response = await self._client.chat.completions.create(
            model=(model),
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "risk_findings",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "findings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "severity": {"type": "string"},
                                        "rationale": {"type": "string"},
                                    },
                                    "required": [
                                        "category",
                                        "severity",
                                        "rationale",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["findings"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        output_text = response.choices[0].message.content or ""
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - runtime guard.
            raise RuntimeError(f"LLM returned an unreadable payload: {output_text}") from exc
        
        findings_list = (
            parsed.get("findings", []) if isinstance(parsed, dict) else []
        )

        if findings_list:
            log.info(
                "found violations in screenshot %d for file=%s",
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
