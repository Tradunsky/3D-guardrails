"""LLM integration for asset trust analysis."""

from __future__ import annotations

import base64
import json
import logging
from typing import List

from openai import OpenAI

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail
from dddguardrails.schemas import RiskFinding, CATEGORIES, RiskCategory


log = logging.getLogger("dddguardrails.llm")


class OpenAIGuardrail(Guardrail):
    """Minimal client wrapper for the OpenAI Responses API."""

    def __init__(self, api_key: str = settings.openai_api_key):
        self._client = OpenAI(api_key=api_key)
        self._default_model = settings.openai_model

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
        """Classify a single screenshot."""
        cat_names = {c.name.lower() for c in risk_categories}
        
        categories_text = "\n".join(f"- {c.name.lower()}: {c.description}" for c in risk_categories)
        instructions = (
            "You are a trust and safety analyst. Review this render of the 3D asset "
            f"named '{file_name}' ({file_format}) and flag any category that could undermine "
            "user trust. Only use the allowed categories and respond with a JSON object containing a 'findings' array."
            f"Allowed categories:\n"
            f"{categories_text}"
            "\n\nSeverity must be one of: none, low, medium, high."
            "\n If a category is not present, omit it from the list."
        )

        model_to_use = model or self._default_model
        log.info(
            "classifying view %d | model=%s file=%s",
            view_number,
            model_to_use,
            file_name,
        )

        b64 = base64.b64encode(screenshot).decode("ascii")
        content = [
            {
                "type": "input_text",
                "text": instructions,
            },
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        ]

        response = self._client.responses.create(
            model=model_to_use,
            input=[{"role": "user", "content": content}],
            instructions=instructions,
            text={
                "format": {
                    "type": "json_schema",
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
                }
            },
        )
        output_text = response.output[0].content[0].text if response.output else ""
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - runtime guard.
            raise RuntimeError("LLM returned an unreadable payload.") from exc
        
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
