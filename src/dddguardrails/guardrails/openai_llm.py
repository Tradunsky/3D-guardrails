"""LLM integration for asset trust analysis."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List

from openai import OpenAI, api_key

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail


log = logging.getLogger("dddguardrails.llm")


class OpenAIGuardrail(Guardrail):
    """Minimal client wrapper for the OpenAI Responses API."""

    def __init__(self, api_key: str=settings.openai_api_key, model: str=settings.openai_model) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM classification.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def classify(self, *, screenshots: List[bytes], file_name: str, file_format: str) -> List[Dict[str, Any]]:
        """Send screenshots to the LLM and return normalized findings."""

        instructions = (
            "You are a trust and safety analyst. Review each render of the 3D asset "
            f"named '{file_name}' ({file_format}) and flag any category that could undermine "
            "user trust. Only use the allowed categories and respond with a JSON object containing a 'findings' array."
        )
        content = [
            {
                "type": "input_text",
                "text": (
                    f"Allowed categories:\n"
                    + "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
                    + "\n\nSeverity must be one of: none, low, medium, high."
                    " If a category is not present, omit it from the list."
                ),
            }
        ]
        for idx, image_bytes in enumerate(screenshots, start=1):
            b64 = base64.b64encode(image_bytes).decode("ascii")
            content.extend(
                [
                    {"type": "input_text", "text": f"View {idx}"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                ]
            )

        log.info("classifying | model=%s views=%d file=%s", self._model, len(screenshots), file_name)
        response = self._client.responses.create(
            model=self._model,
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
                                    "required": ["category", "severity", "rationale"],
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
        # Extract the findings array from the response object
        findings_list = parsed.get("findings", []) if isinstance(parsed, dict) else []
        normalized: List[Dict[str, Any]] = []
        for finding in findings_list:
            category = finding.get("category", "").strip().lower()
            if category not in CATEGORIES:
                continue
            normalized.append(
                {
                    "category": category,
                    "severity": finding.get("severity", "none").lower(),
                    "rationale": finding.get("rationale", ""),
                }
            )
        return normalized

