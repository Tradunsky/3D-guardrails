"""LLM integration for asset trust analysis."""

from __future__ import annotations

import base64
import json
import logging
from typing import List

from openai import OpenAI, api_key

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail, RiskFinding, CATEGORIES


log = logging.getLogger("dddguardrails.llm")


class OpenAIGuardrail(Guardrail):
    """Minimal client wrapper for the OpenAI Responses API."""

    def __init__(self, api_key: str=settings.openai_api_key) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM classification.")
        self._client = OpenAI(api_key=api_key)
        self._default_model = settings.openai_model

    def classify(self, *, screenshots: List[bytes], file_name: str, file_format: str, model: str | None = None) -> List[RiskFinding]:
        """Send screenshots to the LLM one by one until first violation is found."""
        categories = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
        instructions = (
            "You are a trust and safety analyst. Review this render of the 3D asset "
            f"named '{file_name}' ({file_format}) and flag any category that could undermine "
            "user trust. Only use the allowed categories and respond with a JSON object containing a 'findings' array."
            f"Allowed categories:\n"
            f"{categories}"
            "\n\nSeverity must be one of: none, low, medium, high."
            "\n If a category is not present, omit it from the list."
        )

        model_to_use = model or self._default_model
        log.info("classifying | model=%s views=%d file=%s", model_to_use, len(screenshots), file_name)

        # Process screenshots one by one until first violation is found
        for idx, image_bytes in enumerate(screenshots, start=1):
            b64 = base64.b64encode(image_bytes).decode("ascii")
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

            log.debug("checking screenshot %d/%d for file=%s", idx, len(screenshots), file_name)
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

            # Early exit if violations found
            if findings_list:
                log.info("found violations in screenshot %d/%d for file=%s", idx, len(screenshots), file_name)
                normalized: List[RiskFinding] = []
                for finding in findings_list:
                    category = finding.get("category", "").strip().lower()
                    if category not in CATEGORIES:
                        continue
                    normalized.append(
                        RiskFinding(
                            category=category,
                            severity=finding.get("severity", "none").lower(),
                            rationale=finding.get("rationale", ""),
                            view_number=idx,
                        )
                    )
                return normalized

        # No violations found in any screenshot
        log.info("no violations found for file=%s after checking %d screenshots", file_name, len(screenshots))
        return []

