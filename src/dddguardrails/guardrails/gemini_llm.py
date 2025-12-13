from logging import getLogger
from typing import Any, Dict, List
from google import genai
from google.genai import types
from dddguardrails.guardrail import Guardrail, RiskFinding, CATEGORIES
from dddguardrails.config import settings

log = getLogger()

class GeminiGuardrail(Guardrail):
    def __init__(self, api_key:str=settings.gemini_api_key) -> None:
        self._client: genai.Client = genai.Client(
            api_key=api_key
        )
        self._default_model = settings.gemini_model

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

        response_format = {
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
        }

        model_to_use = model or self._default_model
        log.info("classifying | model=%s views=%d file=%s", model_to_use, len(screenshots), file_name)

        # Process screenshots one by one until first violation is found
        for idx, image_bytes in enumerate(screenshots, start=1):
            content = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=instructions),
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png")
                    ]
                )
            ]

            log.debug("checking screenshot %d/%d for file=%s", idx, len(screenshots), file_name)
            response = self._client.models.generate_content(
                model=model_to_use,
                contents=content,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_json_schema=response_format
                ),
            )

            findings_list = response.parsed.get("findings", []) if isinstance(response.parsed, dict) else []

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

