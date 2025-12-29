from logging import getLogger
from typing import List
from google import genai
from google.genai import types
from dddguardrails.guardrail import Guardrail
from dddguardrails.schemas import RiskFinding, CATEGORIES, RiskCategory
from dddguardrails.config import settings

log = getLogger()


class GeminiGuardrail(Guardrail):
    def __init__(self, api_key: str = settings.gemini_api_key, base_url: str | None = None) -> None:
        http_options = None
        if base_url:
            http_options = genai.types.HttpOptions(base_url=base_url)
        
        self._client: genai.Client = genai.Client(api_key=api_key, http_options=http_options)
        self._default_model = settings.gemini_model

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
        view_text = f"view #{view_number}" if view_number > 0 else "multiple views (tiled in a 2x3 grid)"
        instructions = (
            f"You are a trust and safety analyst. Review these {view_text} of the 3D asset "
            f"named '{file_name}' ({file_format}) and flag any category that could undermine "
            "user trust. If multiple views are provided, scan each tile carefully. "
            "Only use the allowed categories and respond with a JSON object containing a 'findings' array."
            f"Allowed categories:\n"
            f"{categories_text}"
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
        log.info(
            "classifying view %d | model=%s file=%s",
            view_number,
            model_to_use,
            file_name,
        )

        content = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=instructions),                    
                    types.Part.from_bytes(data=screenshot, mime_type="image/png"),
                ],
            )
        ]

        response = await self._client.aio.models.generate_content(
            model=model_to_use,
            contents=content,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=response_format,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            ),
        )

        findings_list = (
            response.parsed.get("findings", [])
            if isinstance(response.parsed, dict)
            else []
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
