from logging import getLogger
from typing import List
import json
import base64

import ollama

from dddguardrails.guardrail import Guardrail
from dddguardrails.schemas import RiskFinding, CATEGORIES, RiskCategory
from dddguardrails.config import settings

log = getLogger()


class OllamaGuardrail(Guardrail):
    def __init__(self, base_url: str | None = None) -> None:
        self._client = ollama.AsyncClient(host=base_url)
        self._default_model = settings.ollama_model

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
        
        categories_text = "\n".join(f"- {c.name}: {c.description}" for c in risk_categories)
        view_text = f"view #{view_number}" if view_number > 0 else "multiple views (tiled)"
        instructions = (
            f"You are a trust and safety analyst. Review these {view_text} of the 3D asset "
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

        # Convert image bytes to base64 string for Ollama
        b64_image = base64.b64encode(screenshot).decode("ascii")

        response = await self._client.chat(
            model=model_to_use,
            messages=[
                {
                    "role": "user",
                    "content": f"{instructions}\n\nRespond with a JSON object containing a 'findings' array.",
                    "images": [b64_image],
                }
            ],
        )

        # Extract response content
        if (
            not response
            or not response.get("message")
            or not response["message"].get("content")
        ):
            raise RuntimeError(
                f"No response content from Ollama for screenshot {view_number}"
            )

        content = response["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse JSON response from Ollama: {exc}"
            ) from exc

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
