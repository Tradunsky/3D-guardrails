from logging import getLogger
from typing import List
import json
import base64

import ollama

from dddguardrails.guardrail import Guardrail, RiskFinding, CATEGORIES
from dddguardrails.config import settings

log = getLogger()


class OllamaGuardrail(Guardrail):
    def __init__(self, base_url: str = settings.ollama_base_url) -> None:
        self._client = ollama.Client(host=base_url)
        self._default_model = settings.ollama_model

    def classify(
        self,
        *,
        screenshots: List[bytes],
        file_name: str,
        file_format: str,
        model: str | None = None,
    ) -> List[RiskFinding]:
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
        log.info(
            "classifying | model=%s views=%d file=%s",
            model_to_use,
            len(screenshots),
            file_name,
        )

        # Process screenshots one by one until first violation is found
        for idx, image_bytes in enumerate(screenshots, start=1):
            # Convert image bytes to base64 string for Ollama
            b64_image = base64.b64encode(image_bytes).decode("ascii")

            log.debug(
                "checking screenshot %d/%d for file=%s",
                idx,
                len(screenshots),
                file_name,
            )

            response = self._client.chat(
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
                    f"No response content from Ollama for screenshot {idx}"
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

            # Early exit if violations found
            if findings_list:
                log.info(
                    "found violations in screenshot %d/%d for file=%s",
                    idx,
                    len(screenshots),
                    file_name,
                )
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
        log.info(
            "no violations found for file=%s after checking %d screenshots",
            file_name,
            len(screenshots),
        )
        return []
