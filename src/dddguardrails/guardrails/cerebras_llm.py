from typing import List

from dddguardrails.config import settings
from dddguardrails.guardrail import Guardrail
from dddguardrails.guardrails.openai_llm import OpenAIGuardrail
from dddguardrails.schemas import RiskCategory, RiskFinding


class CerebrasGuardrail(Guardrail):
    def __init__(self, api_key: str = settings.cerebras_api_key, base_url: str | None = None) -> None:
        self.guardrail = OpenAIGuardrail(api_key, base_url or settings.cerebras_base_url)
        self.guardrail._default_model = settings.cerebras_model

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
        return await self.guardrail.classify(
            screenshot=screenshot,
            view_number=view_number,
            file_name=file_name,
            file_format=file_format,
            risk_categories=risk_categories,
            model=model
        )
