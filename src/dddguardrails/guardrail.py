"""LLM integration for asset trust analysis."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from dddguardrails.config import settings


CATEGORIES = {
    "weapons": "Explosives, firearms, or realistic weapon replicas that could enable violence.",
    "nudity": "Explicit nudity, sexual content, or fetish imagery.",
    "self_harm": "Content encouraging suicide or self-harm.",
    "extremism": "Symbols or content tied to extremist, terroristic, or hate organizations.",
    "hate_symbols": "Imagery targeting protected classes with hate speech or symbols.",
    "misleading": "Fraudulent or deceptive items (fake credentials, scam artifacts).",
}

log = logging.getLogger("dddguardrails.llm")

CATEGORIES = {
    "weapons": "Explosives, firearms, or realistic weapon replicas that could enable violence.",
    "nudity": "Explicit nudity, sexual content, or fetish imagery.",
    "self_harm": "Content encouraging suicide or self-harm.",
    "extremism": "Symbols or content tied to extremist, terroristic, or hate organizations.",
    "hate_symbols": "Imagery targeting protected classes with hate speech or symbols.",
    "misleading": "Fraudulent or deceptive items (fake credentials, scam artifacts).",
}


class Guardrail:
    def classify(self, *, screenshots: List[bytes], file_name: str, file_format: str) -> List[Dict[str, Any]]:
        pass
