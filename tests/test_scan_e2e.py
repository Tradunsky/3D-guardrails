import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dddguardrails import api
from dddguardrails.config import settings
from dddguardrails.guardrail import CATEGORIES


@pytest.mark.e2e
def test_scan_detects_guardrail_violation():
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY must be set (loaded via .env) for full E2E run.")

    asset_path = Path(__file__).resolve().parents[1] / "data" / "nazi_pistol.glb"
    assert asset_path.exists(), "Expected sample GLB asset to be present for E2E test."

    client = TestClient(api.app)
    response = client.post(
        "/scan",
        files={"file": (asset_path.name, asset_path.read_bytes(), "model/gltf-binary")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["file_name"] == asset_path.name
    assert payload["file_format"] == "glb"
    assert payload["metadata"]["views_evaluated"] > 0
    assert isinstance(payload["findings"], list)
    assert payload["findings"], "Expected at least one guardrail finding for the sample asset."

    for finding in payload["findings"]:
        assert finding["category"] in CATEGORIES
        assert finding["severity"] in {"none", "low", "medium", "high"}
    assert any(finding["severity"] != "none" for finding in payload["findings"])
