from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dddguardrails import api
from dddguardrails.config import settings
from dddguardrails.schemas import CATEGORIES, RiskCategory


# Expected violation status for each test file
# True = violations expected, False = no violations expected
EXPECTED_VIOLATIONS = {
    "nazi_gemany_eagle_logo.glb": True,
    "swastika_3D.glb": True,
    "bikini_woman.glb": True,
    "belt.glb": True,
    "candy.glb": True,
    "diao_chann.glb": False,
    "homer.obj": False,
    "bimba.obj": False,
    "rocker_arm.obj": False,
}


@pytest.mark.e2e
@pytest.mark.parametrize("filename,expects_violation", EXPECTED_VIOLATIONS.items())
def test_scan_detects_guardrail_violation(filename, expects_violation):
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY must be set (loaded via .env) for full E2E run.")

    data_dir = Path(__file__).resolve().parent / "data"
    asset_path = data_dir / filename
    assert asset_path.exists(), f"Expected test asset {filename} to be present for E2E test."

    mime_type = "multipart/form-data"

    client = TestClient(api.app)
    response = client.post(
        "/scan",
        files={"file": (asset_path.name, asset_path.read_bytes(), mime_type)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["file_name"] == asset_path.name
    file_format = filename.split('.')[-1] if '.' in filename else "unknown"
    assert payload["file_format"] == file_format
    assert payload["metadata"]["views_evaluated"] > 0
    assert isinstance(payload["findings"], list)

    # Validate finding structure
    cat_names = {c.name for c in CATEGORIES}
    for finding in payload["findings"]:
        assert finding["category"] in cat_names
        assert finding["severity"] in {"none", "low", "medium", "high"}

    # Check if violations were detected as expected
    has_violations = any(finding["severity"] != "none" for finding in payload["findings"])

    if expects_violation:
        assert has_violations, f"Expected violations for {filename} but none were found."
    else:
        assert not has_violations, f"Expected no violations for {filename} but violations were found: {payload['findings']}"


CUSTOM_RISK_CATEGORIES_VIOLATIONS = {
    "candy.glb": True,    
    "homer.obj": False,
}

@pytest.mark.e2e
@pytest.mark.parametrize("filename,expects_violation", CUSTOM_RISK_CATEGORIES_VIOLATIONS.items())
def test_scan_with_custom_risk_categories(filename, expects_violation):
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY must be set (loaded via .env) for full E2E run.")

    data_dir = Path(__file__).resolve().parent / "data"
    asset_path = data_dir / filename
    assert asset_path.exists(), f"Expected test asset {filename} to be present for E2E test."
    risk_categories = [
        RiskCategory(
            name="weapons", 
            description="Explosives, firearms, or realistic weapon replicas that could enable violence."
        ),
        RiskCategory(
            name="hate_symbols", 
            description="Imagery targeting protected classes with hate speech or symbols."
        ),
    ]

    mime_type = "multipart/form-data"

    client = TestClient(api.app)
    response = client.post(
        "/scan",
        files={"file": (asset_path.name, asset_path.read_bytes(), mime_type)},
        json={"risk_categories": risk_categories},
    )
    
    assert response.status_code == 200, response.text
    payload = response.json()

    # Validate finding structure
    cat_names = {c.name for c in CATEGORIES}
    for finding in payload["findings"]:
        assert finding["category"] in cat_names
        assert finding["severity"] in {"none", "low", "medium", "high"}

    # Check if violations were detected as expected
    has_violations = any(finding["severity"] != "none" for finding in payload["findings"])

    if expects_violation:
        assert has_violations, f"Expected violations for {filename} but none were found."
    else:
        assert not has_violations, f"Expected no violations for {filename} but violations were found: {payload['findings']}"