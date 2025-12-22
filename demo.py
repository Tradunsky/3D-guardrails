#!/usr/bin/env python3
"""Gradio demo for 3D Guardrails with MCP support."""

import sys

from io import BytesIO
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

# Add src to path so we can import dddguardrails
sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import pandas as pd
from fastapi import UploadFile

from dddguardrails.api import scan_asset

log = getLogger(__name__)


async def scan_3d_asset(
    file_path: Optional[str], llm_provider: str, model: str
) -> Tuple[pd.DataFrame, str]:
    """
    Scan a 3D asset using the 3D Guardrails business logic directly.

    Args:
        file_path: The uploaded 3D file path
        llm_provider: LLM provider to use ('openai', 'gemini', 'ollama')
        model: Specific model to use

    Returns:
        DataFrame with findings and status message
    """
    if file_path is None:
        return pd.DataFrame(
            columns=["Category", "Severity", "Rationale", "View Number"]
        ), "Please upload a 3D file to scan."

    try:
        with open(file_path, mode="rb") as f:
            upload_file = UploadFile(
                file=BytesIO(f.read()), filename=Path(file_path).name
            )

        result = await scan_asset(
            file=upload_file,
            llm_provider=llm_provider,
            model=model.strip() if model and model.strip() else None,
        )

        # Process findings for display
        findings_data = []
        for finding in result.findings:
            findings_data.append(
                {
                    "Category": finding.category,
                    "Severity": finding.severity.upper(),
                    "Rationale": finding.rationale,
                    "View Number": finding.view_number,
                }
            )

        if not findings_data:
            findings_data = [
                {
                    "Category": "No violations detected",
                    "Severity": "",
                    "Rationale": "",
                    "View Number": "",
                }
            ]

        findings_df = pd.DataFrame(findings_data)

        # Status message
        violation_count = len([f for f in result.findings if f.severity != "none"])
        views_evaluated = result.metadata["views_evaluated"]

        if violation_count > 0:
            status = f"⚠️ Found {violation_count} violation(s) after evaluating {views_evaluated} views."
        else:
            status = (
                f"✅ No violations detected after evaluating {views_evaluated} views."
            )

        return findings_df, status

    except Exception as e:
        log.error("❌ Error: ", e, exc_info=True)
        return pd.DataFrame(
            columns=["Category", "Severity", "Rationale", "View Number"]
        ), f"❌ Error: {str(e)}"


dataset_dir = Path(__file__).parent / "tests/data"

# Create the Gradio Interface with MCP support
demo = gr.Interface(
    fn=scan_3d_asset,
    inputs=[
        gr.Model3D(
            label="3D Model File",
        )
    ],
    additional_inputs=[
        gr.Dropdown(
            label="LVM Provider",
            choices=["gemini", "openai", "ollama"],
            value="gemini",
            info="Select the AI model provider for analysis",
        ),
        gr.Dropdown(
            label="Model (Optional, Editable)",
            value="gemini-3-flash-preview",
            choices=[                
                "gemini-3-flash-preview",
                "gemini-3-pro-preview",
                "gemini-2.5-flash-preview-09-2025",
                "gpt-4o",
                "gpt-5.2",
                "qwen3-vl:235b-cloud",
            ],
            info="Leave empty to use the provider's default model",
            allow_custom_value=True,
        )
    ],
    outputs=[
        gr.Dataframe(
            label="Risk Findings",
            headers=["Category", "Severity", "Rationale", "View Number"],
        ),
        gr.Textbox(label="Status"),
    ],
    title="🛡️ 3D Guardrails with MCP",
    description="Scan 3D assets for trust and safety risks using multimodal AI with MCP (Model Context Protocol) enabled. Supported formats: GLB, GLTF, FBX, OBJ, STL, PLY. Risk categories: Weapons, Nudity, Self-harm, Extremism, Hate symbols, Misleading content.\n Github: https://github.com/Tradunsky/3D-guardrails",
    analytics_enabled=False,
    # examples=[
        # [str(dataset_dir / file), "gemini", "gemini-3-pro-preview"]
        # for file in os.listdir(dataset_dir)
    # ],
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True,
    )
