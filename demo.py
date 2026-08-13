#!/usr/bin/env python3
"""Gradio demo for 3D Guardrails with MCP support."""

import os
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
from dddguardrails.schemas import CATEGORIES, RiskCategory
from dddguardrails.config import settings

log = getLogger(__name__)


async def scan_3d_asset(
    file_path: Optional[str], 
    llm_provider_model: str,
    res_w: int,
    res_h: int,
    risk_cats_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    """
    Scan a 3D asset using the 3D Guardrails business logic directly.

    Args:
        file_path: The uploaded 3D file path
        llm_provider_model: LLM provider/model to use
        res_w: Resolution width
        res_h: Resolution height
        risk_cats_df: DataFrame with custom risk categories

    Returns:
        DataFrame with findings and status message
    """
    if file_path is None:
        return pd.DataFrame(
            columns=["Category", "Severity", "Rationale", "View Number"]
        ), "Please upload a 3D file to scan.", pd.DataFrame(
            columns=["Stage", "Legend", "ms"]
        )

    try:
        llm_provider, model = llm_provider_model.split("/", maxsplit=1)
        # Convert risk categories dataframe to list of RiskCategory for the API
        risk_cats = []
        for _, row in risk_cats_df.iterrows():
            if row["name"] and row["description"]:
                risk_cats.append(RiskCategory(name=str(row["name"]), description=str(row["description"])))

        with open(file_path, mode="rb") as f:
            upload_file = UploadFile(
                file=BytesIO(f.read()), filename=Path(file_path).name
            )

        result = await scan_asset(
            file=upload_file,
            llm_provider=llm_provider,
            model=model.strip() if model and model.strip() else None,
            provider_base_url=None,
            resolution_width=res_w,
            resolution_height=res_h,
            risk_categories=risk_cats if risk_cats else None,
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
        
        latency = result.metadata["latency"]
        latencies_data = [
            {"Stage": "Total", "Legend": "Total Latency", "ms": latency["total_ms"]},
            {"Stage": "Breakdown", "Legend": "Rendering", "ms": latency["rendering_ms"]},
            {"Stage": "Breakdown", "Legend": "LLM", "ms": latency["llm_ms"]},
        ]
        latencies_df = pd.DataFrame(latencies_data)
        
        return findings_df, status, latencies_df

    except Exception as e:
        log.error("❌ Error: ", e, exc_info=True)
        return pd.DataFrame(
            columns=["Category", "Severity", "Rationale", "View Number"]
        ), f"❌ Error: {str(e)}", pd.DataFrame(
            columns=["Stage", "Legend", "ms"]
        )


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
            label="VLM (Editable)",
            value="cerebras/gemma-4-31b",
            choices=[
                "gemini/gemini-3.6-flash",
                "gemini/gemini-3.5-flash-lite",
                "gemini/gemini-3.1-pro-preview",
                "gemini/gemma-4-31b-it",
                "gemini/gemma-4-26b-a4b-it",
                "openai/gpt-5.6-luna",
                "openai/gpt-5.6-terra",
                "ollama/qwen3-vl:235b-cloud",
                "groq/qwen/qwen3.6-27b",
                "cerebras/gemma-4-31b",
            ],
            info="Leave empty to use the provider's default model",
            allow_custom_value=True,
        ),
        gr.Slider(
            label="Resolution Width (reduce for faster processing)",
            minimum=64,
            maximum=2048,
            step=64,
            value=settings.screenshot_resolution[0],
        ),
        gr.Slider(
            label="Resolution Height (reduce for faster processing)",
            minimum=64,
            maximum=2048,
            step=64,
            value=settings.screenshot_resolution[1],
        ),
        gr.Dataframe(
            label="Risk Categories (edit, add or remove rows)",
            headers=["name", "description"],
            datatype=["str", "str"],
            value=[[c.name, c.description] for c in CATEGORIES],
            column_count=(2, "fixed"),
            interactive=True,
        )
    ],
    outputs=[
        gr.Dataframe(
            label="Risk Findings",
            headers=["Category", "Severity", "Rationale", "View Number"],
        ),
        gr.Textbox(label="Status"),
        gr.BarPlot(
            x="ms",
            y="Stage",
            x_title="Latency (ms)",
            y_title="",
            color="Legend",
            title="Processing Latency Breakdown",
            tooltip=["Legend", "ms"],
            height=250,
        ),
    ],
    title="🛡️ 3D Guardrails with MCP",
    description="Scan 3D assets for trust and safety risks using multimodal AI with MCP (Model Context Protocol) enabled. Supported formats: GLB, GLTF, FBX, OBJ, STL, PLY. Risk categories: Weapons, Nudity, Self-harm, Extremism, Hate symbols, Misleading content.\n Github: https://github.com/Tradunsky/3D-guardrails",
    analytics_enabled=False,
    examples=[
        [
            str(dataset_dir / file),
            "cerebras/gemma-4-31b",
            settings.screenshot_resolution[0],
            settings.screenshot_resolution[1],
            [[c.name, c.description] for c in CATEGORIES]
        ]
        for file in os.listdir(dataset_dir) if (dataset_dir / file).is_file()
    ],
    cache_examples=False,
    cache_mode="lazy",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        mcp_server=True, 
    )
