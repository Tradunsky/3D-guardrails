"""Configuration helpers for 3D guardrail processing."""

from __future__ import annotations

import os
from typing import Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration container."""

    supported_formats: Tuple[str, ...] = (
        "glb",
        "gltf",
        "obj",
        "fbx",
        "stl",
        "ply",
    )
    screenshot_resolution: Tuple[int, int] = (256, 256)
    camera_distance: float = 1.2
    multi_view_angles: Tuple[Tuple[int, int], ...] = (
        (0, 10),
        (90, 10),
        (180, 10),
        (270, 10),
        (45, 45),
        (225, 45),
    )
    openai_api_key: str

    ollama_api_key: str = os.getenv("OLLAMA_API_KEY", "sk-no-key-local-ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")

    gemini_api_key: str
    gemini_base_url: str = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

    groq_api_key: str = os.getenv("GROQ_API_KEY")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    cerebras_api_key: str = os.getenv("CEREBRAS_API_KEY")
    cerebras_base_url: str = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra="allow")



settings = Settings()

import pyvista
import numpy as np

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True

# Optional - set parameters like theme or window size
pyvista.set_plot_theme('document')
pyvista.global_theme.window_size = np.array([1024, 768]) * 2

extensions = [
    ...,
    "sphinx_gallery.gen_gallery",
]

# Add the PyVista image scraper to SG
sphinx_gallery_conf = {
    "image_scrapers": ('pyvista', ...),
}