"""Utilities for working with 3D assets and generating multi-view renders."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence, Tuple

import pyvista as pv
import numpy as np
import trimesh
import logging
from PIL import Image
from dddguardrails.config import settings    

log = logging.getLogger(__name__)

class AssetProcessingError(RuntimeError):
    """Raised when an uploaded asset cannot be processed."""



def _to_radians(angles: Iterable[int]) -> Tuple[float, float, float]:
    """
    Convert camera angles specified in degrees to a 3‑tuple of radians.
    """
    vals = list(angles)
    if len(vals) == 2:
        azimuth_deg, elevation_deg = vals
        roll_deg = 0
    elif len(vals) == 3:
        azimuth_deg, elevation_deg, roll_deg = vals
    else:
        raise AssetProcessingError(
            "Camera angles must be 2 or 3 values (azimuth, elevation[, roll])."
        )
    return (
        float(np.deg2rad(azimuth_deg)),
        float(np.deg2rad(elevation_deg)),
        float(np.deg2rad(roll_deg)),
    )


def _spherical_to_cartesian(distance: float, azimuth: float, elevation: float) -> np.ndarray:
    """Standard Y-up spherical to cartesian conversion."""
    x = distance * np.cos(elevation) * np.cos(azimuth)
    z = distance * np.cos(elevation) * np.sin(azimuth)
    y = distance * np.sin(elevation)
    return np.array([x, y, z])

def _get_mesh_stats(loaded):
    bounds = loaded.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = np.linalg.norm(extent) / 2.0
    return center, radius

def _get_camera_positions(center, radius, distance_multiplier=1.2):
    fov = np.pi / 3.0
    render_distance = (radius / np.sin(fov / 2.0)) * distance_multiplier
    positions = []
    for az_deg, el_deg in settings.multi_view_angles:
        az_rad, el_rad, _ = _to_radians((az_deg, el_deg))
        camera_pos = _spherical_to_cartesian(render_distance, az_rad, el_rad) + center
        positions.append(camera_pos)
    return positions

def _get_texture_image(material):
    if material is None: return None
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        return material.baseColorTexture
    if hasattr(material, 'image') and material.image is not None:
        return material.image
    return None

BG_COLOR = [0.05, 0.05, 0.05, 1.0]

def render_tiled_views(
    contents: bytes,
    extension: str,
    resolution: Tuple[int, int],
) -> bytes:
    """Render all views and stitch them into a single tiled image (2 rows, 3 columns)."""
    start_total = time.perf_counter()
    
    file_obj = io.BytesIO(contents)
    loaded = trimesh.load(file_obj, file_type=extension, skip_materials=False)
    log.info("Mesh loaded successfully for tiled render, type: %s", type(loaded).__name__)
    
    center, radius = _get_mesh_stats(loaded)
    cam_positions = _get_camera_positions(center, radius)
    
    # Calculate cell size for a 2x3 grid to match target resolution
    # resolution is (width, height)
    total_w, total_h = resolution
    cell_w = total_w // 3
    cell_h = total_h // 2
    
    pl = pv.Plotter(off_screen=True, window_size=(cell_w, cell_h), lighting=None)
    
    try:
        # Add mesh once - this logic is proven to work in render_views_generator
        if isinstance(loaded, trimesh.Scene):
            for g in loaded.geometry.values():
                if isinstance(g, trimesh.Trimesh): 
                    mesh = pv.wrap(g)
                    tex = None
                    if hasattr(g.visual, 'material'):
                        image = _get_texture_image(g.visual.material)
                        if image is not None:
                            tex = pv.Texture(np.array(image))
                    pl.add_mesh(mesh, texture=tex)
        else: 
            mesh = pv.wrap(loaded)
            tex = None
            if hasattr(loaded.visual, 'material'):
                image = _get_texture_image(loaded.visual.material)
                if image is not None:
                    tex = pv.Texture(np.array(image))
            pl.add_mesh(mesh, texture=tex)

        pl.background_color = BG_COLOR[:3]
        pl.add_light(pv.Light(position=(0, 0, 1), color='white', intensity=1.5, light_type='camera light'))
        pl.add_light(pv.Light(position=(0, 1, 0), color=[0.9, 0.95, 1.0], intensity=1.0))
        pl.add_light(pv.Light(position=(1, 0, 0), color=[1.0, 0.95, 0.9], intensity=0.7))

        views = []
        for idx, pos in enumerate(cam_positions[:6]):
            pl.camera_position = [pos, center, (0.0, 1.0, 0.0)]
            pl.camera.view_angle = 60
            pl.render()
            img_array = pl.screenshot(None, return_img=True)
            views.append(Image.fromarray(img_array))

        # Stitch them: 2 rows, 3 columns
        tiled_img = Image.new('RGB', (cell_w * 3, cell_h * 2), color=(0, 0, 0))
        for i, img in enumerate(views):
            row, col = divmod(i, 3)
            tiled_img.paste(img, (col * cell_w, row * cell_h))

        with io.BytesIO() as bio:
            tiled_img.save(bio, format="PNG")
            img_bytes = bio.getvalue()

        total_ms = (time.perf_counter() - start_total) * 1000
        log.info("Rendered and stitched 6 views in %.3f ms", total_ms)
        return img_bytes
    finally:
        pl.close()
    
