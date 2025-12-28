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

def render_views_generator(
    contents: bytes,
    extension: str,
    resolution: Tuple[int, int],
) -> Generator[bytes, None, None]:
    start_total = time.perf_counter()
    
    # LOADING
    file_obj = io.BytesIO(contents)
    loaded = trimesh.load(file_obj, file_type='glb', skip_materials=False)
    trimesh_total_ms = (time.perf_counter() - start_total) * 1000
    log.info("Mesh loaded successfully, type: %s | time=%f ms", type(loaded).__name__, trimesh_total_ms)
    start_time = time.perf_counter()
    center, radius = _get_mesh_stats(loaded)
    cam_positions = _get_camera_positions(center, radius)
    

    pl = pv.Plotter(off_screen=True, window_size=resolution, lighting=None)
    try:
        if isinstance(loaded, trimesh.Scene):
            for name, g in loaded.geometry.items():
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
        
        for idx, pos in enumerate(cam_positions):
            pl.camera_position = [pos, center, (0.0, 1.0, 0.0)]
            pl.camera.view_angle = 60
            pl.render()
            img_array = pl.screenshot(None, return_img=True)
            
            # Convert to PNG bytes
            img_pil = Image.fromarray(img_array)
            with io.BytesIO() as bio:
                img_pil.save(bio, format="PNG")
                img_bytes = bio.getvalue()

            render_total_ms = (time.perf_counter() - start_time) * 1000
            log.info("Rendered view %d in %.3f ms", idx, render_total_ms)

            yield img_bytes
            start_time = time.perf_counter()
    finally:
        pl.close()
    
