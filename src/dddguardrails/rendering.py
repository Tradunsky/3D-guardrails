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


def _get_texture_image(material):
    if material is None: return None
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        return material.baseColorTexture
    if hasattr(material, 'image') and material.image is not None:
        return material.image
    return None


def _add_meshes_to_plotter(loaded, pl: pv.Plotter) -> None:
    """Add all meshes from a trimesh.Scene or trimesh.Trimesh to the plotter.
    
    For Scenes, applies the scene-graph transforms so geometry is in world space,
    which is critical for correct bounds and camera positioning.
    """
    if isinstance(loaded, trimesh.Scene):
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            g = loaded.geometry[geom_name]
            if not isinstance(g, trimesh.Trimesh) or len(g.vertices) == 0:
                continue
            mesh = pv.wrap(g)
            tex = None
            if hasattr(g.visual, 'material'):
                image = _get_texture_image(g.visual.material)
                if image is not None:
                    try:
                        tex = pv.Texture(np.array(image))
                    except Exception:
                        tex = None
            actor = pl.add_mesh(mesh, texture=tex, smooth_shading=True)
            actor.user_matrix = transform
    else:
        mesh = pv.wrap(loaded)
        tex = None
        if hasattr(loaded.visual, 'material'):
            image = _get_texture_image(loaded.visual.material)
            if image is not None:
                try:
                    tex = pv.Texture(np.array(image))
                except Exception:
                    tex = None
        pl.add_mesh(mesh, texture=tex, smooth_shading=True)


def _compute_camera_params(pl: pv.Plotter, fov_deg: float = 45.0, distance_multiplier: float = 2.0):
    """Derive center and camera distance from PyVista's own renderer bounds.
    
    This is the single source of truth — PyVista already knows what was added
    and where it lives in world space.
    
    Args:
        pl: Plotter with all meshes already added.
        fov_deg: Vertical field of view in degrees.
        distance_multiplier: Extra margin factor (>1) so the object doesn't fill 100% of frame.
    
    Returns:
        (center, render_distance)
    """
    bounds = pl.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    if bounds is None:
        raise AssetProcessingError("Plotter has no bounds — was any mesh added?")
    
    mn = np.array([bounds[0], bounds[2], bounds[4]])
    mx = np.array([bounds[1], bounds[3], bounds[5]])
    center = (mn + mx) / 2.0
    extent = mx - mn
    
    # Use the largest axis-aligned half-extent as the "radius"
    # This is tighter than diagonal but more predictable for non-spherical models
    half_extents = extent / 2.0
    radius = float(np.max(half_extents))
    
    # Correct perspective formula: distance so that the radius fills half the FOV
    # tan(fov/2) = radius / distance  =>  distance = radius / tan(fov/2)
    fov_rad = np.deg2rad(fov_deg)
    render_distance = (radius / np.tan(fov_rad / 2.0)) * distance_multiplier
    
    log.info(
        "Bounds: mn=%s mx=%s center=%s radius=%.3f render_distance=%.3f",
        mn, mx, center, radius, render_distance
    )
    return center, render_distance


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
    
    # Calculate cell size for a 2x3 grid to match target resolution
    # resolution is (width, height)
    total_w, total_h = resolution
    cell_w = total_w // 3
    cell_h = total_h // 2
    
    fov_deg = 45.0
    
    pl = pv.Plotter(off_screen=True, window_size=(cell_w, cell_h), lighting=None)
    
    try:
        # Add all meshes with world-space transforms applied
        _add_meshes_to_plotter(loaded, pl)
        
        pl.background_color = BG_COLOR[:3]
        pl.add_light(pv.Light(position=(0, 0, 1), color='white', intensity=1.5, light_type='camera light'))
        pl.add_light(pv.Light(position=(0, 1, 0), color=[0.9, 0.95, 1.0], intensity=1.0))
        pl.add_light(pv.Light(position=(1, 0, 0), color=[1.0, 0.95, 0.9], intensity=0.7))

        # Derive center and render distance from the plotter's own bounds
        center, render_distance = _compute_camera_params(pl, fov_deg=fov_deg, distance_multiplier=1.5)

        views = []
        for az_deg, el_deg in settings.multi_view_angles[:6]:
            az_rad, el_rad, _ = _to_radians((az_deg, el_deg))
            cam_pos = _spherical_to_cartesian(render_distance, az_rad, el_rad) + center
            
            pl.camera_position = [cam_pos.tolist(), center.tolist(), (0.0, 1.0, 0.0)]
            pl.camera.view_angle = fov_deg
            pl.reset_camera_clipping_range()
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
