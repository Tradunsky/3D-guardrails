"""Utilities for working with 3D assets and generating multi-view renders."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# Set EGL platform for headless rendering.
# This must be set before pyrender is imported.
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
from PIL import Image
import logging
from dddguardrails.config import settings    

log = logging.getLogger(__name__)

class AssetProcessingError(RuntimeError):
    """Raised when an uploaded asset cannot be processed."""


@dataclass(slots=True)
class RenderConfig:
    resolution: Tuple[int, int]
    distance: float
    view_angles: Sequence[Tuple[int, int]]


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


def render_views_generator(
    contents: bytes,
    extension: str,
) -> Generator[bytes, None, None]:
    """
    Generator that yields screenshots one by one from 3D asset bytes.
    """    
    start_time = time.perf_counter()
    log.info("Starting rendering generator | extension=%s", extension)
    
    config = RenderConfig(
        resolution=settings.screenshot_resolution,
        distance=settings.camera_distance,
        view_angles=settings.multi_view_angles,
    )
    
    # Load mesh from bytes    
    file_obj = io.BytesIO(contents)    
    loaded = trimesh.load(file_obj, file_type=extension, skip_materials=False)
    trimesh_total_ms = (time.perf_counter() - start_time) * 1000
    log.info("Mesh loaded successfully, type: %s | time=%f ms", type(loaded).__name__, trimesh_total_ms)
    start_time = time.perf_counter()

    # Convert to pyrender meshes
    if isinstance(loaded, trimesh.Scene):
        meshes_to_render = []
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and not geom.is_empty:
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                meshes_to_render.append(pr_mesh)
    else:
        meshes_to_render = [pyrender.Mesh.from_trimesh(loaded, smooth=False)]
    
    if not meshes_to_render:
        raise AssetProcessingError("No valid geometry found in the uploaded asset.")

    log.info("Created %d pyrender meshes", len(meshes_to_render))
    
    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.05, 1.0], ambient_light=[0.3, 0.3, 0.3])
    for pr_mesh in meshes_to_render:
        scene.add(pr_mesh)
    
    bounds = loaded.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = np.linalg.norm(extent) / 2.0
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)

    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=6.0)
    scene.add_node(pyrender.Node(light=key_light, matrix=np.eye(4)))
    
    fill_light = pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=3.5)
    scene.add(fill_light, pose=trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    
    rim_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5)
    scene.add(rim_light, pose=trimesh.transformations.rotation_matrix(np.pi/3, [0, 1, 0]))
    
    back_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.0)
    scene.add(back_light, pose=trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    
    r = pyrender.OffscreenRenderer(viewport_width=config.resolution[0], viewport_height=config.resolution[1])
    
    try:
        fov = np.pi / 3.0
        base_distance = radius / np.sin(fov / 2.0)
        render_distance = base_distance * config.distance
        
        for idx, (azimuth_deg, elevation_deg) in enumerate(config.view_angles, start=1):
            azimuth_rad, elevation_rad, _ = _to_radians((azimuth_deg, elevation_deg, 0))
            camera_position = _spherical_to_cartesian(render_distance, azimuth_rad, elevation_rad)
            camera_position = camera_position + center
            
            target = center
            up = np.array([0, 1, 0])
            f = target - camera_position
            f = f / np.linalg.norm(f)
            s = np.cross(f, up)
            if np.linalg.norm(s) < 1e-3:
                s = np.cross(f, np.array([0, 1, 0]))
            s = s / np.linalg.norm(s)
            u = np.cross(s, f)
            
            R = np.eye(4)
            R[:3, 0] = s
            R[:3, 1] = u
            R[:3, 2] = -f
            
            T = np.eye(4)
            T[:3, 3] = camera_position
            pose = T @ R
            
            scene.set_pose(camera_node, pose)
            
            log.info("Rendering view %d/%d (azimuth=%d, elevation=%d)", 
                    idx, len(config.view_angles), azimuth_deg, elevation_deg)
                        
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            image = Image.fromarray(color)
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                render_total_ms = (time.perf_counter() - start_time) * 1000
                log.info("Rendered view %d in %.3f ms", idx, render_total_ms)
                yield output.getvalue()
            start_time = time.perf_counter()
            
                
    finally:
        r.delete()
