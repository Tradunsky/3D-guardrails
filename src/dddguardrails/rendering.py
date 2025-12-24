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


def _render(
    file_path: str    
) -> List[bytes]:
    """
    Worker function to render the scene in a separate process.
    Loads mesh from file path to ensure a clean OpenGL context and avoid pickling issues.
    """
    # Import settings and create logger inside worker to avoid pickling module-level state    
    
    log.info("Worker process started, loading mesh from %s", file_path)
    
    config = RenderConfig(
        resolution=settings.screenshot_resolution,
        distance=settings.camera_distance,
        view_angles=settings.multi_view_angles,
    )
    # Load mesh with materials directly in worker process
    loaded = trimesh.load(file_path, skip_materials=False)
    log.info("Mesh loaded successfully, type: %s", type(loaded).__name__)
    
    # Convert to pyrender meshes
    if isinstance(loaded, trimesh.Scene):
        # For scenes, we need to convert each geometry with its materials
        meshes_to_render = []
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and not geom.is_empty:
                # Use smooth=False to preserve textures and original appearance
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                meshes_to_render.append(pr_mesh)
    else:
        # Single mesh
        meshes_to_render = [pyrender.Mesh.from_trimesh(loaded, smooth=False)]
    
    log.info("Created %d pyrender meshes", len(meshes_to_render))
    
    # Scene with darker background and lower ambient light for better contrast
    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.05, 1.0], ambient_light=[0.3, 0.3, 0.3])
    
    # Add all meshes to scene
    for pr_mesh in meshes_to_render:
        scene.add(pr_mesh)
    
    # Calculate bounding box for all meshes
    if isinstance(loaded, trimesh.Scene):
        bounds = loaded.bounds
    else:
        bounds = loaded.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    radius = np.linalg.norm(extent) / 2.0
    
    # Setup Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)

     
    # Setup Enhanced Lighting for better visual quality
    # 1. Main Key Light (camera-aligned headlight)
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=6.0)
    key_light_node = pyrender.Node(light=key_light, matrix=np.eye(4))
    scene.add_node(key_light_node)
    
    # 2. Top-down Fill Light (softer, cooler tone)
    fill_light = pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=3.5)
    fill_node = scene.add(fill_light, pose=trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    
    # 3. Rim Light from the side for edge definition
    rim_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5)
    rim_pose = trimesh.transformations.rotation_matrix(np.pi/3, [0, 1, 0])
    rim_node = scene.add(rim_light, pose=rim_pose)
    
    # 4. Bottom Fill Light (warm tone to fill shadows)
    back_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.0)
    back_node = scene.add(back_light, pose=trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    
    
    # Initializing renderer
    r = pyrender.OffscreenRenderer(viewport_width=config.resolution[0], viewport_height=config.resolution[1])
    
    renders: List[bytes] = []
    
    try:
        # Calculate distance based on bounding sphere
        fov = np.pi / 3.0  # match camera yfov
        # Calculate distance to fit the bounding sphere perfectly
        base_distance = radius / np.sin(fov / 2.0)
        
        # Apply the distance from config as a scale factor
        render_distance = base_distance * config.distance
        
        for azimuth_deg, elevation_deg in config.view_angles:
            # Calculate camera position
            azimuth_rad, elevation_rad, _ = _to_radians((azimuth_deg, elevation_deg, 0))
            camera_position = _spherical_to_cartesian(render_distance, azimuth_rad, elevation_rad)
            
            # Position camera relative to center
            camera_position = camera_position + center
            
            # Look at center
            target = center
            up = np.array([0, 1, 0]) # Y-up
            f = target - camera_position
            f = f / np.linalg.norm(f)
            s = np.cross(f, up)
            # If camera is directly above/below, s might be zero. Handle that.
            if np.linalg.norm(s) < 1e-3:
                # Camera is looking along Z. Up vector can be Y.
                s = np.cross(f, np.array([0, 1, 0]))
            
            s = s / np.linalg.norm(s)
            u = np.cross(s, f)
            
            # rotation matrix columns: s, u, -f
            R = np.eye(4)
            R[:3, 0] = s
            R[:3, 1] = u
            R[:3, 2] = -f
            
            # Translation
            T = np.eye(4)
            T[:3, 3] = camera_position
            
            pose = T @ R
            
            # Update camera node pose
            scene.set_pose(camera_node, pose)
            # Update key light node pose to match camera (headlight effect)
            # scene.set_pose(key_light_node, pose)
            
            log.info("Rendering view %d/%d (azimuth=%d, elevation=%d)", 
                    len(renders) + 1, len(config.view_angles), azimuth_deg, elevation_deg)
            
            start_time = time.time()
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            # Convert to PIL and bytes
            image = Image.fromarray(color)
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                renders.append(output.getvalue())
            
            end_time = time.time()
            log.info("Rendered view %d/%d (azimuth=%d, elevation=%d) in %f seconds", 
                    len(renders), len(config.view_angles), azimuth_deg, elevation_deg, end_time - start_time)
                
    finally:
        r.delete()
    
    log.info("All renders complete, returning %d screenshots", len(renders))
    return renders


def generate_multiview_images(file_path: str) -> List[bytes]:
    """
    Generate renders from multiple viewpoints using pyrender in a separate process.
    Pass file path to worker to avoid pickling issues with mesh data.
    
    Args:
        file_path: Path to the 3D asset file on disk
        config: Rendering configuration
    
    Returns:
        List of PNG image bytes for each view
    """
    log.info("Start generating screenshots")
    return _render(file_path)
