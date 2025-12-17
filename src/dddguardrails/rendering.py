"""Utilities for working with 3D assets and generating multi-view renders."""

from __future__ import annotations

import io
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Any

# Set EGL platform for headless rendering.
# This must be set before pyrender is imported.
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import trimesh
import pyrender
from PIL import Image


class AssetProcessingError(RuntimeError):
    """Raised when an uploaded asset cannot be processed."""


@dataclass(slots=True)
class RenderConfig:
    resolution: Tuple[int, int]
    distance: float
    view_angles: Sequence[Tuple[int, int]]


def _ensure_mesh(asset: trimesh.base.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(asset, trimesh.Scene):
        if not asset.geometry:
            raise AssetProcessingError("The provided scene is empty.")
        combined = trimesh.util.concatenate(tuple(asset.geometry.values()))
        return combined
    if isinstance(asset, trimesh.Trimesh):
        if asset.is_empty:
            raise AssetProcessingError("The provided mesh is empty.")
        return asset
    if isinstance(asset, list):
        meshes = [mesh for mesh in asset if isinstance(mesh, trimesh.Trimesh)]
        if not meshes:
            raise AssetProcessingError(
                "The provided asset did not contain mesh geometry."
            )
        return trimesh.util.concatenate(meshes)
    raise AssetProcessingError(f"Unsupported asset type: {type(asset)!r}")


def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh from disk and normalize it for downstream rendering."""
    try:
        loaded = trimesh.load(
            path,
            force="mesh",
            validate=True,
            skip_materials=True,
        )
    except Exception as exc:  # pragma: no cover - upstream lib error surface.
        raise AssetProcessingError("Unable to parse the 3D asset.") from exc

    mesh = _ensure_mesh(loaded)
    mesh.remove_unreferenced_vertices()
    # mesh.remove_duplicate_faces()
    mesh.rezero()
    return mesh


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
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    return np.array([x, y, z])


def _render_worker(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: RenderConfig
) -> List[bytes]:
    """
    Worker function to render the scene in a separate process.
    Recreates the mesh and scene to ensure a clean OpenGL context.
    """
    # Create mesh from data
    tm = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = pyrender.Mesh.from_trimesh(tm)
    
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(mesh)
    
    # Setup Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    
    # Setup Light (attached to camera or fixed?)
    # Attaching a light to the camera node ensures it illuminates the object from the view
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node) # Add separately or attach to camera? 
    # Let's attach light to camera so it moves with it
    
    # Initializing renderer
    r = pyrender.OffscreenRenderer(viewport_width=config.resolution[0], viewport_height=config.resolution[1])
    
    renders: List[bytes] = []
    
    try:
        for azimuth_deg, elevation_deg in config.view_angles:
            # Calculate camera position
            azimuth_rad, elevation_rad, _ = _to_radians((azimuth_deg, elevation_deg, 0))
            camera_position = _spherical_to_cartesian(config.distance, azimuth_rad, elevation_rad)
            
            # Look at origin
            # pyrender uses a specific coordinate system. 
            # We can use trimesh.transformations or build a query.
            # However, pyrender's camera looks along -Z. 
            # We need a generic look_at function.
            
            # Simple look_at implementation
            target = np.array([0, 0, 0])
            up = np.array([0, 0, 1]) # Assuming Z up for the mesh
            f = target - camera_position
            f = f / np.linalg.norm(f)
            s = np.cross(f, up)
            # If camera is directly above/below, s might be zero. Handle that?
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
            # Update light node pose to match camera (headlight effect)
            scene.set_pose(light_node, pose)
            
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            
            # Convert to PIL and bytes
            image = Image.fromarray(color)
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                renders.append(output.getvalue())
                
    finally:
        r.delete()
        
    return renders


def generate_multiview_images(
    mesh: trimesh.Trimesh, config: RenderConfig
) -> List[bytes]:
    """
    Generate renders from multiple viewpoints using pyrender in a separate process.
    """
    # Extract data to pass to worker
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    
    # We use a ProcessPoolExecutor to ensure isolated OpenGL context
    # max_workers=1 because we want to run this batch of renders in one isolated environment
    # but we don't need parallel rendering of *individual views* (too much overhead).
    # We just want to get off the main process.
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_render_worker, vertices, faces, config)
        return future.result()
