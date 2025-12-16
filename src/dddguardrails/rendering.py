"""Utilities for working with 3D assets and generating multi-view renders."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import trimesh
import pyrender
import pyrr


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

    `trimesh.Scene.set_camera` expects three Euler angles
    (rx, ry, rz) which are forwarded to `euler_matrix(*angles)`.
    We allow callers to specify either:

    - (azimuth, elevation)       -> roll defaults to 0
    - (azimuth, elevation, roll) -> all explicitly provided
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


def generate_multiview_images(
    mesh: trimesh.Trimesh, config: RenderConfig
) -> List[bytes]:
    """Generate renders from multiple viewpoints using pyrender for headless rendering."""
    # Convert trimesh mesh to pyrender mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Create pyrender scene
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)

    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)  # 60 degrees FOV

    # Set up renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=config.resolution[0],
        viewport_height=config.resolution[1]
    )

    renders: List[bytes] = []

    for azimuth_deg, elevation_deg in config.view_angles:
        azimuth_rad, elevation_rad, _ = _to_radians((azimuth_deg, elevation_deg, 0))

        # Calculate camera position
        distance = config.distance
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)

        camera_position = np.array([x, y, z])
        center = mesh.centroid

        # Create look-at matrix using pyrr
        look_at_matrix = pyrr.matrix44.create_look_at(
            eye=camera_position,
            target=center,
            up=np.array([0, 0, 1])
        )

        # Invert to get camera pose and ensure it's a proper transformation matrix
        camera_pose = np.linalg.inv(look_at_matrix)
        camera_pose[3, :] = [0, 0, 0, 1]  # Ensure bottom row is correct

        # Add camera to scene with pose
        camera_node = scene.add(camera, pose=camera_pose)

        # Render
        color, _ = renderer.render(scene)
        renders.append(color.tobytes())

        # Remove camera for next iteration
        scene.remove_node(camera_node)

    renderer.delete()
    return renders
