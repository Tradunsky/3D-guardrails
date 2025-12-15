"""Utilities for working with 3D assets and generating multi-view renders."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import trimesh


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
    """Generate renders from multiple viewpoints."""
    scene = mesh.scene()
    scene.camera.fov = (45, 45)
    scene.camera.resolution = config.resolution
    renders: List[bytes] = []
    for azimuth_deg, elevation_deg in config.view_angles:
        scene.set_camera(
            angles=_to_radians((azimuth_deg, elevation_deg)),
            distance=config.distance,
            center=mesh.centroid,
        )
        png_bytes = scene.save_image(resolution=config.resolution, visible=False)
        if not png_bytes:
            raise AssetProcessingError("Failed to render multi-view screenshot.")
        renders.append(io.BytesIO(png_bytes).getvalue())
    return renders
