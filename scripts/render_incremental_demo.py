from pathlib import Path

import imageio.v3 as iio
import numpy as np
import open3d as o3d


def load_snapshot(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)

    return {
        key: data[key]
        for key in data.files
    }


def create_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        points.astype(np.float64)
    )

    cloud.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64)
    )

    return cloud


def render_snapshot(
    snapshot_path: Path,
    output_path: Path,
    palette: np.ndarray,
) -> None:
    snapshot = load_snapshot(snapshot_path)

    points = snapshot["points"]
    labels = snapshot["labels"]

    # TODO:
    # Apply uncertain → unknown here if desired.

    semantic_colors = (
        palette[labels].astype(np.float32) / 255.0
    )

    accumulated_cloud = create_point_cloud(
        points=points,
        colors=semantic_colors,
    )

    current_points = snapshot[
        "current_points_world"
    ]

    current_colors = np.ones_like(
        current_points,
        dtype=np.float32,
    )

    current_cloud = create_point_cloud(
        points=current_points,
        colors=current_colors,
    )

    # TODO:
    # 1. Create OffscreenRenderer
    # 2. Add accumulated_cloud
    # 3. Add current_cloud
    # 4. Set a fixed camera
    # 5. Render to image
    # 6. Save output_path