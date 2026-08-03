from pathlib import Path

import numpy as np


def save_colored_ply(
    output_path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    Save a colored point cloud as an ASCII PLY file.

    Args:
        output_path:
            Destination .ply path.
        points:
            Array with shape (N, 3).
        colors:
            Array with shape (N, 3). Values may be in [0, 1]
            or [0, 255].
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected points with shape (N, 3), got {points.shape}"
        )

    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(
            f"Expected colors with shape (N, 3), got {colors.shape}"
        )

    if len(points) != len(colors):
        raise ValueError(
            "Points and colors must contain the same number of rows."
        )

    # Remove invalid 3D points.
    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    # Convert colors from [0, 1] to [0, 255], if necessary.
    if np.issubdtype(colors.dtype, np.floating):
        if colors.size > 0 and colors.max() <= 1.0:
            colors = colors * 255.0

    colors = np.clip(
        np.rint(colors),
        0,
        255,
    ).astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vertex_data = np.column_stack(
        (
            points,
            colors,
        )
    )

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        np.savetxt(
            file,
            vertex_data,
            fmt=[
                "%.6f",
                "%.6f",
                "%.6f",
                "%d",
                "%d",
                "%d",
            ],
        )