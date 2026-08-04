import numpy as np

def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge points that fall inside the same voxel.

    Each output point and color is the average of all samples
    inside that voxel.
    """
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    voxel_data = {}

    for point, color, voxel_index in zip(
        points,
        colors,
        voxel_indices,
    ):
        key = tuple(voxel_index)

        if key not in voxel_data:
            voxel_data[key] = {
                "point_sum": point.copy(),
                "color_sum": color.copy(),
                "count": 1,
            }
        else:
            voxel_data[key]["point_sum"] += point
            voxel_data[key]["color_sum"] += color
            voxel_data[key]["count"] += 1


    downsampled_points = []
    downsampled_colors = []

    for data in voxel_data.values():
        count = data["count"]
        downsampled_points.append(data["point_sum"] / count)
        downsampled_colors.append(data["color_sum"] / count)

    return (
        np.asarray(downsampled_points),
        np.asarray(downsampled_colors),
    )