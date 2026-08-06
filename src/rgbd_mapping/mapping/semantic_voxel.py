from typing import Any

import numpy as np


def semantic_voxel_fusion(
    points: np.ndarray,
    rgb_colors: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    voxel_size: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Fuse semantic point observations that fall into the same voxel.

    Args:
        points:
            World-frame point coordinates, shape (N, 3).
        rgb_colors:
            RGB values in [0, 1], shape (N, 3).
        labels:
            Semantic class IDs, shape (N,).
        confidences:
            Per-point semantic confidence values in [0, 1],
            shape (N,).
        voxel_size:
            Voxel side length in meters.

    Returns:
        fused_points:
            Mean position of each voxel, shape (M, 3).
        fused_rgb_colors:
            Mean RGB color of each voxel, shape (M, 3).
        fused_labels:
            Confidence-weighted winning label, shape (M,).
        fused_confidences:
            Semantic agreement score, shape (M,).
        observation_counts:
            Number of observations in each voxel, shape (M,).
    """
    points = np.asarray(points)
    rgb_colors = np.asarray(rgb_colors)
    labels = np.asarray(labels)
    confidences = np.asarray(confidences)

    # ---------------------------------------------------------
    # Step 1: Validate inputs
    # ---------------------------------------------------------

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected points shape (N, 3), got {points.shape}"
        )

    number_of_points = len(points)

    if number_of_points == 0:
        raise ValueError("Cannot fuse an empty point cloud.")

    if rgb_colors.shape != points.shape:
        raise ValueError(
            "RGB colors must have the same shape as points: "
            f"{rgb_colors.shape} vs {points.shape}"
        )

    if labels.shape != (len(points),):
        raise ValueError(
            f"Expected labels shape ({len(points)},), "
            f"got {labels.shape}"
        )

    if confidences.shape != (len(points),):
        raise ValueError(
            f"Expected confidences shape ({len(points)},), "
            f"got {confidences.shape}"
        )

    if voxel_size <= 0:
        raise ValueError(
            f"voxel_size must be positive, got {voxel_size}"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("Points contain NaN or infinity.")

    if not np.all(np.isfinite(rgb_colors)):
        raise ValueError("RGB colors contain NaN or infinity.")

    if not np.all(np.isfinite(confidences)):
        raise ValueError("Confidences contain NaN or infinity.")

    if np.any(confidences < 0) or np.any(confidences > 1):
        raise ValueError(
            "Confidences must lie in the range [0, 1]."
        )

    # ---------------------------------------------------------
    # Step 2: Convert positions into integer voxel indices
    # ---------------------------------------------------------

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    # voxelize the pointcloud scene

    # Each voxel key maps to one accumulator.
    voxel_data: dict[
        tuple[int, int, int],
        dict[str, Any],
    ] = {}

    # ---------------------------------------------------------
    # Step 3: Accumulate observations voxel by voxel
    # ---------------------------------------------------------

    for (point, color, label, confidence, voxel_index,) in zip(points, rgb_colors, labels, confidences, voxel_indices,):

        key = (int(voxel_index[0]), int(voxel_index[1]), int(voxel_index[2])), 

        if key not in voxel_data:
            voxel_data[key] = {
                "point_sum": np.zeros(
                    3,
                    dtype=np.float64,
                ),
                "color_sum": np.zeros(
                    3,
                    dtype=np.float64,
                ),
                "count": 0,
                "label_scores": {},
            }

        accumulator = voxel_data[key]

        accumulator["point_sum"] += point
        accumulator["color_sum"] += color
        accumulator["count"] += 1

        label_id = int(label)
        confidence_value = float(confidence)

        label_scores: dict[int, float] = (
            accumulator["label_scores"]
        )

        try:
            # if the id has been recorded in label_scores
            label_scores[label_id] += confidence_value
        except:
            # if this is the first time to add the id into label_scores
            label_scores[label_id] = confidence_value

    # ---------------------------------------------------------
    # Step 4: Convert accumulators into output arrays
    # ---------------------------------------------------------

    number_of_voxels = len(voxel_data)

    fused_points = np.empty(
        (number_of_voxels, 3),
        dtype=np.float32,
    )

    fused_rgb_colors = np.empty(
        (number_of_voxels, 3),
        dtype=np.float32,
    )

    fused_labels = np.empty(
        number_of_voxels,
        dtype=np.int64,
    )

    fused_confidences = np.empty(
        number_of_voxels,
        dtype=np.float32,
    )

    observation_counts = np.empty(
        number_of_voxels,
        dtype=np.int32,
    )

    # Sorting gives deterministic output order.
    sorted_voxel_keys = sorted(voxel_data)

    for output_index, key in enumerate(sorted_voxel_keys):
        accumulator = voxel_data[key]

        count = accumulator["count"]
        label_scores = accumulator["label_scores"]

        fused_points[output_index] = accumulator["point_sum"] / count
        fused_rgb_colors[output_index] = accumulator["color_sum"] / count

        winning_label, winning_score = max(label_scores.items(), key = lambda item: item[1])

        total_score = sum(label_scores.values())

        fused_labels[output_index] = winning_label

        if total_score > 0:
            fused_confidences[output_index] = winning_score / total_score
        else:
            fused_confidences[output_index] = 0.0

        observation_counts[output_index] = count

    return (
        fused_points,
        fused_rgb_colors,
        fused_labels,
        fused_confidences,
        observation_counts,
    )