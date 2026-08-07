import numpy as np

from src.rgbd_mapping.mapping.semantic_voxel_map import (
    SemanticVoxelMap,
)


def create_test_observations() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create six semantic point observations.

    With voxel_size = 0.01:

        points 0, 1, 2 → voxel (0, 0, 0)
        points 3, 4    → voxel (1, 0, 0)
        point 5        → voxel (-1, 0, 0)

    The negative point is included to verify that voxel
    indexing correctly uses np.floor().
    """
    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.004, 0.003, 0.002],
            [0.008, 0.002, 0.005],
            [0.011, 0.001, 0.001],
            [0.018, 0.004, 0.003],
            [-0.001, 0.002, 0.002],
        ],
        dtype=np.float32,
    )

    rgb_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [
            3,
            3,
            7,
            12,
            12,
            5,
        ],
        dtype=np.int64,
    )

    confidences = np.array(
        [
            0.8,
            0.7,
            0.4,
            0.9,
            0.6,
            0.75,
        ],
        dtype=np.float32,
    )

    return (
        points,
        rgb_colors,
        labels,
        confidences,
    )


def test_batch_and_incremental_updates_match() -> None:
    """
    Processing observations in one update or several updates
    should produce the same final voxel map.
    """
    (
        points,
        rgb_colors,
        labels,
        confidences,
    ) = create_test_observations()

    voxel_size = 0.01

    # -----------------------------------------------------
    # Map A: process all observations in one update
    # -----------------------------------------------------

    batch_map = SemanticVoxelMap(
        voxel_size=voxel_size,
    )

    batch_map.update(
        points=points,
        rgb_colors=rgb_colors,
        labels=labels,
        confidences=confidences,
    )

    batch_output = batch_map.export()

    # -----------------------------------------------------
    # Map B: process the same observations incrementally
    # -----------------------------------------------------

    incremental_map = SemanticVoxelMap(
        voxel_size=voxel_size,
    )

    split_index = 3

    incremental_map.update(
        points=points[:split_index],
        rgb_colors=rgb_colors[:split_index],
        labels=labels[:split_index],
        confidences=confidences[:split_index],
    )

    incremental_map.update(
        points=points[split_index:],
        rgb_colors=rgb_colors[split_index:],
        labels=labels[split_index:],
        confidences=confidences[split_index:],
    )

    incremental_output = incremental_map.export()

    # -----------------------------------------------------
    # TODO 1: Compare the number of voxels
    # -----------------------------------------------------

    # Hint:
    # assert len(batch_map) == ...
    # assert batch_output.points.shape == ...

    assert len(batch_map) == len(incremental_map)
    assert batch_output.points.shape == incremental_output.points.shape

    # -----------------------------------------------------
    # TODO 2: Compare floating-point outputs
    # -----------------------------------------------------

    # Use np.allclose() for:
    #   points
    #   rgb_colors
    #   semantic_agreement
    #   mean_model_confidence

    if not np.allclose(batch_output.points, incremental_output.points):
        print("Points invalid")

    if not np.allclose(batch_output.rgb_colors, incremental_output.rgb_colors):
        print("rgb invalid")
    
    if not np.allclose(batch_output.semantic_agreement, incremental_output.semantic_agreement):
        print("semantic agreement invalid")
    
    if not np.allclose(batch_output.mean_model_confidence, incremental_output.mean_model_confidence):
        print("mean_model_confidence invalid")

    # -----------------------------------------------------
    # TODO 3: Compare integer outputs
    # -----------------------------------------------------

    # Use np.array_equal() for:
    #   labels
    #   observation_counts

    if not np.array_equal(batch_output.labels, incremental_output.labels):
        print("labels invalid")

    if not np.array_equal(batch_output.observation_counts, incremental_output.observation_counts):
        print("observation_counts invalid")

    print("test finished")

test_batch_and_incremental_updates_match()