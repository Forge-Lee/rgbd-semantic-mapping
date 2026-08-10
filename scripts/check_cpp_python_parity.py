from src.rgbd_mapping.mapping.semantic_voxel_map import SemanticVoxelMap
import numpy as np
import subprocess

if __name__ == "__main__":
    map = SemanticVoxelMap(0.01)
    points = np.array([
        [ 0.001, 0.002, 0.003],
        [ 0.004, 0.005, 0.006],
        [ 0.011, 0.000, 0.000],
        [-0.001, 0.000, 0.000],
        [ 0.003, 0.001, 0.001],
    ], dtype=np.float64)
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)
    labels = np.array([1, 2, 3, 4, 1], dtype=np.int64)
    confidences = np.array([0.9, 0.6, 0.8, 0.7, 0.5], dtype=np.float64)

    map.update(
        points=points,
        rgb_colors=colors,
        labels=labels,
        confidences=confidences
    )

    outputs = map.export()
    result = subprocess.run(
        ["cpp/build/parity_driver"],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed_res = result.stdout.split('\n')
    cleaned_res = []

    for i in range(1, len(parsed_res)):
        res_line = parsed_res[i]
        if res_line:
            cleaned_res.append(np.array(res_line.split(','), dtype=np.float64)[3:])

    cleaned_array = np.array(cleaned_res)

    voxel_points = outputs.points
    voxel_rgb = outputs.rgb_colors
    voxel_label = outputs.labels.reshape(len(outputs.labels), 1)
    voxel_agreement = outputs.semantic_agreement.reshape(len(outputs.labels), 1)
    voxel_mean_conf = outputs.mean_model_confidence.reshape(len(outputs.labels), 1)
    voxel_count = outputs.observation_counts.reshape(len(outputs.labels), 1)

    voxel_summary = np.concatenate(
        (voxel_points, voxel_rgb, voxel_label, voxel_agreement, voxel_mean_conf, voxel_count),
        axis=1
    )

    comparison = np.isclose(cleaned_array, voxel_summary)

    print(comparison)