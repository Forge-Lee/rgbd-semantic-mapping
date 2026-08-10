from pathlib import Path
import imageio.v3 as iio
import numpy as np

from scripts.inspect_raw_images import main as inspect_main
from src.rgbd_mapping.geometry.pointcloud_io import save_colored_ply
from src.rgbd_mapping.geometry.backprojection import backproject_semantic_rgbd
from src.rgbd_mapping.semantics.inference import SemanticSegmenter
from src.rgbd_mapping.geometry.transforms import camera_pose_to_matrix, transform_points
from src.rgbd_mapping.semantics.palette import create_palette
from src.rgbd_mapping.mapping.semantic_voxel_map import SemanticVoxelMap

POINTCLOUD_OUTPUT_DIR = Path("outputs/day10/incre_test")
DATASET_ROOT = Path("data/raw/rgbd_dataset_freiburg1_xyz")
PARITY_OUTPUT_DIR = Path("outputs/parity")
PARITY_FRAMES = 20

def save_coarse_class_layers(
    output_directory: Path,
    points: np.ndarray,
    coarse_labels: np.ndarray,
    coarse_palette: np.ndarray,
    coarse_class_names: list[str],
) -> None:
    # semantic map optimizing codes
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for coarse_id, coarse_name in enumerate(
        coarse_class_names
    ):
        class_mask = (
            coarse_labels == coarse_id
        )

        point_count = int(
            np.count_nonzero(class_mask)
        )

        if point_count == 0:
            continue

        class_color = (
            coarse_palette[coarse_id]
            .astype(np.float32)
            / 255.0
        )

        class_colors = np.repeat(
            class_color[None, :],
            repeats=point_count,
            axis=0,
        )

        output_path = (
            output_directory
            / f"{coarse_id:02d}_{coarse_name}.ply"
        )

        save_colored_ply(
            output_path=output_path,
            points=points[class_mask],
            colors=class_colors,
        )

        print(
            f"{coarse_name:24s}: "
            f"{point_count:8d} voxels"
        )

def visualize():
    # get one matched frame to test the code validity first
    records = inspect_main()
    
    # frame_step = [10, 20, 50]
    step = 10
    # voxel_size = [0.01, 0.02, 0.05]
    size = 0.02
    experiment_results = []
    fx: float = 525.0 # official intrinsic parameter
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5

    POINTCLOUD_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    snapshot_dir = Path("outputs/demo_run/snapshots")
    run_dir = Path("outputs/demo_run")

    selected_records = records[::step]

    segmenter = SemanticSegmenter(device="cpu")
    semantic_map = SemanticVoxelMap(voxel_size=size)
    PARITY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parity_observation_batches = []
    parity_checkpoints = []

    palette = create_palette(number_of_classes=len(segmenter.id_to_label))

    for frame_index, record in enumerate(selected_records):
        pair = record.rgbd_pair

        rgb = iio.imread(
            DATASET_ROOT / pair.rgb_path
        )

        depth_meters = iio.imread(
            DATASET_ROOT / pair.depth_path
        ).astype(np.float32) / 5000

        prediction = segmenter.predict(rgb)

        (points_camera, rgb_colors, point_labels, point_confidences,) = backproject_semantic_rgbd(
            rgb=rgb,
            depth_meters=depth_meters,
            semantic_labels=prediction.labels,
            semantic_confidence=prediction.confidence,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            stride=4,
        )

        transform_world_camera = camera_pose_to_matrix(
            translation=record.pose.translation,
            quaternion_xyzw=record.pose.quaternion_xyzw,
        )

        points_world = transform_points(
            points=points_camera,
            transform=transform_world_camera,
        )

        frame_ids = np.full(
            (len(points_world), 1),
            frame_index,
            dtype=np.float64,
        )

        frame_observations = np.concatenate(
            (
                frame_ids,
                points_world,
                rgb_colors,
                point_labels.reshape(-1, 1),
                point_confidences.reshape(-1, 1),
            ),
            axis=1,
        )

        parity_observation_batches.append(
            frame_observations
        )

        semantic_map.update(
            points=points_world,
            rgb_colors=rgb_colors,
            labels=point_labels,
            confidences=point_confidences,
        )

        print(
            f"Processed frame {frame_index}: "
            f"{len(points_world)} observations, "
            f"{len(semantic_map)} voxels"
        )

        parity_checkpoints.append(
            (
                frame_index,
                len(points_world),
                len(semantic_map),
            )
        )

        # if frame_index % 5 == 0:
        #     snapshot = semantic_map.export()

        #     np.savez_compressed(
        #         snapshot_dir / f"snapshot_{frame_index:04d}.npz",
        #         points=snapshot.points,
        #         rgb_colors=snapshot.rgb_colors,
        #         labels=snapshot.labels,
        #         semantic_agreement=snapshot.semantic_agreement,
        #         mean_model_confidence=snapshot.mean_model_confidence,
        #         observation_counts=snapshot.observation_counts,
        #         current_points_world=points_world,
        #     )

        #     rgb_output_path = (
        #         run_dir
        #         / "rgb"
        #         / f"rgb_{frame_index:04d}.png"
        #     )

        #     iio.imwrite(
        #         rgb_output_path,
        #         rgb,
        #     )

        #     semantic_output_path = (
        #         run_dir
        #         / "semantic_2d"
        #         / f"semantic_{frame_index:04d}.png"
        #     )

        #     semantic_colors = (
        #         palette[prediction.labels]
        #         * 255.0
        #     ).astype(np.uint8)

        #     alpha = 0.45

        #     overlay = (
        #         (1.0 - alpha)
        #         * rgb.astype(np.float32)
        #         + alpha
        #         * semantic_colors.astype(np.float32)
        #     )

        #     overlay = np.clip(
        #         overlay,
        #         0,
        #         255,
        #     ).astype(np.uint8)

        #     iio.imwrite(
        #         semantic_output_path,
        #         overlay,
        #     )

        # if frame_index % 10 == 0:
        #     snapshot = semantic_map.export()

        #     semantic_colors = (
        #         palette[snapshot.labels]
        #         .astype(np.float32)
        #         / 255.0
        #     )

        #     save_colored_ply(
        #         output_path=(
        #             POINTCLOUD_OUTPUT_DIR
        #             / f"semantic_{frame_index:04d}.ply"
        #         ),
        #         points=snapshot.points,
        #         colors=semantic_colors,
        #     )

        #     save_colored_ply(
        #         output_path=(
        #             POINTCLOUD_OUTPUT_DIR
        #             / f"rgb_{frame_index:04d}.ply"
        #         ),
        #         points=snapshot.points,
        #         colors=snapshot.rgb_colors,
        #     )

    all_observations = np.concatenate(
        parity_observation_batches,
        axis=0,
    )

    python_output = semantic_map.export()

    python_summary = np.concatenate(
        (
            python_output.points,
            python_output.rgb_colors,
            python_output.labels.reshape(-1, 1),
            python_output.semantic_agreement.reshape(-1, 1),
            python_output.mean_model_confidence.reshape(-1, 1),
            python_output.observation_counts.reshape(-1, 1),
        ),
        axis=1,
    )

    np.savetxt(
        PARITY_OUTPUT_DIR
        / "incremental_observations.csv",
        all_observations,
        delimiter=",",
        header=(
            "frame_id,x,y,z,"
            "r,g,b,label,confidence"
        ),
        comments="",
        fmt=[
            "%.0f",   # frame_id
            "%.17g",  # x
            "%.17g",  # y
            "%.17g",  # z
            "%.17g",  # r
            "%.17g",  # g
            "%.17g",  # b
            "%.0f",   # label
            "%.17g",  # confidence
        ],
    )

    np.savetxt(
        PARITY_OUTPUT_DIR
        / "python_final_map.csv",
        python_summary,
        delimiter=",",
        header=(
            "x,y,z,"
            "r,g,b,"
            "label,semantic_agreement,"
            "mean_model_confidence,"
            "observation_count"
        ),
        comments="",
        fmt=[
            "%.17g", "%.17g", "%.17g",
            "%.17g", "%.17g", "%.17g",
            "%.0f",
            "%.17g",
            "%.17g",
            "%.0f",
        ],
    )

    checkpoint_array = np.asarray(
        parity_checkpoints,
        dtype=np.int64,
    )

    np.savetxt(
        PARITY_OUTPUT_DIR
        / "python_checkpoints.csv",
        checkpoint_array,
        delimiter=",",
        header=(
            "frame_id,"
            "observation_count,"
            "voxel_count"
        ),
        comments="",
        fmt="%d",
    )

if __name__ == "__main__":
    visualize()