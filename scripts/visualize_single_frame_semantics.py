from pathlib import Path
import time
import numpy as np

from scripts.inspect_raw_images import main as inspect_main
from src.rgbd_mapping.geometry.pointcloud_io import save_colored_ply
from src.rgbd_mapping.mapping.multiframe import build_world_semantic_pointcloud
from src.rgbd_mapping.semantics.inference import SemanticSegmenter
# from src.rgbd_mapping.mapping.remapping import CoarseLabelRemapper, COARSE_CLASS_NAMES
from src.rgbd_mapping.semantics.palette import create_palette
from src.rgbd_mapping.mapping.semantic_voxel import semantic_voxel_fusion

POINTCLOUD_OUTPUT_DIR = Path("outputs/day5")

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
    frame_step = [10]
    # voxel_size = [0.01, 0.02, 0.05]
    voxel_size = [0.02]
    experiment_results = []

    POINTCLOUD_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segmenter = SemanticSegmenter(device="cpu")
    # remapper = CoarseLabelRemapper(
    #     fine_id_to_label=segmenter.id_to_label
    # )
    for step in frame_step:
        for size in voxel_size:
            start_time = time.perf_counter()
            (merged_points, merged_colors, merged_labels, merged_confidences) = build_world_semantic_pointcloud(
                records=records, 
                frame_step=step, 
                segmenter=segmenter,
                # remapper = remapper
            )
            (fused_points, fused_rgb_colors, fused_labels, fused_confidences, observation_counts) = semantic_voxel_fusion(
                points=merged_points, 
                rgb_colors=merged_colors, 
                labels=merged_labels, 
                confidences=merged_confidences, 
                voxel_size=size
            )
            
            palette = create_palette(number_of_classes=len(segmenter.id_to_label))
                    
            semantic_colors = (
                palette[fused_labels]
                .astype(np.float32)
                / 255.0
            )

            build_time_seconds = (time.perf_counter() - start_time)

            output_path = (
                POINTCLOUD_OUTPUT_DIR
                / (
                    f"map_step_{step}"
                    f"_voxel_{size:.2f}.ply"
                )
            )

            output_semantic_path = (
                POINTCLOUD_OUTPUT_DIR
                / (
                    f"map_step_{step}"
                    f"_voxel_{size:.2f}_return_semantic.ply"
                )
            )

            # visualization tests
            print(f"Saving outputs/day4/multiframe_semantic_world_{step}_{size}.ply")
            save_colored_ply(
                output_path=output_path,
                points=fused_points,
                colors=fused_rgb_colors,
            )

            save_colored_ply(
                output_path=output_semantic_path,
                points=fused_points,
                colors=semantic_colors,
            )

            # save_coarse_class_layers(
            #     output_directory=Path(
            #         "outputs/semantics/coarse_layers"
            #     ),
            #     points=fused_points,
            #     coarse_labels=fused_labels,
            #     coarse_palette=palette,
            #     coarse_class_names=COARSE_CLASS_NAMES,
            # )

            raw_point_count = len(merged_points)
            downsampled_point_count = len(fused_points)

            retention_ratio = (
                downsampled_point_count / raw_point_count
                if raw_point_count > 0
                else 0.0
            )

            ply_size_mb = (
                output_path.stat().st_size
                / (1024 * 1024)
            )

            experiment_results.append(
                {
                    "frame_step": step,
                    "voxel_size_m": size,
                    "pixel_stride": 4,
                    "selected_frames": len(records[::step]),
                    "raw_point_count": raw_point_count,
                    "downsampled_point_count":
                        downsampled_point_count,
                    "retention_ratio": round(
                        retention_ratio,
                        6,
                    ),
                    "build_time_seconds": round(
                        build_time_seconds,
                        4,
                    ),
                    "ply_size_mb": round(
                        ply_size_mb,
                        4,
                    ),
                    "output_path": str(output_path),
                    "notes": "",
                }
            )
            print("Raw semantic observations:", len(merged_points))
            print("Fused voxel count:", len(fused_points))

            print(
                "Retention ratio:",
                len(fused_points) / len(merged_points),
            )

            print(
                "Mean observations per voxel:",
                observation_counts.mean(),
            )

            print(
                "Multi-observation voxel ratio:",
                np.mean(observation_counts > 1),
            )

            print(
                "Mean fused confidence:",
                fused_confidences.mean(),
            )

            print(
                "Low-agreement voxel ratio:",
                np.mean(fused_confidences < 0.6),
            )



if __name__ == "__main__":
    visualize()