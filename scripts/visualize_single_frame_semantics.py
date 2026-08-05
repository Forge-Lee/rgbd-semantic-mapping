from pathlib import Path
import csv
import time
import numpy as np

from scripts.inspect_raw_images import main as inspect_main
from src.rgbd_mapping.geometry.pointcloud_io import save_colored_ply
from src.rgbd_mapping.geometry.voxel import voxel_downsample
from src.rgbd_mapping.mapping.multiframe import build_world_semantic_pointcloud
from src.rgbd_mapping.semantics.inference import SemanticSegmenter
from src.rgbd_mapping.semantics.palette import create_palette

POINTCLOUD_OUTPUT_DIR = Path("outputs/day4")

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
    for step in frame_step:
        for size in voxel_size:
            start_time = time.perf_counter()
            (merged_points, merged_colors, merged_labels, merged_confidences) = build_world_semantic_pointcloud(records=records, frame_step=step, segmenter=segmenter)
            palette = create_palette(
                        number_of_classes=len(segmenter.id_to_label)
                    )
                    
            semantic_colors = (
                palette[merged_labels]
                .astype(np.float32)
                / 255.0
            )

            downsampled_points, downsampled_colors = voxel_downsample(
                points= merged_points,
                colors= merged_colors,
                voxel_size=size
            )

            build_time_seconds = (
                time.perf_counter() - start_time
            )

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
                    f"_voxel_{size:.2f}_semantic.ply"
                )
            )

            # visualization tests
            print(f"Saving outputs/day4/multiframe_semantic_world_{step}_{size}.ply")
            save_colored_ply(
                output_path=output_path,
                points=downsampled_points,
                colors=downsampled_colors,
            )

            save_colored_ply(
                output_path=output_semantic_path,
                points=merged_points,
                colors=semantic_colors,
            )

            raw_point_count = len(merged_points)
            downsampled_point_count = len(downsampled_points)

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



if __name__ == "__main__":
    visualize()