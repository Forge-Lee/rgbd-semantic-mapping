from pathlib import Path

from scripts.inspect_raw_images import main as inspect_main
from src.rgbd_mapping.geometry.pointcloud_io import save_colored_ply
from src.rgbd_mapping.geometry.voxel import voxel_downsample
from src.rgbd_mapping.mapping.multiframe import build_world_pointcloud

def visualize():
    # get one matched frame to test the code validity first
    records = inspect_main()
    frame_step = [10, 20, 50]
    voxel_size = [0.01, 0.02, 0.05]
    for step in frame_step:
        for size in voxel_size:
            merged_points, merged_colors = build_world_pointcloud(records=records, frame_step=step)
            downsampled_points, downsampled_colors = voxel_downsample(
                points= merged_points,
                colors= merged_colors,
                voxel_size=size
            )

            # visualization tests
            print(f"Saving outputs/day3/multiframe_world_{step}_{size}.ply")
            save_colored_ply(
                output_path=Path(f"outputs/day3/multiframe_world_{step}_{size}.ply"),
                points=downsampled_points,
                colors=downsampled_colors,
            )


if __name__ == "__main__":
    visualize()