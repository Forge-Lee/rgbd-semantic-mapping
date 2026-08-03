import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scripts.inspect_raw_images import main as inspect_main, DATASET_ROOT
from src.rgbd_mapping.geometry.backprojection import backproject_rgbd
from src.rgbd_mapping.geometry.pointcloud_io import save_colored_ply

def visualize():
    # get one matched frame to test the code validity first
    records = inspect_main()
    record = records[len(records) // 2]
    pair = record.rgbd_pair

    rgb_path = DATASET_ROOT / pair.rgb_path
    depth_path = DATASET_ROOT / pair.depth_path

    print(rgb_path)

    rgb = iio.imread(rgb_path)
    depth_raw = iio.imread(depth_path)

    depth_meters = depth_raw.astype(np.float32) / 5000
    # valid_depth = depth_meters[depth_meters > 0]


    points, colors = backproject_rgbd(rgb, depth_meters)

    # sanity checks
    print("Points:", points.shape)
    print("Colors:", colors.shape)

    print(
        "X range:",
        points[:, 0].min(),
        points[:, 0].max(),
    )

    print(
        "Y range:",
        points[:, 1].min(),
        points[:, 1].max(),
    )

    print(
        "Z range:",
        points[:, 2].min(),
        points[:, 2].max(),
    )    

    # visualization tests
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")

    axis.scatter(
        points[:, 0],
        points[:, 2],
        -points[:, 1],
        c=colors,
        s=1,
    )

    axis.set_xlabel("X: right")
    axis.set_ylabel("Z: forward")
    axis.set_zlabel("-Y: up")

    

    OUTPUT_PATH = Path("outputs/day2/single_frame_pointcloud.png")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    PLY_OUTPUT_PATH = Path(
        "outputs/day2/single_frame_pointcloud.ply"
    )

    save_colored_ply(
        output_path=PLY_OUTPUT_PATH,
        points=points,
        colors=colors,
    )

    print(
        "Saved point cloud to:",
        PLY_OUTPUT_PATH.resolve(),
)

    plt.tight_layout()
    plt.savefig(
        "outputs/day2/single_frame_pointcloud.png",
        dpi=200,
    )
    plt.show()

if __name__ == "__main__":
    visualize()