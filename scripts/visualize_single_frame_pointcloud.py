import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scripts.inspect_raw_images import main as inspect_main, DATASET_ROOT
from src.rgbd_mapping.geometry.backprojection import backproject_rgbd
from src.rgbd_mapping.geometry.pointcloud_io import save_colored_ply
from src.rgbd_mapping.geometry.transforms import camera_pose_to_matrix, transform_points

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

    # recovered 3-d pointclouds in cam frame
    points_camera, colors = backproject_rgbd(rgb, depth_meters)

    # transform to world frame
    # transform matrix from cam frame to world frame
    transform_world_camera = camera_pose_to_matrix(
        translation=record.pose.translation,
        quaternion_xyzw=record.pose.quaternion_xyzw,
    )

    # utilize the wTc to get the world frame coordinates
    points_world = transform_points(
        points=points_camera,
        transform=transform_world_camera,
    )

    # sanity check
    # origin point
    camera_origin = np.array([
        [0.0, 0.0, 0.0]
    ])

    camera_origin_world = transform_points(
        camera_origin,
        transform_world_camera,
    )

    print(
        "Camera origin in world:",
        camera_origin_world[0],
    )

    print(
        "GT translation:",
        record.pose.translation,
    )

    # distance between certain points
    camera_distance = np.linalg.norm(
        points_camera[0] - points_camera[100]
    )

    world_distance = np.linalg.norm(
        points_world[0] - points_world[100]
    )

    print(
        "Camera-frame distance:",
        camera_distance,
    )

    print(
        "World-frame distance:",
        world_distance,
    )

    # print out the parameters
    np.set_printoptions(
        precision=5,
        suppress=True,
    )

    print("T_world_camera:")
    print(transform_world_camera)

    print(
        "Camera-frame centroid:",
        points_camera.mean(axis=0),
    )

    print(
        "World-frame centroid:",
        points_world.mean(axis=0),
    )

    print(
        "World X range:",
        points_world[:, 0].min(),
        points_world[:, 0].max(),
    )

    print(
        "World Y range:",
        points_world[:, 1].min(),
        points_world[:, 1].max(),
    )

    print(
        "World Z range:",
        points_world[:, 2].min(),
        points_world[:, 2].max(),
    )

    # visualization tests
    camera_output_path = Path(
        "outputs/day3/single_frame_camera.ply"
    )

    world_output_path = Path(
        "outputs/day3/single_frame_world.ply"
    )

    save_colored_ply(
        output_path=camera_output_path,
        points=points_camera,
        colors=colors,
    )

    save_colored_ply(
        output_path=world_output_path,
        points=points_world,
        colors=colors,
    )

    camera_output_path.parent.mkdir(parents=True, exist_ok=True)

    # PLY_OUTPUT_PATH = Path(
    #     "outputs/day2/single_frame_pointcloud.ply"
    # )

    # fig = plt.figure(figsize=(10, 8))
    # axis = fig.add_subplot(111, projection="3d")

    # axis.scatter(
    #     points_world[:, 0],
    #     points_world[:, 2],
    #     -points_world[:, 1],
    #     c=colors,
    #     s=1,
    # )

    # axis.set_xlabel("X: right")
    # axis.set_ylabel("Z: forward")
    # axis.set_zlabel("-Y: up")

    # save_colored_ply(
    #     output_path=PLY_OUTPUT_PATH,
    #     points=points_world,
    #     colors=colors,
    # )

    # print(
    #     "Saved point cloud to:",
    #     PLY_OUTPUT_PATH.resolve(),
    # )

    # plt.tight_layout()
    # plt.savefig(
    #     "outputs/day2/single_frame_pointcloud.png",
    #     dpi=200,
    # )
    # plt.show()

if __name__ == "__main__":
    visualize()