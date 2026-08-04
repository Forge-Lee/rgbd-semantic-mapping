import imageio.v3 as iio
import numpy as np

from scripts.inspect_raw_images import DATASET_ROOT
from src.rgbd_mapping.geometry.backprojection import backproject_rgbd
from src.rgbd_mapping.geometry.transforms import camera_pose_to_matrix, transform_points

def build_world_pointcloud(
    records,
    dataset_root = DATASET_ROOT,
    fx: float = 525.0, # official intrinsic parameter
    fy: float = 525.0,
    cx: float = 319.5,
    cy: float = 239.5,
    frame_step: int = 20,
    pixel_stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a merged colored point cloud in the world frame.
    """
    all_world_points = []
    all_colors = []

    selected_records = records[::frame_step]
    for record in selected_records:
        pair = record.rgbd_pair
        rgb_path = dataset_root / pair.rgb_path
        depth_path = dataset_root / pair.depth_path

        print(rgb_path)

        rgb = iio.imread(rgb_path)
        depth_raw = iio.imread(depth_path)

        depth_meters = depth_raw.astype(np.float32) / 5000

        # recovered 3-d pointclouds in cam frame
        points_camera, colors = backproject_rgbd(
            rgb = rgb, 
            depth_meters = depth_meters,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            stride= pixel_stride,
        )

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

        all_world_points.append(points_world)
        all_colors.append(colors)

    # merge the selected points
    merged_points = np.concatenate(
    all_world_points,
        axis=0,
    )

    merged_colors = np.concatenate(
        all_colors,
        axis=0,
    )

    return merged_points, merged_colors