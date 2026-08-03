import numpy as np
from scipy.spatial.transform import Rotation


def camera_pose_to_matrix(
    translation: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """
    Construct T_world_camera from a TUM ground-truth pose.

    Args:
        translation:
            Camera position in the world frame, shape (3,).
        quaternion_xyzw:
            Camera orientation, ordered as [qx, qy, qz, qw],
            shape (4,).

    Returns:
        A 4x4 homogeneous transformation matrix that maps
        camera-frame points into the world frame.
    """
    translation = np.asarray(
        translation,
        dtype=np.float64,
    )

    quaternion_xyzw = np.asarray(
        quaternion_xyzw,
        dtype=np.float64,
    )

    # shape checks
    if translation.shape != (3,):
        raise ValueError(
            f"Expected translation shape (3,), got {translation.shape}"
        )

    if quaternion_xyzw.shape != (4,):
        raise ValueError(
            "Expected quaternion shape (4,), "
            f"got {quaternion_xyzw.shape}"
        )

    quaternion_norm = np.linalg.norm(quaternion_xyzw)

    if quaternion_norm == 0:
        raise ValueError("Quaternion norm cannot be zero.")

    quaternion_xyzw = (
        quaternion_xyzw / quaternion_norm
    )

    rotation_world_camera = (
        Rotation.from_quat(
            quaternion_xyzw
        ).as_matrix()
    )

    transform_world_camera = np.eye(
        4,
        dtype=np.float64,
    )

    transform_world_camera[:3, :3] = (
        rotation_world_camera
    )

    transform_world_camera[:3, 3] = translation

    return transform_world_camera

def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Apply a 4x4 rigid transformation to an Nx3 point cloud.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    # shape checks
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected points shape (N, 3), got {points.shape}"
        )

    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected transform shape (4, 4), got {transform.shape}"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError(
            "Point cloud contains non-finite values."
        )

    points_homogeneous = np.column_stack(
        (
            points,
            np.ones(len(points)),
        )
    )

    points_world_homogeneous = (
        points_homogeneous @ transform.T
    )

    points_world = (
        points_world_homogeneous[:, :3]
    )

    return points_world