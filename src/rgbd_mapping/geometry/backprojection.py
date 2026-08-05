import numpy as np



def backproject_rgbd(
    rgb: np.ndarray,
    depth_meters: np.ndarray,
    fx: float = 525.0, # official intrinsic parameter
    fy: float = 525.0,
    cx: float = 319.5,
    cy: float = 239.5,
    stride: int = 4,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an RGB-D frame into a colored point cloud.

    Returns:
        points: shape (N, 3), coordinates in camera frame
        colors: shape (N, 3), RGB values in [0, 1]
    """
    # downsampling to speed up the process
    depth_sampled = depth_meters[::stride, ::stride]
    rgb_sampled = rgb[::stride, ::stride]

    height, width = depth_meters.shape
    v_coordinates, u_coordinates = np.mgrid[
        0:height:stride,
        0:width:stride,
    ]

    valid_mask = (
        np.isfinite(depth_sampled)
        & (depth_sampled >= min_depth)
        & (depth_sampled <= max_depth)
    ) # avoid invalid depth

    z = depth_sampled

    x = (u_coordinates - cx) * z / fx # 2 dim 2 3 dim backprojection
    y = (v_coordinates - cy) * z / fy

    # filter the valid points
    points = np.column_stack(
        (
            x[valid_mask],
            y[valid_mask],
            z[valid_mask],
        )
    )

    # get color
    colors = rgb_sampled[valid_mask]
    colors = colors.astype(np.float32) / 255.0 # convert to matplotlib color

    return points, colors

    
def backproject_semantic_rgbd(
    rgb: np.ndarray,
    depth_meters: np.ndarray,
    semantic_labels: np.ndarray,
    semantic_confidence: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stride: int = 4,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Returns:
        points_camera: (N, 3)
        rgb_colors: (N, 3)
        labels: (N,)
        confidence: (N,)
    """
    # backprojection with semantic segmentation process
    depth_sampled = depth_meters[::stride, ::stride]
    rgb_sampled = rgb[::stride, ::stride]

    # semantic segmentation process
    labels_sampled = semantic_labels[::stride, ::stride]
    confidence_sampled = semantic_confidence[::stride, ::stride]
    
    height, width = depth_meters.shape
    v_coordinates, u_coordinates = np.mgrid[
        0:height:stride,
        0:width:stride,
    ]

    valid_mask = (
        np.isfinite(depth_sampled)
        & (depth_sampled >= min_depth)
        & (depth_sampled <= max_depth)
    ) # avoid invalid depth

    z = depth_sampled

    x = (u_coordinates - cx) * z / fx # 2 dim 2 3 dim backprojection
    y = (v_coordinates - cy) * z / fy

    # filter the valid points
    points = np.column_stack(
        (
            x[valid_mask],
            y[valid_mask],
            z[valid_mask],
        )
    )

    # get color
    colors = rgb_sampled[valid_mask]
    colors = colors.astype(np.float32) / 255.0 # convert to matplotlib color
    # get label and confidence
    point_labels = labels_sampled[valid_mask]
    point_confidence = confidence_sampled[valid_mask]

    return points, colors, point_labels, point_confidence