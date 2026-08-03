from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

# schemas
class RGBDPair:
    def __init__(
        self,
        rgb_timestamp: float,
        rgb_path: Path,
        depth_timestamp: float,
        depth_path: Path,
    ) -> None:
        self.rgb_timestamp = rgb_timestamp
        self.rgb_path = rgb_path
        self.depth_timestamp = depth_timestamp
        self.depth_path = depth_path
        self.time_difference = float("inf")

    def get_time_difference(self) -> float:
        return abs(self.rgb_timestamp - self.depth_timestamp)

class PoseEntry:
    def __init__(
        self,
        timestamp: float,
        translation: np.ndarray,
        quaternion_xyzw: np.ndarray,
    ) -> None:
        self.timestamp = timestamp
        self.translation = translation
        self.quaternion_xyzw = quaternion_xyzw

class RGBDRecord:
    def __init__(
        self,
        rgb_timestamp,
        rgb_path,
        depth_timestamp,
        depth_path,
        pose_timestamp,
        translation,
        quaternion_xyzw,
    ):
        self.rgb_timestamp = rgb_timestamp
        self.rgb_path = rgb_path
        self.depth_timestamp = depth_timestamp
        self.depth_path = depth_path
        self.pose_timestamp = pose_timestamp
        self.translation = translation
        self.quaternion_xyzw = quaternion_xyzw

# loading functions
DATASET_ROOT = Path("data/raw/rgbd_dataset_freiburg1_xyz")
OUTPUT_PATH = Path("outputs/day1/raw_frame_check.png")

def read_data_entries(
    index_file: Path,
) -> list[tuple[float, Path]]:
    """Read all valid timestamp-path entries."""
    res = []
    with index_file.open("r", encoding="utf-8") as file:
        for line in file:
            line_cleaned = line.strip()
            if line_cleaned and not line_cleaned.startswith("#"):
                line_splitted = line_cleaned.split(" ")
                res.append((float(line_splitted[0]), line_splitted[1]))

    if not res:
        raise RuntimeError(f"No entries found in {index_file}")
    
    return res

def read_pose_entries(
    groundtruth_file: Path,
) -> list[PoseEntry]:
    poses: list[PoseEntry] = []

    with groundtruth_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 8:
                raise ValueError(
                    f"Invalid ground-truth line: {line}"
                )

            values = [float(value) for value in parts]

            pose = PoseEntry(
                timestamp=values[0],
                translation=np.array(
                    values[1:4],
                    dtype=np.float64,
                ),
                quaternion_xyzw=np.array(
                    values[4:8],
                    dtype=np.float64,
                ),
            )

            poses.append(pose)

    if not poses:
        raise RuntimeError(
            f"No poses found in {groundtruth_file}"
        )

    return poses

# frame associating functions
def is_better(
    candidate_count: int,
    candidate_error: float,
    current_count: int,
    current_error: float,
) -> bool:
    if candidate_count > current_count:
        return True

    if (
        candidate_count == current_count
        and candidate_error < current_error
    ):
        return True

    return False

def associate_rgb_depth_dp(
    rgb_entries,
    depth_entries,
    max_time_difference: float = 0.03,
):
    MATCH = 1
    SKIP_RGB = 2
    SKIP_DEPTH = 3

    n_rgb = len(rgb_entries)
    n_depth = len(depth_entries)

    match_counts = np.zeros(
        (n_rgb + 1, n_depth + 1),
        dtype=np.int32,
    )

    total_errors = np.zeros(
        (n_rgb + 1, n_depth + 1),
        dtype=np.float64,
    )

    actions = np.zeros(
        (n_rgb + 1, n_depth + 1),
        dtype=np.uint8,
    )

    # 边界状态：只有 RGB，没有 Depth
    for i in range(1, n_rgb + 1):
        actions[i, 0] = SKIP_RGB

    # 边界状态：只有 Depth，没有 RGB
    for j in range(1, n_depth + 1):
        actions[0, j] = SKIP_DEPTH

    for i in range(1, n_rgb + 1):
        rgb_timestamp, _ = rgb_entries[i - 1]

        for j in range(1, n_depth + 1):
            depth_timestamp, _ = depth_entries[j - 1]

            # 候选 1：跳过当前 RGB
            best_count = match_counts[i - 1, j]
            best_error = total_errors[i - 1, j]
            best_action = SKIP_RGB

            # 候选 2：跳过当前 Depth
            candidate_count = match_counts[i, j - 1]
            candidate_error = total_errors[i, j - 1]

            if is_better(
                candidate_count,
                candidate_error,
                best_count,
                best_error,
            ):
                best_count = candidate_count
                best_error = candidate_error
                best_action = SKIP_DEPTH

            # 候选 3：匹配当前 RGB 与 Depth
            difference = abs(
                rgb_timestamp - depth_timestamp
            )

            if difference <= max_time_difference:
                candidate_count = (
                    match_counts[i - 1, j - 1] + 1
                )
                candidate_error = (
                    total_errors[i - 1, j - 1]
                    + difference
                )

                if is_better(
                    candidate_count,
                    candidate_error,
                    best_count,
                    best_error,
                ):
                    best_count = candidate_count
                    best_error = candidate_error
                    best_action = MATCH

            match_counts[i, j] = best_count
            total_errors[i, j] = best_error
            actions[i, j] = best_action

    # TODO：根据 actions 从右下角回溯
    pairs = []

    return pairs
