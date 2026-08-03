from pathlib import Path

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
        rgbd_pair: RGBDPair,
        pose: PoseEntry,
    ) -> None:
        self.rgbd_pair = rgbd_pair
        self.pose = pose

    def get_pose_time_difference(self) -> float:
        return abs(
            self.rgbd_pair.depth_timestamp
            - self.pose.timestamp
        )

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
def associate_rgb_depth(
    rgb_entries: list[tuple[float, Path]],
    depth_entries: list[tuple[float, Path]],
    max_time_difference: float = 0.03,
) -> list[RGBDPair]:
    pairs: list[RGBDPair] = []
    candidates = []

    for rgb_index, (rgb_timestamp, rgb_path) in enumerate(rgb_entries):
        for depth_index, (depth_timestamp, depth_path) in enumerate(depth_entries):
            difference = abs(rgb_timestamp - depth_timestamp)

            if difference <= max_time_difference:
                candidates.append((difference, rgb_index, depth_index)) # schema: (difference, rgb_index, depth_index)

    candidates.sort(key=lambda item: item[0])
    used_rgb = set()
    used_depth = set()

    for (difference, rgb_index, depth_index) in candidates:
        if rgb_index in used_rgb or depth_index in used_depth:
            continue

        used_rgb.add(rgb_index)
        used_depth.add(depth_index)
        curr_rgb_entry = rgb_entries[rgb_index]
        curr_depth_entry = depth_entries[depth_index]
        curr_pair = RGBDPair(curr_rgb_entry[0], curr_rgb_entry[1], curr_depth_entry[0], curr_depth_entry[1])
        curr_pair.time_difference = difference
        pairs.append(curr_pair)

    pairs.sort(
        key=lambda pair: pair.depth_timestamp
    )

    return pairs

def find_nearest_pose(
    timestamp,
    pose_entries,
):
    nearest_pose = min(
        pose_entries,
        key=lambda pose: abs(
            pose.timestamp - timestamp
        ),
    )

    return nearest_pose

def associate_rgbd_with_poses(
    rgbd_pairs,
    pose_entries,
    max_time_difference=0.02,
):
    records = []

    first_pose_timestamp = pose_entries[0].timestamp
    last_pose_timestamp = pose_entries[-1].timestamp

    for pair in rgbd_pairs:
        timestamp = pair.depth_timestamp

        if (
            timestamp < first_pose_timestamp
            or timestamp > last_pose_timestamp
        ):
            continue

        pose = find_nearest_pose(
            timestamp,
            pose_entries,
        )

        difference = abs(
            timestamp - pose.timestamp
        )

        if difference > max_time_difference:
            print(
                "Pose difference too large:",
                timestamp,
                pose.timestamp,
                difference,
            )
            continue

        records.append(
            RGBDRecord(pair, pose)
        )

    return records

# main function

def main() -> None:
    rgb_entries  = read_data_entries(DATASET_ROOT / "rgb.txt")
    depth_entries = read_data_entries(DATASET_ROOT / "depth.txt")
    groundtruth_entries = read_pose_entries(DATASET_ROOT / "groundtruth.txt")

    pairs = associate_rgb_depth(
        rgb_entries,
        depth_entries,
        max_time_difference=0.03,
    )

    print(f"Matched pairs: {len(pairs)}")

    for pair in pairs[:5]:
        print(
            pair.rgb_timestamp,
            pair.depth_timestamp,
            pair.time_difference,
        )

    differences = np.array(
        [pair.time_difference for pair in pairs],
        dtype=np.float64,
    )

    print(f"Mean time difference: {differences.mean():.6f} s")
    print(f"Max time difference: {differences.max():.6f} s")
    print(f"Median time difference: {np.median(differences):.6f} s")

    records = associate_rgbd_with_poses(
        pairs,
        groundtruth_entries,
    )

    pose_differences = np.array([
        record.get_pose_time_difference()
        for record in records
    ])

    print(f"RGB-D pairs: {len(pairs)}")
    print(f"RGB-D-pose records: {len(records)}")

    print(
        f"Mean pose difference: "
        f"{pose_differences.mean():.6f} s"
    )
    print(
        f"Median pose difference: "
        f"{np.median(pose_differences):.6f} s"
    )
    print(
        f"Max pose difference: "
        f"{pose_differences.max():.6f} s"
    )

    return records


if __name__ == "__main__":
    main()