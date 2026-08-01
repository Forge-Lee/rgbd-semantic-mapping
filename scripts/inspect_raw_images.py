from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


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
                res.append((line_splitted[0], line_splitted[1]))

    if not res:
        raise RuntimeError(f"No entries found in {index_file}")
    
    return res

def find_nearest_entry(
    query_timestamp: float,
    candidates: list[tuple[float, Path]],
) -> tuple[float, Path]:
    """Find the candidate whose timestamp is closest to the query."""
    nearest_entry = min(
        candidates,
        key=lambda entry: abs(entry[0] - query_timestamp),
    )
    return nearest_entry

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

def associate_rgb_depth(
    rgb_entries: list[tuple[float, Path]],
    depth_entries: list[tuple[float, Path]],
    max_time_difference: float = 0.03,
) -> list[RGBDPair]:
    pairs: list[RGBDPair] = []

    for rgb in rgb_entries:
        nearest_depth = find_nearest_entry(rgb[0], depth_entries)
        curr_pair = RGBDPair(rgb[0], rgb[1], nearest_depth[0], nearest_depth[1])
        curr_time_diff = curr_pair.get_time_difference()
        if curr_time_diff > max_time_difference:
            continue
        pairs.append(curr_pair)

    return pairs

def associate_rgbd_gt(
    rgbd_pairs: list[RGBDPair],
    gt_entries: list[tuple[float, Path]],
    max_time_difference: float = 0.03,
) -> list[RGBDPair]:
    pairs: list[RGBDPair] = []

    for rgbd in rgbd_pairs:
        nearest_depth = find_nearest_entry(rgbd.rgb_timestamp, gt_entries)
        curr_pair = RGBDPair(rgb[0], rgb[1], nearest_depth[0], nearest_depth[1])
        curr_time_diff = curr_pair.get_time_difference()
        if curr_time_diff > max_time_difference:
            continue
        pairs.append(curr_pair)

    return pairs

def load_rgbd_pair(
    dataset_root: Path,
    pair: RGBDPair,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = iio.imread(dataset_root / pair.rgb_path)
    depth_raw = iio.imread(dataset_root / pair.depth_path)
    depth_meters = depth_raw.astype(np.float32) / 5000.0

    return rgb, depth_meters

def main() -> None:
    rgb_entries  = read_data_entries(DATASET_ROOT / "rgb.txt")
    depth_entries = read_data_entries(DATASET_ROOT / "depth.txt")
    groundtruth_entries = read_data_entries(DATASET_ROOT / "groundtruth.txt")

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



if __name__ == "__main__":
    main()