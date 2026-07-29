from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


DATASET_ROOT = Path("data/raw/rgbd_dataset_freiburg1_xyz")
OUTPUT_PATH = Path("outputs/day1/raw_frame_check.png")


def first_data_entry(index_file: Path) -> tuple[float, Path]:
    """Read the first non-comment data entry from a TUM index file."""

    # TODO 1:
    # Open index_file in read mode.
    # Iterate through its lines.
    # Remove surrounding whitespace from each line.
    # Skip empty lines and lines beginning with "#".
    # Split the first valid line into timestamp and relative path.
    # Return:
    #     float timestamp
    #     Path relative_path
    with index_file.open("r", encoding="utf-8") as file:
        for line in file:
            line_cleaned = line.strip()
            if line_cleaned and not line_cleaned.startswith("#"):
                line_splitted = line_cleaned.split(" ")
                break

    return line_splitted[0], line_splitted[1]


def main() -> None:
    # TODO 2:
    rgb_timestamp, rgb_relative_path = first_data_entry(DATASET_ROOT / "rgb.txt")
    depth_timestamp, depth_relative_path = first_data_entry(DATASET_ROOT / "depth.txt")

    # TODO 3:
    # Combine DATASET_ROOT with each relative path.
    # Read the RGB image and raw depth image using iio.imread().
    rgb_path = DATASET_ROOT / rgb_relative_path
    depth_path = DATASET_ROOT / depth_relative_path

    rgb = iio.imread(rgb_path)
    depth_raw = iio.imread(depth_path)

    # TODO 4:
    # Convert raw depth values into meters.
    # TUM uses a scale factor of 5000.
    # Be careful to convert to float32 before division.
    depth_meters = depth_raw.astype(np.float32) / 5000
    valid_depth = depth_meters[depth_meters > 0]
    
    

    # TODO 5:
    # Print basic information:
    # - timestamps
    # - timestamp difference
    # - shapes
    # - dtypes
    # - valid depth range
    print("=====Basic Information=====")

    print("- timestamps")
    print(f"    rgb timestamp {rgb_timestamp}")
    print(f"    depth timestamp {depth_timestamp}")
    print(f"    stamp difference {abs(float(rgb_timestamp) - float(depth_timestamp))}")

    print("- shapes")
    print(f"    rgb shape {rgb.shape}")
    print(f"    depth shape {depth_raw.shape}")

    print("- dtypes")
    print(f"    rgb dtype {rgb.dtype}")
    print(f"    depth raw dtype {depth_raw.dtype}")
    print(f"    depth in meters dtype {depth_meters.dtype}")

    print("- valid depth range")
    print(f"    {valid_depth.min()} - {valid_depth.max()}")
    print(f"    median {np.median(valid_depth)}")


    # TODO 6:
    # Plot RGB and depth side by side.
    # Save the figure to OUTPUT_PATH.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    depth_upper_bound = np.percentile(valid_depth, 98)

    depth_plot = axes[1].imshow(
        depth_meters,
        vmin=0,
        vmax=depth_upper_bound,
    )

    figure.colorbar(depth_plot, ax=axes[1])

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160)
    plt.close(figure)



if __name__ == "__main__":
    main()