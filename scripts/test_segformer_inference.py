from pathlib import Path
import time

import imageio.v3 as imageio
import numpy as np

from src.rgbd_mapping.semantics.inference import (
    SemanticSegmenter,
)


DATASET_ROOT = Path(
    "data/raw/rgbd_dataset_freiburg1_xyz"
)

RGB_PATH = (
    DATASET_ROOT
    / "rgb/1305031102.175304.png"
)


def main() -> None:
    rgb = imageio.imread(RGB_PATH)

    # sanity check
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    print("RGB shape:", rgb.shape)
    print("RGB dtype:", rgb.dtype)

    segmenter = SemanticSegmenter(
        device="cpu"
    )

    start_time = time.perf_counter()

    prediction = segmenter.predict(rgb)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    print("Labels shape:", prediction.labels.shape)
    print(
        "Confidence shape:",
        prediction.confidence.shape,
    )

    print(
        "Confidence range:",
        prediction.confidence.min(),
        prediction.confidence.max(),
    )

    print(
        "Inference time:",
        elapsed_seconds,
        "seconds",
    )

    assert prediction.labels.shape == rgb.shape[:2]
    assert (
        prediction.confidence.shape
        == rgb.shape[:2]
    )

    label_ids, pixel_counts = np.unique(
        prediction.labels,
        return_counts=True,
    )

    sorted_indices = np.argsort(
        pixel_counts
    )[::-1]

    print("\nMost frequent predicted classes:")

    for index in sorted_indices[:15]:
        label_id = int(label_ids[index])
        pixel_count = int(pixel_counts[index])

        class_name = segmenter.get_class_name(
            label_id
        )

        pixel_ratio = (
            pixel_count
            / prediction.labels.size
        )

        print(
            f"{label_id:3d} "
            f"{class_name:25s} "
            f"{pixel_count:7d} pixels "
            f"({pixel_ratio:.2%})"
        )

    output_directory = Path(
        "outputs/semantics/single_frame"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_directory / "labels.npy",
        prediction.labels,
    )

    np.save(
        output_directory / "confidence.npy",
        prediction.confidence,
    )


if __name__ == "__main__":
    main()